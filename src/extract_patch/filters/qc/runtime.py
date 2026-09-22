from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .model import build_model, preprocessing

ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_CHECKPOINT = ASSETS / "qc_20260912T072309Z.pt"
MANIFEST_PATH = ASSETS / "manifest.json"

_CLIENT: QcClient | None = None


def _images_to_uint8_batch(images: Sequence[Image.Image], input_size: int):
    """Resize exactly like the legacy torchvision pipeline, without float expansion."""
    import numpy as np

    arrays = []
    for image in images:
        resized = image.convert("RGB").resize(
            (input_size, input_size),
            resample=Image.Resampling.BILINEAR,
        )
        arrays.append(np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(arrays), dtype=np.uint8)


def _normalize_uint8_batch(batch):
    """Reproduce ToTensor + Normalize bit-for-bit after uint8 IPC transport."""
    import numpy as np
    import torch
    from torchvision.transforms import functional as functional

    array = np.ascontiguousarray(batch)
    if array.dtype != np.uint8:
        raise QcFilterSetupError(f"QC uint8 batch has dtype {array.dtype}, expected uint8")
    tensor = torch.from_numpy(array)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise QcFilterSetupError(
            f"QC uint8 batch must be NCHW with 3 channels, got {tuple(tensor.shape)}"
        )
    tensor = tensor.to(dtype=torch.get_default_dtype()).div(255)
    return functional.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


class QcFilterSetupError(RuntimeError):
    """Raised when the qc filter cannot be initialized or reached."""


def default_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def resolve_checkpoint(path: str | Path | None) -> Path:
    if path in (None, "", "null"):
        return DEFAULT_CHECKPOINT
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    bundled = ASSETS / candidate.name
    if bundled.is_file():
        return bundled
    raise QcFilterSetupError(f"qc checkpoint not found: {candidate}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_qc_client(client: "QcClient | None") -> None:
    global _CLIENT
    _CLIENT = client


def get_qc_client() -> "QcClient | None":
    return _CLIENT


class QcSession:
    def __init__(
        self,
        checkpoint: Path,
        device,
        threshold_override: float | None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise QcFilterSetupError(
                "The qc post-filter requires PyTorch. Install with: pip install torch torchvision"
            ) from exc

        if not checkpoint.is_file():
            raise QcFilterSetupError(f"qc checkpoint not found: {checkpoint}")
        expected = default_manifest()
        if checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve():
            digest = _file_sha256(checkpoint)
            if digest != expected["sha256"]:
                raise QcFilterSetupError(
                    f"Bundled qc checkpoint sha256 mismatch: {digest}"
                )

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("model_name") not in (None, "mobilenet_v3_large"):
            raise QcFilterSetupError(f"Unsupported qc model: {payload.get('model_name')}")
        self.input_size = int(payload.get("input_size") or expected["input_size"])
        self.checkpoint_threshold = float(
            payload.get("threshold", expected["threshold"])
        )
        self.threshold = (
            float(threshold_override)
            if threshold_override is not None
            else self.checkpoint_threshold
        )
        self.version = str(payload.get("version") or expected["model_version"])
        self.device = device
        self.model = build_model(num_classes=2)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(device).eval()
        self.transform = preprocessing(self.input_size)
        self._torch = torch

    def _predict_tensor(self, tensor) -> list[float]:
        torch = self._torch
        tensor = tensor.to(self.device, non_blocking=self.device.type == "cuda")
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[:, 1]
        return [float(value) for value in probabilities.detach().cpu()]

    def predict_numpy(self, batch) -> list[float]:
        import numpy as np

        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(batch, dtype="float32"))
        if tensor.ndim != 4:
            raise QcFilterSetupError(f"QC batch must be NCHW, got {tuple(tensor.shape)}")
        return self._predict_tensor(tensor)

    def predict_uint8(self, batch) -> list[float]:
        return self._predict_tensor(_normalize_uint8_batch(batch))

    def predict(self, images: Sequence[Image.Image]) -> list[float]:
        if not images:
            return []
        tensors = [self.transform(image.convert("RGB")) for image in images]
        import numpy as np

        return self.predict_numpy(np.stack([tensor.numpy() for tensor in tensors]))


class QcClient:
    """CPU-side client that sends compact uint8 batches to the shared QC process."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        authkey: bytes,
        input_size: int,
        threshold: float,
        version: str,
        device: str,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.authkey = bytes(authkey)
        self.input_size = int(input_size)
        self.threshold = float(threshold)
        self.version = str(version)
        self.device = str(device)
        self._conn = None
        self._lock = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_conn"] = None
        state["_lock"] = None
        return state

    def _lock_obj(self):
        if self._lock is None:
            import threading

            self._lock = threading.Lock()
        return self._lock

    def _connection(self):
        if self._conn is None:
            from multiprocessing.connection import Client

            self._conn = Client((self.host, self.port), authkey=self.authkey)
        return self._conn

    def predict(self, images: Sequence[Image.Image]) -> list[float]:
        if not images:
            return []
        batch = _images_to_uint8_batch(images, self.input_size)
        with self._lock_obj():
            try:
                conn = self._connection()
                conn.send(batch)
                payload = conn.recv()
            except (EOFError, OSError, ConnectionError) as exc:
                self._conn = None
                raise QcFilterSetupError("QC service closed while a batch was in flight") from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise QcFilterSetupError(str(payload["error"]))
        scores = list(payload)
        if len(scores) != len(images):
            raise QcFilterSetupError(
                f"QC service returned {len(scores)} scores for {len(images)} patches"
            )
        return [float(score) for score in scores]
