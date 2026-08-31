from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import AppConfig, config_dict
from .models import HeuristicDecision, PatchPlan, PatchRecord, SlideResult


def make_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{os.getpid()}"


class RunLogger:
    def __init__(self, root: Path, run_id: str, config: AppConfig) -> None:
        self.run_id = run_id
        root.mkdir(parents=True, exist_ok=True)
        redirected_log = os.environ.get("EXTRACT_PATCH_LOG_PATH")
        self.path = Path(redirected_log) if redirected_log else root / f"{run_id}.log"
        self._state_dir = root / ".state" / run_id
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._manifest = self._state_dir / "patch_manifest.csv"
        self._failures = self._state_dir / "failures.txt"
        self._signature = self._state_dir / "config.sha256"
        self.config_signature = hashlib.sha256(
            json.dumps(config_dict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            self._signature.exists()
            and self._signature.read_text(encoding="utf-8").strip() != self.config_signature
        ):
            raise ValueError(
                f"Run {run_id!r} already exists with a different resolved configuration"
            )
        manifest_exists = self._manifest.exists() and self._manifest.stat().st_size > 0
        self._manifest_handle = self._manifest.open(
            "a", encoding="utf-8", newline="", buffering=1
        )
        self._manifest_writer = csv.DictWriter(
            self._manifest_handle,
            fieldnames=[
                "slide_id",
                "index",
                "x",
                "y",
                "level",
                "size",
                "output",
                "status",
                "error",
            ],
        )
        if not manifest_exists:
            self._manifest_writer.writeheader()
        self._signature.write_text(self.config_signature + "\n", encoding="utf-8")
        self._configure_logging(config.logging.level)
        self.logger.info("run=%s config_signature=%s", run_id, self.config_signature[:12])

    def _configure_logging(self, level: str) -> None:
        self.logger = logging.getLogger(f"extract_patch.{self.run_id}")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.propagate = False
        if self.logger.handlers:
            return
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(processName)s %(threadName)s %(message)s"
        )
        if os.environ.get("EXTRACT_PATCH_STDOUT_LOGGED") != "1":
            file_handler = logging.FileHandler(self.path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def record_patch(self, slide_id: str, record: PatchRecord) -> None:
        with self._lock:
            self._manifest_writer.writerow({"slide_id": slide_id, **asdict(record)})

    def record_slide(self, result: SlideResult) -> None:
        with self._lock:
            self.logger.info(
                "slide=%s status=%s patches=%d reader=%s heuristic=%s elapsed=%.2fs error=%s",
                result.slide_id,
                result.status,
                result.patch_count,
                result.reader or "-",
                result.heuristic or "-",
                result.elapsed_seconds,
                result.error or "-",
            )
            if result.status == "failed":
                with self._failures.open("a", encoding="utf-8") as handle:
                    handle.write(f"{result.slide_id}\t{result.error or 'unknown error'}\n")

    def completed_outputs(self, slide_id: str) -> set[str]:
        with self._lock:
            self._manifest_handle.flush()
        if not self._manifest.exists():
            return set()
        outputs: set[str] = set()
        with self._manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["slide_id"] == slide_id and row["status"] == "success":
                    outputs.add(row["output"])
        return outputs

    def finalize(self, results: list[SlideResult]) -> dict[str, Any]:
        with self._lock:
            if not self._manifest_handle.closed:
                self._manifest_handle.flush()
                self._manifest_handle.close()
        success = bool(results) and all(result.status in {"success", "skipped"} for result in results)
        summary = {
            "run_id": self.run_id,
            "status": "success" if success else "partial",
            "slides_total": len(results),
            "slides_success": sum(result.status == "success" for result in results),
            "slides_skipped": sum(result.status == "skipped" for result in results),
            "slides_failed": sum(result.status == "failed" for result in results),
            "patches": sum(result.patch_count for result in results),
            "results": [asdict(result) for result in results],
        }
        self.logger.info(
            "summary status=%s slides=%d success=%d failed=%d patches=%d",
            summary["status"],
            summary["slides_total"],
            summary["slides_success"],
            summary["slides_failed"],
            summary["patches"],
        )
        if success:
            shutil.rmtree(self._state_dir, ignore_errors=True)
            try:
                self._state_dir.parent.rmdir()
            except OSError:
                pass
        return summary


def save_overlay(
    thumbnail: Image.Image,
    decision: HeuristicDecision,
    plans: list[PatchPlan],
    destination: Path,
) -> None:
    image = np.asarray(thumbnail.convert("RGB"))
    result = image.copy()
    if decision.region is not None:
        mask = decision.region.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 220, 0), max(2, min(image.shape[:2]) // 300))
    output = Image.fromarray(result)
    draw = ImageDraw.Draw(output)
    if decision.region is not None:
        slide_w, slide_h = decision.region.slide_size
        thumb_w, thumb_h = decision.region.thumbnail_size
        for plan in plans:
            x0 = plan.x * thumb_w / slide_w
            y0 = plan.y * thumb_h / slide_h
            width = plan.read_size * thumb_w / slide_w
            height = plan.read_size * thumb_h / slide_h
            draw.rectangle((x0, y0, x0 + width, y0 + height), outline=(255, 0, 0), width=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination, quality=92)
