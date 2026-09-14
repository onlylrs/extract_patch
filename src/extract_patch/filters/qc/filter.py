from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from ...models import PatchPlan
from ..base import Config, FilterDecision, PatchFilter, register
from .runtime import QcFilterSetupError, get_qc_client


@register
class QcPatchFilter(PatchFilter):
    """Keep patches whose MobileNet QC keep-probability meets the threshold."""

    name = "qc"

    def __call__(
        self,
        image: Image.Image,
        plan: PatchPlan,
        config: Config | None = None,
    ) -> FilterDecision:
        try:
            return self.evaluate(image, plan, config)
        except QcFilterSetupError:
            raise
        except Exception as exc:
            return FilterDecision(
                keep=False,
                name=self.name,
                reason=f"{type(exc).__name__}: {exc}",
                metrics={"exception_type": type(exc).__name__},
            )

    def evaluate(
        self,
        image: Image.Image,
        plan: PatchPlan,
        config: Config | None = None,
    ) -> FilterDecision:
        return self.evaluate_many(((image, plan),), config)[0]

    def evaluate_many(
        self,
        items: Sequence[tuple[Image.Image, PatchPlan]],
        config: Config | None = None,
    ) -> list[FilterDecision]:
        if not items:
            return []
        client = get_qc_client()
        if client is None:
            raise QcFilterSetupError(
                "QC service is not running; start extract/preview with qc in post_filter_pipe"
            )
        scores = client.predict([image for image, _plan in items])
        decisions: list[FilterDecision] = []
        for score in scores:
            keep = score >= client.threshold
            decisions.append(
                FilterDecision(
                    keep=keep,
                    name=self.name,
                    reason=(
                        "QC classifier accepted the patch"
                        if keep
                        else "QC classifier rejected the patch"
                    ),
                    metrics={
                        "score": score,
                        "threshold": client.threshold,
                        "model_version": client.version,
                        "device": client.device,
                    },
                )
            )
        return decisions


def qc(
    image: Image.Image,
    plan: PatchPlan,
    config: Config | None = None,
) -> FilterDecision:
    return QcPatchFilter()(image, plan, config)
