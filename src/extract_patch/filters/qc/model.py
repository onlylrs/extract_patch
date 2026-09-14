from __future__ import annotations


def _require_torchvision():
    try:
        import torch.nn as nn
        from torchvision.models import mobilenet_v3_large
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ImportError as exc:
        raise RuntimeError(
            "The qc post-filter requires PyTorch. Install with: pip install torch torchvision"
        ) from exc
    return nn, mobilenet_v3_large, transforms, InterpolationMode


def build_model(num_classes: int = 2):
    nn, mobilenet_v3_large, _transforms, _interpolation = _require_torchvision()
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def preprocessing(input_size: int):
    _nn, _mobilenet, transforms, interpolation = _require_torchvision()
    return transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size), interpolation=interpolation.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
