"""Differentiable structural loss used only to update the Scene clock."""

import os

import torch
import torch.nn.functional as F


OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1))
FEATURE_SIZES = ((30, 40), (60, 80), (120, 160))
LOSS_VARIANT = os.environ.get(
    "ED3DGS_JOINT_CLOCK_LOSS_VARIANT", "combined")
LOSS_NAMES = {
    "combined": "0.5*MIND+0.5*polarity_invariant_NGF",
    "mind": "MIND",
    "ngf": "polarity_invariant_NGF",
    "routed_ngf": "multiscale_observable_polarity_invariant_NGF",
}
if LOSS_VARIANT not in LOSS_NAMES:
    raise ValueError(f"invalid joint clock loss variant: {LOSS_VARIANT!r}")


def loss_name():
    return LOSS_NAMES[LOSS_VARIANT]


def _gray(image, size=FEATURE_SIZES[-1]):
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("expected a CHW grayscale or RGB image")
    if image.shape[0] == 1:
        value = image
    else:
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
        value = (image * weights).sum(dim=0, keepdim=True)
    return F.interpolate(value.unsqueeze(0), size=size,
                         mode="bilinear", align_corners=False)[0, 0]


def mind_descriptor(image):
    gray = _gray(image)
    height, width = gray.shape
    source = gray[None, None]
    padded = F.pad(source, (1, 1, 1, 1), mode="replicate")
    distances = []
    for dy, dx in OFFSETS:
        shifted = padded[:, :, 1 + dy:1 + dy + height,
                         1 + dx:1 + dx + width]
        squared = (source - shifted).square()
        smoothed = F.avg_pool2d(
            F.pad(squared, (1, 1, 1, 1), mode="replicate"),
            kernel_size=3, stride=1)[0, 0]
        distances.append(smoothed)
    distance = torch.stack(distances, dim=0)
    denominator = distance.mean(dim=0, keepdim=True) + torch.finfo(distance.dtype).eps
    descriptor = torch.exp(-distance / denominator)
    if not bool(torch.isfinite(descriptor).all()):
        raise FloatingPointError("non-finite MIND descriptor")
    return descriptor


def _normalized_gradient(image, size=FEATURE_SIZES[-1]):
    gray = _gray(image, size=size).unsqueeze(0).unsqueeze(0)
    kernel_x = gray.new_tensor(((-1.0, 0.0, 1.0),
                                (-2.0, 0.0, 2.0),
                                (-1.0, 0.0, 1.0))).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel_x)[0, 0]
    gy = F.conv2d(padded, kernel_y)[0, 0]
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
    return gx / magnitude, gy / magnitude, magnitude


def _ngf_loss(rendered, target, size, observable):
    rendered_gx, rendered_gy, rendered_magnitude = _normalized_gradient(
        rendered, size=size)
    target_gx, target_gy, target_magnitude = _normalized_gradient(
        target, size=size)
    dot = (rendered_gx * target_gx.detach()
           + rendered_gy * target_gy.detach())
    residual = 1.0 - dot.square()
    if not observable:
        return residual.mean()
    rendered_edge = (rendered_magnitude.detach().square() - 1e-6).clamp_min(
        0.0).sqrt()
    target_edge = (target_magnitude.detach().square() - 1e-6).clamp_min(
        0.0).sqrt()
    reliability = torch.minimum(rendered_edge, target_edge)
    reliability = reliability / reliability.mean().clamp_min(1e-6)
    reliability = reliability.clamp(max=4.0)
    return (reliability * residual).sum() / reliability.sum().clamp_min(1e-6)


def image_loss(rendered, observed):
    """Return MIND/NGF alignment loss with gradient only through rendered."""
    if rendered.shape != observed.shape or rendered.ndim != 3:
        raise ValueError("rendered and observed must have the same CHW shape")
    target = observed.detach()
    rendered_mind = mind_descriptor(rendered)
    target_mind = mind_descriptor(target)
    mind = torch.abs(rendered_mind - target_mind).mean()
    ngf = _ngf_loss(
        rendered, target, size=FEATURE_SIZES[-1], observable=False)
    routed_ngf = torch.stack([
        _ngf_loss(rendered, target, size=size, observable=True)
        for size in FEATURE_SIZES
    ]).mean()
    if LOSS_VARIANT == "mind":
        total = mind
    elif LOSS_VARIANT == "ngf":
        total = ngf
    elif LOSS_VARIANT == "routed_ngf":
        total = routed_ngf
    else:
        total = 0.5 * mind + 0.5 * ngf
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("non-finite joint clock loss")
    return total, {
        "mind": mind,
        "ngf": ngf,
        "routed_ngf": routed_ngf,
    }


def batch_image_loss(rendered, observed):
    if rendered.shape != observed.shape or rendered.ndim != 4:
        raise ValueError("rendered and observed must have the same BCHW shape")
    values = []
    minds = []
    ngfs = []
    routed_ngfs = []
    for prediction, target in zip(rendered, observed):
        value, components = image_loss(prediction, target)
        values.append(value)
        minds.append(components["mind"])
        ngfs.append(components["ngf"])
        routed_ngfs.append(components["routed_ngf"])
    return torch.stack(values).mean(), {
        "mind": torch.stack(minds).mean(),
        "ngf": torch.stack(ngfs).mean(),
        "routed_ngf": torch.stack(routed_ngfs).mean(),
    }
