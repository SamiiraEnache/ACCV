import torch
import timm


_DROPOUT_MODEL_NAMES = {
    "efficientnetv2_s",
    "convnext_tiny",
    "swin_tiny_patch4_window7_224",
}


def _is_head(name):
    parts = name.lower().split(".")
    return any(part in {"classifier", "fc", "head"} for part in parts)


def build_model(model_name, n_classes, drop_rate=0.2, drop_path_rate=0.2):
    kwargs = {
        "pretrained": True,
        "num_classes": n_classes,
        "in_chans": 3,
    }

    if model_name in _DROPOUT_MODEL_NAMES:
        kwargs["drop_rate"] = drop_rate
        kwargs["drop_path_rate"] = drop_path_rate

    return timm.create_model(model_name, **kwargs)


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = _is_head(name)
    return model


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_optimizer_phase2(model, backbone_lr, head_lr, weight_decay):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_head(name):
            head_params.append(param)
        else:
            backbone_params.append(param)

    assert head_params, "No head parameters found. Expected classifier, fc, or head parameters."

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
