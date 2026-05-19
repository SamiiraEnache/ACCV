from torchvision import transforms

from configs.dataset_configs import NO_VFLIP_DATASETS


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RepeatGrayscaleChannels:
    def __call__(self, x):
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


def _post_tensor_transforms(is_rgb):
    steps = [transforms.ToTensor()]
    if not is_rgb:
        steps.append(RepeatGrayscaleChannels())
    steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return steps


def get_train_transform(image_size, level, is_rgb, dataset_name):
    level = level.lower()
    if level not in {"light", "medium", "heavy"}:
        raise ValueError("level must be one of: light, medium, heavy")

    no_vertical_flip = dataset_name in NO_VFLIP_DATASETS
    rotation_degrees = 10
    if not no_vertical_flip and level == "medium":
        rotation_degrees = 20
    elif not no_vertical_flip and level == "heavy":
        rotation_degrees = 25

    steps = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=rotation_degrees),
    ]

    if not no_vertical_flip and level == "medium":
        steps.append(transforms.RandomVerticalFlip(p=0.3))
    elif not no_vertical_flip and level == "heavy":
        steps.append(transforms.RandomVerticalFlip(p=0.5))

    if level == "medium" and is_rgb:
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))

    if level == "heavy":
        steps.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)))
        if is_rgb:
            steps.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05))

    steps.extend(_post_tensor_transforms(is_rgb))
    if level == "heavy":
        steps.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)))
    return transforms.Compose(steps)


def get_val_transform(image_size, is_rgb):
    steps = [transforms.Resize((image_size, image_size))]
    steps.extend(_post_tensor_transforms(is_rgb))
    return transforms.Compose(steps)
