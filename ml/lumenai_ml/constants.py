WINDOW_MIN_HU = -1000
WINDOW_MAX_HU = 400
ISOTROPIC_SPACING_MM = (1.0, 1.0, 1.0)
DEFAULT_IMAGE_SIZE = 640

YOLO_CLASSES = [
    "solid",
    "part-solid",
    "ground-glass",
]

TEXTURE_TO_CLASS = {
    1: "ground-glass",
    2: "ground-glass",
    3: "part-solid",
    4: "solid",
    5: "solid",
}
