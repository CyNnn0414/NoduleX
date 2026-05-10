# ML Pipeline

This folder contains the YOLO-centered lung nodule detection pipeline.

## Workflow

1. Load DICOM CT series
2. Convert raw pixels to Hounsfield Units
3. Apply lung windowing with `[-1000, 400]`
4. Resample to isotropic 1 mm spacing
5. Convert 3D volumes into 2D slices
6. Normalize contrast to `[0, 1]`
7. Resize slices to YOLO input size
8. Convert grayscale slices to 3-channel RGB PNGs
9. Parse LIDC-IDRI XML annotations
10. Convert contour points to 2D bounding boxes
11. Filter nodules below 3 mm
12. Save aligned YOLO label files
13. Train on 70% of patients and test on 30%

## Class mapping

- `ground-glass`
- `part-solid`
- `solid`

The initial class mapping uses the LIDC `texture` score as a practical baseline. You can replace that mapping later with a richer pathology taxonomy if your annotation strategy changes.
