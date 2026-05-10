# Wireframes

## Doctor view

```text
+--------------------------------------------------------------------------------------+
| NoduleX                                       [Doctor Console] [Patient Companion]  |
+--------------------------------------------------------------------------------------+
| Studies rail        | Clinical viewer                           | Findings review    |
| - Jane Doe CT       | +--------------------------------------+  | - Solid nodule     |
| - Upload new study  | | CT slice / overlay toggle            |  |   7.8 mm          |
|                     | |                                      |  |   Accept Reject   |
| Upload form         | |   [AI bounding boxes by slice]       |  | - Ground-glass    |
| - Patient metadata  | |                                      |  |   4.2 mm          |
| - DICOM files       | +--------------------------------------+  |   Accept Reject   |
| - Upload            | Study details / count / accession         |                    |
+--------------------------------------------------------------------------------------+
```

## Patient view

```text
+--------------------------------------------------------------------------------------+
| NoduleX                                       [Doctor Console] [Patient Companion]  |
+--------------------------------------------------------------------------------------+
| Study summary       | Patient scan view                          | Patient chat      |
| - Approved nodules  | +--------------------------------------+   | "What is a       |
| - Sizes             | | CT slice / approved overlays only     |   |  lung nodule?"  |
| - Plain language    | |                                      |   | [Ask]           |
|                     | |   [patient-safe annotations]          |   | grounded answer  |
|                     | +--------------------------------------+   | from this study   |
+--------------------------------------------------------------------------------------+
```

## Notes

- The doctor view exposes all AI findings plus accept/reject controls.
- The patient view is filtered to approved findings and uses simpler language.
- The overlay toggle exists to reduce automation bias during clinician review.
