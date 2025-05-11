# MOT to YOLO Annotation Converter

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python script to convert Multiple Object Tracking (MOT) datasets (such as MOT17, MOT20) directly into the YOLO (You Only Look Once) object detection format.

## Understanding MOT and YOLO Formats

### MOT (Multiple Object Tracking) Format

*   **`gt/gt.txt`**: Contains ground truth annotations. Each line typically represents an object instance:
    `<frame_id>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <active/conf>, <class_id>, <visibility>`
*   **`seqinfo.ini`**: Provides metadata for each sequence, including crucial `imWidth` and `imHeight` for normalization.
*   **Image Files**: Usually located in `img1/` directory within each sequence folder (e.g., `000001.jpg`).

### YOLO Format

*   **Label Files (`.txt`)**: One per image, with the same name. Each line represents an object:
    `<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`
    (Coordinates are normalized between 0 and 1).
*   **`dataset.yaml`**: Defines paths to train/validation data and class names.

## Prerequisites

*   Python 3.6 or higher.
*   No external libraries are required (uses `os`, `shutil`, `argparse`, `random`, `collections`, `configparser`).
*   Your MOT dataset should follow the standard directory structure (e.g., `MOT20/train/MOT20-01/gt/gt.txt`, `MOT20/train/MOT20-01/img1/`, `MOT20/train/MOT20-01/seqinfo.ini`).

## Usage

Run the script from the command line, providing the necessary arguments.

```bash
python mot_to_yolo_converter.py --mot_root_dir /path/to/MOT_dataset --yolo_output_dir /path/to/YOLO_output [OPTIONS]
```

### Command-Line Arguments

| Argument                | Required | Default      | Description                                                                                                                                                                                               |
| :---------------------- | :------- | :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--mot_root_dir`        | Yes      | N/A          | Path to the root of the MOT dataset (e.g., `/path/to/MOT20`).                                                                                                                                             |
| `--yolo_output_dir`     | Yes      | N/A          | Path to the directory where YOLO formatted data will be saved.                                                                                                                                            |
| `--train_seqs`          | No       | `None`       | Space-separated list of sequence names for the training set (e.g., `MOT20-01 MOT20-02`). Overrides `--split_ratio`.                                                                                       |
| `--val_seqs`            | No       | `None`       | Space-separated list of sequence names for the validation set (e.g., `MOT20-03 MOT20-05`). Overrides `--split_ratio`.                                                                                      |
| `--split_ratio`         | No       | `0.8`        | Ratio of data for training if `--train_seqs` and `--val_seqs` are not provided (e.g., `0.8` for 80% train, 20% val).                                                                                   |
| `--target_mot_classes`  | No       | `[1]`        | List of MOT class IDs to include (e.g., `1` for pedestrians). See MOT Class IDs section below.                                                                                                            |
| `--yolo_class_id`       | No       | `0`          | The single YOLO class ID (0-indexed) to map the target MOT classes to.                                                                                                                                    |
| `--yolo_class_name`     | No       | `pedestrian` | The name for the target YOLO class (for `dataset.yaml`).                                                                                                                                                    |
| `--mot_train_subdir`    | No       | `train`      | Subdirectory name for MOT training data (e.g., 'train', 'MOT17/train').                                                                                                                                   |
| `--force_output`        | No       | `False`      | (Flag) Overwrite output directory if it exists.                                                                                                                                                           |

### Examples

1.  **Basic Conversion (MOT20 pedestrians to YOLO class 0):**
    This will use 80% of sequences for training and 20% for validation, targeting MOT class 1 (Pedestrian).
    ```bash
    python mot_to_yolo_converter.py \
        --mot_root_dir /data/MOT20 \
        --yolo_output_dir /data/MOT20_yolo
    ```

2.  **Convert specific MOT classes (e.g., Pedestrians and Cars from MOT20) to a single YOLO class "object":**
    MOT20 classes: Pedestrian (1), Car (3).
    ```bash
    python mot_to_yolo_converter.py \
        --mot_root_dir /data/MOT20 \
        --yolo_output_dir /data/MOT20_objects_yolo \
        --target_mot_classes 1 3 \
        --yolo_class_id 0 \
        --yolo_class_name "object"
    ```

3.  **Explicitly define training and validation sequences:**
    ```bash
    python mot_to_yolo_converter.py \
        --mot_root_dir /data/MOT20 \
        --yolo_output_dir /data/MOT20_yolo_custom_split \
        --train_seqs MOT20-01 MOT20-02 MOT20-06 \
        --val_seqs MOT20-03 MOT20-05 \
        --target_mot_classes 1 \
        --yolo_class_id 0 \
        --yolo_class_name "pedestrian"
    ```

4.  **Overwrite existing output directory:**
    ```bash
    python mot_to_yolo_converter.py \
        --mot_root_dir /data/MOT20 \
        --yolo_output_dir /data/MOT20_yolo \
        --force_output
    ```

## Output Directory Structure

The script will create the following structure in your `yolo_output_dir`:

```
<yolo_output_dir>/
├── images/
│   ├── train/
│   │   ├── <seq_name>_<frame_name>.jpg  (e.g., MOT20-01_000001.jpg)
│   │   └── ...
│   └── val/
│       ├── <seq_name>_<frame_name>.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── <seq_name>_<frame_name>.txt  (e.g., MOT20-01_000001.txt)
│   │   └── ...
│   └── val/
│       ├── <seq_name>_<frame_name>.txt
│       └── ...
└── dataset.yaml
```

### Example `dataset.yaml`

The generated `dataset.yaml` will look like this (paths will be absolute):

```yaml
path: /path/to/your/yolo_output_dir # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
# test: # test images (optional)

# Classes
names:
  0: pedestrian # or your specified --yolo_class_name
```

## MOT Class IDs (Example from MOT20)

The `--target_mot_classes` argument uses numeric IDs. For MOT20, these are:

*   `1`: Pedestrian
*   `2`: Person on vehicle
*   `3`: Car
*   `4`: Bicycle
*   `5`: Motorbike
*   `6`: Non-motorized vehicle
*   `7`: Static person
*   `8`: Distractor
*   `9`: Occluder
*   `10`: Occluder on the ground
*   `11`: Occluder full
*   `12`: Reflection

*Always refer to the specific MOT dataset's documentation for accurate class IDs.*

## Important Considerations

*   **Single YOLO Class Output**: This script is designed to map one or more MOT classes to a *single* YOLO class ID. If you need to map different MOT classes to different YOLO classes (e.g., MOT pedestrians to YOLO class 0, MOT cars to YOLO class 1), you would need to run the script multiple times with different configurations or modify the script.
*   **Image Filenames**: The script assumes image filenames in the `img1` directory are numeric and correspond to frame IDs (e.g., `000001.jpg`, `000002.png`). This is standard for MOT datasets.
*   **`gt.txt` Format**: Assumes the standard MOT `gt.txt` format where columns 7 (`active`) and 8 (`mot_class`) are 1-indexed, and bounding box coordinates are in columns 3-6.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you add one). If no LICENSE file is present, you can state:
Distributed under the MIT License. See `LICENSE` for more information.
(Or simply omit this if you don't intend to add a formal license file to your repo initially).
