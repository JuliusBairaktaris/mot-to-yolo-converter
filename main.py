import argparse
import configparser
import os
import random
import shutil
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert MOT dataset directly to YOLO format."
    )
    parser.add_argument(
        "--mot_root_dir",
        required=True,
        help="Path to the root of the MOT dataset (e.g., /path/to/MOT20).",
    )
    parser.add_argument(
        "--yolo_output_dir",
        required=True,
        help="Path to the directory where YOLO formatted data will be saved.",
    )
    # --- MODIFICATION: New arguments for explicit sequence assignment ---
    parser.add_argument(
        "--train_seqs",
        nargs="+",  # Allows multiple sequence names
        default=None,
        help="Space-separated list of sequence names for the training set (e.g., MOT20-01 MOT20-02). Overrides --split_ratio.",
    )
    parser.add_argument(
        "--val_seqs",
        nargs="+",
        default=None,
        help="Space-separated list of sequence names for the validation set (e.g., MOT20-03 MOT20-05). Overrides --split_ratio.",
    )
    # --- END MODIFICATION ---
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to be used for training if --train_seqs and --val_seqs are not provided (e.g., 0.8 for 80% train, 20% val).",
    )
    parser.add_argument(
        "--target_mot_classes",
        nargs="+",
        type=int,
        default=[1],
        help="List of MOT class IDs to include (e.g., [1] for pedestrians). "
        "MOT20 classes: 1:Pedestrian, 2:Person on vehicle, 3:Car, 4:Bicycle, "
        "5:Motorbike, 6:Non-motorized vehicle, 7:Static person, "
        "8:Distractor, 9:Occluder, 10:Occluder on the ground, "
        "11:Occluder full, 12:Reflection",
    )
    parser.add_argument(
        "--yolo_class_id",
        type=int,
        default=0,
        help="The single YOLO class ID to map the target MOT classes to.",
    )
    parser.add_argument(
        "--yolo_class_name",
        type=str,
        default="pedestrian",
        help="The name for the target YOLO class (for dataset.yaml).",
    )
    parser.add_argument(
        "--mot_train_subdir",
        type=str,
        default="train",
        help="Subdirectory name for MOT training data (e.g., 'train').",
    )
    parser.add_argument(
        "--force_output",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )
    return parser.parse_args()


def main(args):
    mot_train_path = os.path.join(args.mot_root_dir, args.mot_train_subdir)
    if not os.path.isdir(mot_train_path):
        print(f"Error: MOT training directory not found at {mot_train_path}")
        return

    if os.path.exists(args.yolo_output_dir):
        if args.force_output:
            print(
                f"Warning: Output directory {args.yolo_output_dir} exists. Overwriting."
            )
            shutil.rmtree(args.yolo_output_dir)
        else:
            print(
                f"Error: Output directory {args.yolo_output_dir} already exists. Use --force_output to overwrite."
            )
            return

    # Create YOLO directory structure
    yolo_images_train_path = os.path.join(args.yolo_output_dir, "images", "train")
    yolo_labels_train_path = os.path.join(args.yolo_output_dir, "labels", "train")
    yolo_images_val_path = os.path.join(args.yolo_output_dir, "images", "val")
    yolo_labels_val_path = os.path.join(args.yolo_output_dir, "labels", "val")

    os.makedirs(yolo_images_train_path, exist_ok=True)
    os.makedirs(yolo_labels_train_path, exist_ok=True)
    os.makedirs(yolo_images_val_path, exist_ok=True)
    os.makedirs(yolo_labels_val_path, exist_ok=True)

    print(f"Processing MOT dataset from: {mot_train_path}")
    print(f"Saving YOLO formatted data to: {args.yolo_output_dir}")
    print(
        f"Targeting MOT classes: {args.target_mot_classes} -> YOLO class ID: {args.yolo_class_id} ({args.yolo_class_name})"
    )

    # --- MODIFICATION: Sequence assignment logic ---
    all_available_sequences = sorted([
        s
        for s in os.listdir(mot_train_path)
        if os.path.isdir(os.path.join(mot_train_path, s))
    ])

    if not all_available_sequences:
        print(f"Error: No sequences found in {mot_train_path}")
        return

    print(f"Available sequences in {mot_train_path}: {all_available_sequences}")

    train_sequences = []
    val_sequences = []

    if args.train_seqs or args.val_seqs:
        print("Using explicitly specified train/validation sequences.")
        # Use provided lists, ensuring they exist
        if args.train_seqs:
            for seq_name in args.train_seqs:
                if seq_name in all_available_sequences:
                    if (
                        seq_name not in train_sequences
                    ):  # Avoid duplicates if user lists same seq twice
                        train_sequences.append(seq_name)
                else:
                    print(
                        f"Warning: Specified training sequence '{seq_name}' not found in {mot_train_path}. Ignoring."
                    )

        if args.val_seqs:
            for seq_name in args.val_seqs:
                if seq_name in all_available_sequences:
                    if seq_name in train_sequences:
                        print(
                            f"Error: Sequence '{seq_name}' is specified in both --train_seqs and --val_seqs. This is not allowed. Exiting."
                        )
                        return
                    if seq_name not in val_sequences:  # Avoid duplicates
                        val_sequences.append(seq_name)
                else:
                    print(
                        f"Warning: Specified validation sequence '{seq_name}' not found in {mot_train_path}. Ignoring."
                    )

        # Check if any valid sequences were actually assigned
        if not train_sequences and not val_sequences:
            print(
                "Error: No valid sequences were assigned from --train_seqs or --val_seqs. Exiting."
            )
            return

    else:
        print(f"Using split_ratio ({args.split_ratio}) for train/validation split.")
        sequences_to_split = list(all_available_sequences)  # Make a copy to shuffle
        random.shuffle(sequences_to_split)
        split_point = int(len(sequences_to_split) * args.split_ratio)
        train_sequences = sequences_to_split[:split_point]
        val_sequences = sequences_to_split[split_point:]
    # --- END MODIFICATION ---

    # Ensure lists are sorted for consistent processing order (optional, but good practice)
    train_sequences.sort()
    val_sequences.sort()

    print(f"Training sequences: {train_sequences if train_sequences else 'None'}")
    print(f"Validation sequences: {val_sequences if val_sequences else 'None'}")

    for split_type, seq_list, img_out_path, lbl_out_path in [
        ("train", train_sequences, yolo_images_train_path, yolo_labels_train_path),
        ("val", val_sequences, yolo_images_val_path, yolo_labels_val_path),
    ]:
        if not seq_list:  # If a list is empty (e.g., no val sequences specified or resulting from split)
            print(
                f"\nNo sequences for {split_type} split. Skipping processing for this split."
            )
            continue

        print(f"\nProcessing {split_type} split ({len(seq_list)} sequences)...")
        for seq_name in seq_list:
            seq_path = os.path.join(mot_train_path, seq_name)
            img_folder = os.path.join(seq_path, "img1")
            gt_file = os.path.join(seq_path, "gt", "gt.txt")
            seqinfo_file = os.path.join(seq_path, "seqinfo.ini")

            # Sanity check if sequence actually exists (should be caught above, but good for robustness)
            if not os.path.isdir(seq_path):
                print(
                    f"Error: Sequence directory {seq_path} for {seq_name} not found. Skipping."
                )
                continue

            if not os.path.exists(gt_file):
                print(
                    f"Warning: Ground truth file not found for sequence {seq_name}. Skipping."
                )
                continue
            if not os.path.exists(seqinfo_file):
                print(
                    f"Warning: seqinfo.ini not found for sequence {seq_name}. Skipping."
                )
                continue

            # Read image dimensions from seqinfo.ini
            config = configparser.ConfigParser()
            config.read(seqinfo_file)
            try:
                im_width = int(config["Sequence"]["imWidth"])
                im_height = int(config["Sequence"]["imHeight"])
            except (KeyError, ValueError) as e:
                print(
                    f"Error reading imWidth/imHeight from {seqinfo_file} for {seq_name}: {e}. Skipping sequence."
                )
                continue

            annotations_per_frame = defaultdict(list)
            with open(gt_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        continue
                    frame_id = int(parts[0])
                    active = int(parts[6])
                    mot_class = int(parts[7])
                    if active == 0 or mot_class not in args.target_mot_classes:
                        continue
                    bb_left = float(parts[2])
                    bb_top = float(parts[3])
                    bb_width = float(parts[4])
                    bb_height = float(parts[5])
                    x_center = (bb_left + bb_width / 2.0) / im_width
                    y_center = (bb_top + bb_height / 2.0) / im_height
                    norm_width = bb_width / im_width
                    norm_height = bb_height / im_height
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_width = max(0.0, min(1.0, norm_width))
                    norm_height = max(0.0, min(1.0, norm_height))
                    yolo_annotation = f"{args.yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                    annotations_per_frame[frame_id].append(yolo_annotation)

            image_files = sorted([
                f
                for f in os.listdir(img_folder)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            processed_frames_count = 0
            for img_filename in image_files:
                try:
                    base_name, _ = os.path.splitext(img_filename)
                    current_frame_id_from_filename = int(base_name)
                    if current_frame_id_from_filename in annotations_per_frame:
                        yolo_annotations_for_frame = annotations_per_frame[
                            current_frame_id_from_filename
                        ]
                        unique_img_name = f"{seq_name}_{img_filename}"
                        unique_lbl_name = f"{seq_name}_{base_name}.txt"
                        shutil.copy2(
                            os.path.join(img_folder, img_filename),
                            os.path.join(img_out_path, unique_img_name),
                        )
                        with open(
                            os.path.join(lbl_out_path, unique_lbl_name), "w"
                        ) as lbl_f:
                            for ann_line in yolo_annotations_for_frame:
                                lbl_f.write(ann_line + "\n")
                        processed_frames_count += 1
                except ValueError:
                    print(
                        f"Warning: Could not parse frame number from filename {img_filename} in {seq_name}. Skipping this image."
                    )
                except Exception as e:
                    print(
                        f"An error occurred processing {img_filename} in {seq_name}: {e}"
                    )

            print(
                f"  Processed sequence: {seq_name} ({processed_frames_count} frames with annotations written to {split_type} output)"
            )

    # Create dataset.yaml
    dataset_yaml_content = f"""
path: {os.path.abspath(args.yolo_output_dir)} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
# test: # test images (optional)

# Classes
names:
  {args.yolo_class_id}: {args.yolo_class_name}
"""
    with open(os.path.join(args.yolo_output_dir, "dataset.yaml"), "w") as f:
        f.write(dataset_yaml_content)

    print("\nConversion complete!")
    print(f"YOLO dataset created at: {args.yolo_output_dir}")
    print(
        f"dataset.yaml created at: {os.path.join(args.yolo_output_dir, 'dataset.yaml')}"
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
