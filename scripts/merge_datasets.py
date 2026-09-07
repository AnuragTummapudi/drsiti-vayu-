from pathlib import Path
import shutil

ROOT = Path.home() / "mine-yolo"
RAW = ROOT / "raw_datasets"
OUT = ROOT / "unified_dataset"

MAPPINGS = {
    "Dataset_1": {
        0: 0,  # Dump Truck -> dump_truck
        1: 1,  # Excavator -> excavator
        2: 2,  # HD-Truck -> hd_truck
    },
    "Dataset_2": {
        0: 3,  # pile -> pile
    },
    "Dataset_3": {
        1: 0,  # dump_truck -> dump_truck
        3: 4,  # dust -> dust
        7: 5,  # mining_truck -> mining_truck
        8: 5,  # mining_truck -> mining_truck
    },
}

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def convert_annotation(line, class_map):
    parts = line.split()

    if len(parts) < 5:
        return None

    old_class = int(parts[0])

    if old_class not in class_map:
        return None

    new_class = class_map[old_class]
    coords = list(map(float, parts[1:]))

    # Standard YOLO bounding box:
    # class x_center y_center width height
    if len(coords) == 4:
        return f"{new_class} " + " ".join(parts[1:])

    # Polygon:
    # class x1 y1 x2 y2 x3 y3 ...
    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return f"{new_class} {x_center} {y_center} {width} {height}"

    return None


# Clear previous incomplete merge
for split in ["train", "val", "test"]:
    for folder in [
        OUT / "images" / split,
        OUT / "labels" / split,
    ]:
        for p in folder.iterdir():
            if p.is_file():
                p.unlink()

for dataset, class_map in MAPPINGS.items():

    for source_split, output_split in SPLITS.items():

        image_dir = RAW / dataset / source_split / "images"
        label_dir = RAW / dataset / source_split / "labels"

        if not image_dir.exists():
            continue

        out_images = OUT / "images" / output_split
        out_labels = OUT / "labels" / output_split

        for image_path in image_dir.iterdir():

            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                print(f"WARNING: missing label: {image_path.name}")
                continue

            new_lines = []

            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue

                converted = convert_annotation(line, class_map)

                if converted:
                    new_lines.append(converted)

            if not new_lines:
                continue

            new_name = f"{dataset}_{image_path.name}"

            shutil.copy2(
                image_path,
                out_images / new_name
            )

            (out_labels / f"{Path(new_name).stem}.txt").write_text(
                "\n".join(new_lines) + "\n"
            )

            print(f"Added: {dataset}/{source_split}/{image_path.name}")

print("\nMERGE COMPLETE")
