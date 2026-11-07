import shutil
from pathlib import Path

SOURCE_DIR = Path("data/raw")
DEST_DIR = Path("data/processed")

SPLITS = ["Training", "Testing"]

TUMOR_CLASSES = [
    "glioma",
    "meningioma",
    "pituitary"
]

NO_TUMOR_CLASS = "notumor"


def copy_images(source, destination):
    destination.mkdir(parents=True, exist_ok=True)

    images = [
        file for file in source.iterdir()
        if file.is_file()
    ]

    for image in images:
        target = destination / image.name

        # Avoid filename collisions between tumor categories
        if target.exists():
            target = destination / f"{source.name}_{image.name}"

        shutil.copy2(image, target)

    return len(images)


def prepare_dataset():
    for split in SPLITS:
        print(f"\nPreparing {split} dataset...")

        tumor_count = 0

        for category in TUMOR_CLASSES:
            source = SOURCE_DIR / split / category
            destination = DEST_DIR / split / "Tumor"

            count = copy_images(source, destination)
            tumor_count += count

            print(f"{category}: {count} images copied")

        no_tumor_source = (
            SOURCE_DIR / split / NO_TUMOR_CLASS
        )

        no_tumor_destination = (
            DEST_DIR / split / "No_Tumor"
        )

        no_tumor_count = copy_images(
            no_tumor_source,
            no_tumor_destination
        )

        print(f"Tumor total: {tumor_count}")
        print(f"No Tumor total: {no_tumor_count}")

    print("\nDataset preparation completed!")


if __name__ == "__main__":
    prepare_dataset()
