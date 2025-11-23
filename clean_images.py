import os
from pathlib import Path

# ==== CONFIGURE THESE ====
IMAGES_DIR = Path("D:\\DIP\\Folder1")   # Folder 1: images
LABELS_DIR = Path("D:\\DIP\\Folder2")   # Folder 2: .txt labels
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DRY_RUN = False   # True = just print; False = actually delete
# ==========================

def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"Labels folder not found: {LABELS_DIR}")

    # Get all label stems (filename without extension)
    label_stems = {p.stem for p in LABELS_DIR.glob("*.txt")}

    images_to_delete = []
    images_kept = []

    for img_path in IMAGES_DIR.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if img_path.stem in label_stems:
            images_kept.append(img_path)
        else:
            images_to_delete.append(img_path)

    print(f"Total images found: {len(images_kept) + len(images_to_delete)}")
    print(f"Images with labels (kept): {len(images_kept)}")
    print(f"Images WITHOUT labels (to delete): {len(images_to_delete)}\n")

    if DRY_RUN:
        print("DRY RUN MODE: No files will be deleted.")
        print("Images that WOULD be deleted:")
        for p in images_to_delete:
            print(" -", p)
    else:
        for p in images_to_delete:
            try:
                p.unlink()
                print("Deleted:", p)
            except Exception as e:
                print(f"Error deleting {p}: {e}")

if __name__ == "__main__":
    main()
