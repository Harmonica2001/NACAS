import random
import shutil
from pathlib import Path

def sample_images(src_dir, dst_dir, k=100, extensions=(".jpg", ".jpeg", ".png", ".gif", ".bmp")):
    """
    Sample k random images from src_dir (non-recursive) and copy them to dst_dir.
    Creates dst_dir if it doesn't exist. Returns list of copied file paths.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    if not files:
        return []

    k = min(k, len(files))
    chosen = random.sample(files, k)
    copied = []
    for f in chosen:
        dest = dst / f.name
        # If name collision, add a numeric suffix
        i = 1
        while dest.exists():
            dest = dst.with_name(f"{f.stem}_{i}{f.suffix}")
            i += 1
        shutil.copy2(f, dest)
        copied.append(dest)
    return copied

copied = sample_images(r"AIMS_Water_Monitoring\Datasets\archive\yolo\images", r"C:\Personal\Masters\Part_time_work\NACAS\Job\AIMS_Water_Monitoring\Clustering_attempt_2", k=1000)
print(f"Copied {len(copied)} images to /path/to/dest_folder")