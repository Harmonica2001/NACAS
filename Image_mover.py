from pathlib import Path
import shutil
import pandas as pd

# ----------------------------
# Config
# ----------------------------
CSV_FILE = Path("AIMS_Water_Monitoring\\Feature_Output_2\\cluster_labels.csv")   # your CSV with 2 columns
BASE_IMAGE_DIR = Path(".")              # root folder for relative paths
OUTPUT_DIR = Path("AIMS_Water_Monitoring\\Feature_Output_2\\cluster_folders")    # destination root

# If your CSV has no header, set this to False
HAS_HEADER = True

# Optional: column names if CSV has headers
IMAGE_COL = "image_path"
CLUSTER_COL = "cluster_label"

# ----------------------------
# Load CSV
# ----------------------------
if HAS_HEADER:
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.read_csv(CSV_FILE, header=None, names=[IMAGE_COL, CLUSTER_COL])

if IMAGE_COL not in df.columns or CLUSTER_COL not in df.columns:
    raise ValueError(f"CSV must contain columns '{IMAGE_COL}' and '{CLUSTER_COL}'")

# ----------------------------
# Create output folders
# ----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cluster_ids = sorted(df[CLUSTER_COL].dropna().unique())

for cid in cluster_ids:
    (OUTPUT_DIR / f"cluster_{int(cid)}").mkdir(parents=True, exist_ok=True)

# ----------------------------
# Copy files
# ----------------------------
copied = 0
missing = []

for _, row in df.iterrows():
    rel_path = str(row[IMAGE_COL])
    cluster_id = int(row[CLUSTER_COL])

    src = BASE_IMAGE_DIR / Path(rel_path)
    dst_dir = OUTPUT_DIR / f"cluster_{cluster_id}"
    dst = dst_dir / src.name

    if not src.exists():
        missing.append(str(src))
        continue

    shutil.copy2(src, dst)
    copied += 1

# ----------------------------
# Report
# ----------------------------
print(f"Copied {copied} images into {OUTPUT_DIR}")
if missing:
    print(f"Missing files: {len(missing)}")
    for m in missing[:20]:
        print(m)