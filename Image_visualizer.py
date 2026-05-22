from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Config
# ----------------------------
INPUT_DIR = Path("AIMS_Water_Monitoring/Feature_Output_2")
LABELS_FILE = INPUT_DIR / "cluster_labels.csv"
EMBEDDINGS_FILE = INPUT_DIR / "embeddings.npy"

OUTPUT_DIR = INPUT_DIR / "cluster_visuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THUMB_SIZE = (160, 160)
IMAGES_PER_ROW = 5
MAX_IMAGES_PER_CLUSTER = 25
RANDOM_STATE = 42

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(LABELS_FILE)
embeddings = np.load(EMBEDDINGS_FILE)

if len(df) != len(embeddings):
    raise ValueError(f"Mismatch: {len(df)} labels vs {len(embeddings)} embeddings")

clusters = sorted(df["cluster_label"].unique())

# ----------------------------
# Helper to make montage
# ----------------------------
def make_montage(image_paths, title, out_path):
    n = len(image_paths)
    if n == 0:
        return

    rows = math.ceil(n / IMAGES_PER_ROW)
    fig, axes = plt.subplots(rows, IMAGES_PER_ROW, figsize=(IMAGES_PER_ROW * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, img_path in enumerate(image_paths):
        ax = axes[i]
        try:
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, THUMB_SIZE, method=Image.Resampling.LANCZOS)
            ax.imshow(img)
            ax.set_title(Path(img_path).name, fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{Path(img_path).name}", ha="center", va="center", fontsize=8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ----------------------------
# Make per-cluster montages
# ----------------------------
summary_rows = []
all_cluster_figs = []

for cluster_id in clusters:
    subset = df[df["cluster_label"] == cluster_id].copy()
    image_paths = subset["image_path"].tolist()
    summary_rows.append({"cluster_label": cluster_id, "count": len(image_paths)})

    # sample images if too many
    if len(image_paths) > MAX_IMAGES_PER_CLUSTER:
        image_paths = subset.sample(MAX_IMAGES_PER_CLUSTER, random_state=RANDOM_STATE)["image_path"].tolist()

    out_path = OUTPUT_DIR / f"cluster_{cluster_id}_gallery.png"
    make_montage(image_paths, f"Cluster {cluster_id} ({len(subset)} images)", out_path)
    all_cluster_figs.append(out_path)

# ----------------------------
# Save summary CSV
# ----------------------------
summary_df = pd.DataFrame(summary_rows).sort_values("cluster_label")
summary_df.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

# ----------------------------
# Create combined contact sheet
# ----------------------------
valid_figs = [p for p in all_cluster_figs if p.exists()]
n = len(valid_figs)
if n > 0:
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 8))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, fig_path in enumerate(valid_figs):
        ax = axes[i]
        img = Image.open(fig_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(fig_path.stem, fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_clusters_contact_sheet.png", dpi=300, bbox_inches="tight")
    plt.close()

# ----------------------------
# Optional 2D scatter plot
# ----------------------------
X = StandardScaler().fit_transform(embeddings)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
xy = pca.fit_transform(X)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(xy[:, 0], xy[:, 1], c=df["cluster_label"], cmap="tab20", s=18, alpha=0.85)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Cluster Visualization in 2D")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster_scatter_pca.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved cluster galleries and plots to: {OUTPUT_DIR}")
print(summary_df)