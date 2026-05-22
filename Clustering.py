from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Config
# ----------------------------
INPUT_DIR = Path("AIMS_Water_Monitoring/Feature_Output_2")
EMBEDDINGS_FILE = INPUT_DIR / "embeddings.npy"
PATHS_FILE = INPUT_DIR / "image_paths.csv"

OUTPUT_FILE = INPUT_DIR / "cluster_labels.csv"
ELBOW_PLOT_FILE = INPUT_DIR / "elbow_plot.png"
ELBOW_DATA_FILE = INPUT_DIR / "elbow_data.csv"
CLUSTER_COUNTS_FILE = INPUT_DIR / "cluster_counts.csv"

RANDOM_STATE = 42
K_MIN = 1
K_MAX = 15
N_CLUSTERS = 4 # set this to the k you choose after viewing the elbow plot

# ----------------------------
# Load data
# ----------------------------
embeddings = np.load(EMBEDDINGS_FILE)
paths_df = pd.read_csv(PATHS_FILE)
image_paths = paths_df["image_path"].astype(str).tolist()

if len(image_paths) != len(embeddings):
    raise ValueError(f"Mismatch: {len(image_paths)} paths vs {len(embeddings)} embeddings")

# ----------------------------
# Scale embeddings
# ----------------------------
X = StandardScaler().fit_transform(embeddings)

# ----------------------------
# Elbow plot
# ----------------------------
ks = list(range(K_MIN, K_MAX + 1))
inertias = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    km.fit(X)
    inertias.append(km.inertia_)

elbow_df = pd.DataFrame({"k": ks, "inertia": inertias})
elbow_df.to_csv(ELBOW_DATA_FILE, index=False)

plt.figure(figsize=(8, 5))
plt.plot(ks, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Plot for K-Means")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ELBOW_PLOT_FILE, dpi=300)
plt.close()

# ----------------------------
# Final clustering with chosen k
# ----------------------------
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
labels = kmeans.fit_predict(X)

out = pd.DataFrame({
    "image_path": image_paths,
    "cluster_label": labels
})
out.to_csv(OUTPUT_FILE, index=False)

counts = out["cluster_label"].value_counts().sort_index()
counts.to_csv(CLUSTER_COUNTS_FILE, header=["count"])

print(f"Saved elbow plot to {ELBOW_PLOT_FILE}")
print(f"Saved elbow data to {ELBOW_DATA_FILE}")
print(f"Saved cluster labels to {OUTPUT_FILE}")
print("Cluster counts:")
print(counts)