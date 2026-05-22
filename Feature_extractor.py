from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

# ----------------------------
# Config
# ----------------------------
IMAGE_DIR = Path("AIMS_Water_Monitoring\Clustering_attempt_2")          # folder with images
OUTPUT_DIR = Path("AIMS_Water_Monitoring\Feature_Output_2")         # where exports go
MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 16

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load pretrained model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ----------------------------
# Collect image files
# ----------------------------
image_paths = sorted([
    p for p in IMAGE_DIR.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
])

if not image_paths:
    raise ValueError(f"No images found in {IMAGE_DIR}")

# ----------------------------
# Extract embeddings
# ----------------------------
all_embeddings = []
all_files = []

with torch.no_grad():
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            emb = outputs.last_hidden_state[:, 0, :]

        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy())
        all_files.extend([str(p) for p in batch_paths])

embeddings = np.vstack(all_embeddings)

# ----------------------------
# Save exports
# ----------------------------
np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

df = pd.DataFrame({
    "image_path": all_files
})
df["embedding_index"] = range(len(df))
df.to_csv(OUTPUT_DIR / "image_paths.csv", index=False)

emb_df = pd.DataFrame(embeddings)
emb_df.insert(0, "image_path", all_files)
emb_df.to_csv(OUTPUT_DIR / "embeddings.csv", index=False)

print(f"Saved {len(image_paths)} embeddings to {OUTPUT_DIR}")
print(f"Embedding shape: {embeddings.shape}")