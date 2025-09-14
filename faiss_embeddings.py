import numpy as np
import faiss

# Path to your merged embeddings file
embeddings_path = "merged_embeddings.npy"

# Step 1: Load embeddings
print(f"📂 Loading embeddings from {embeddings_path} ...")
embeddings = np.load(embeddings_path).astype("float32")

print(f"✅ Loaded embeddings shape: {embeddings.shape}")

# Step 2: Build FAISS index
dim = embeddings.shape[1]  # should be 768
index = faiss.IndexFlatL2(dim)   # L2 (cosine similarity also possible with normalization)

print("🔨 Building FAISS index...")
index.add(embeddings)
print(f"✅ FAISS index built with {index.ntotal} vectors of dim {dim}")

# Step 3: Save FAISS index
output_file = "gov_index.faiss"
faiss.write_index(index, output_file)
print(f"💾 Saved FAISS index → {output_file}")
