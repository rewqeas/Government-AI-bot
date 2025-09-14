import fitz
import numpy as np
import google.generativeai as genai
import os
import pickle
from rich.progress import Progress
from gemini_client import GEMINI_API_KEY

# ----------------------------
# Gemini API setup
# ----------------------------
# GEMINI_API_KEY = "YOUR_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------
# PDF text loader
# ----------------------------
def pdf_loader(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ----------------------------
# Chunking helper
# ----------------------------
def chunk_file(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# ----------------------------
# Embedding helper
# ----------------------------
def embed_chunks(chunks):
    embeddings = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Embedding chunks...", total=len(chunks))

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                embeddings.append(np.zeros(768))  # fallback vector
                progress.update(task, advance=1)
                continue

            chunk = chunk[:3000]

            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",  # 768-dim model
                    content=chunk,
                    task_type="retrieval_document"
                )
                emb = response["embedding"]
                if isinstance(emb, dict) and "values" in emb:
                    emb = emb["values"]
                emb = np.array(emb, dtype="float32")
                embeddings.append(emb)
            except Exception as e:
                print(f"⚠️ Error generating embedding for chunk {i}: {e}")
                embeddings.append(np.zeros(768))

            progress.update(task, advance=1)

    return np.array(embeddings, dtype="float32")

# ----------------------------
# Paths
# ----------------------------
pdf_folder = "AI_Goverment_Services"
embedding_folder = "embeddings_for_faiss"
os.makedirs(embedding_folder, exist_ok=True)

all_embeddings = []
all_chunks = []

# ----------------------------
# Process PDFs
# ----------------------------
for filename in os.listdir(pdf_folder):
    if not filename.lower().endswith(".pdf"):
        continue

    file_path = os.path.join(pdf_folder, filename)
    base_name = os.path.splitext(filename)[0]
    embedding_path = os.path.join(embedding_folder, f"{base_name}_embeddings.npy")

    if os.path.exists(embedding_path):
        print(f"⏭️ Skipping {filename}, embeddings already exist.")
        continue   

    print(f"\n📄 Processing: {filename}")

    text = pdf_loader(file_path)
    chunks = chunk_file(text, chunk_size=300, overlap=50)
    embeddings = embed_chunks(chunks)

    if embeddings.shape[0] == 0:
        print(f"⚠️ No embeddings generated for {filename}, skipping...")
        continue

    # Save per-PDF embeddings
    np.save(embedding_path, embeddings)
    print(f"✅ Saved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) → {embedding_path}")

    # Collect global data
    all_embeddings.append(embeddings)
    all_chunks.extend(chunks)

# ----------------------------
# Save merged embeddings + chunks
# ----------------------------
if all_embeddings:
    merged_embeddings = np.vstack(all_embeddings)
    np.save("merged_embeddings.npy", merged_embeddings)

    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n💾 Saved merged embeddings → merged_embeddings.npy")
    print(f"💾 Saved {len(all_chunks)} chunks → chunks.pkl")
else:
    print("⚠️ No embeddings were created.")
