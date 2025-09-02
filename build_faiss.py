import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Make sure faiss_data exists
FAISS_BASE_PATH = os.path.join(os.getcwd(), "faiss_data")
os.makedirs(FAISS_BASE_PATH, exist_ok=True)

# 1. Load a model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Example corpus (replace with your own text/documents)
documents = [
    "Climate change is real and accelerating.",
    "The IPCC reports provide scientific consensus on climate.",
    "FAISS is used for efficient similarity search.",
    "Groq provides high-performance inference for LLMs."
]

# 3. Generate embeddings
embeddings = model.encode(documents)

# 4. Build FAISS index
d = embeddings.shape[1]  # dimension of vectors
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# 5. Save index and documents
faiss.write_index(index, os.path.join(FAISS_BASE_PATH, "demo.index"))

with open(os.path.join(FAISS_BASE_PATH, "demo.pkl"), "wb") as f:
    pickle.dump(documents, f)

print("✅ FAISS index built and saved in faiss_data/")
