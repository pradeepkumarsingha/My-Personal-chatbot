import os
import json
import pickle
from langchain_core.documents import Document 
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = "documents"
CHROMA_DIR = "chroma_db"
BM25_PKL_PATH = os.path.join(CHROMA_DIR, "bm25_corpus.pkl")

docs = []

# 1. Clean JSON Formatter Helper
def format_json_to_text(data):
    """Converts dictionaries, lists of strings, or lists of dicts into clear text."""
    lines = []
    if isinstance(data, list):
        for index, item in enumerate(data):
            # Check if the item inside the list is a dictionary (like in projects.json)
            if isinstance(item, dict):
                item_lines = [f"{key.capitalize()}: {value}" for key, value in item.items()]
                lines.append(f"--- Item {index+1} ---\n" + "\n".join(item_lines))
            # If the item is just a string (like in skills.json -> ["Python", "React"])
            else:
                lines.append(str(item))
        return "\n".join(lines)
        
    elif isinstance(data, dict):
        return "\n".join([f"{key.capitalize()}: {value}" for key, value in data.items()])
    return str(data)
print("📂 Scanning and parsing files...")

# 2. Iterate dynamically over files to avoid duplicates or missing entries
for file in os.listdir(DOCS_DIR):
    path = os.path.join(DOCS_DIR, file)

    if file.endswith(".pdf"):
        # Explicitly load the PDF pages dynamically
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    elif file.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        docs.append(
            Document(
                page_content=text,
                metadata={"source": file}
            )
        )

    elif file.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        # Format JSON items as clean readable paragraphs for better retrieval match
        formatted_text = format_json_to_text(data)
        docs.append(
            Document(
                page_content=formatted_text,
                metadata={"source": file}
            )
        )

print(f"✨ Loaded {len(docs)} document elements.")

# 3. Setup Vector Database (Dense Search)
print("🧠 Generating dense embeddings with ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=CHROMA_DIR
)

# 4. Save BM25 Corpus (Sparse Search)
print("📝 Packing raw documents for BM25 Keyword Search...")
bm25_corpus = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

os.makedirs(CHROMA_DIR, exist_ok=True)
with open(BM25_PKL_PATH, "wb") as f:
    pickle.dump(bm25_corpus, f)

print("🚀 Database Created Successfully with Hybrid Assets!")