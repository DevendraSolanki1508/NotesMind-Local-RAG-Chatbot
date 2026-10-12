from query_engine import load_query_engine
from llama_index.core.schema import QueryBundle
import chromadb

# 1. Check text quality straight from the database
col = chromadb.PersistentClient(path="./chroma_db").get_collection("notesmind")
data = col.get(include=["documents", "metadatas"], limit=60)
print("\n===== SAMPLE STORED CHUNKS (is the text readable?) =====")
for doc, meta in list(zip(data["documents"], data["metadatas"]))[:2] + list(zip(data["documents"], data["metadatas"]))[-2:]:
    print(f"\n[{meta.get('filename')}]\n{doc[:400]}\n")

# 2. Check what retrieval returns for a question (no LLM involved)
retriever, reranker, _ = load_query_engine()
q = input("\nType a question: ")
nodes = retriever.retrieve(q)
nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(q))
print("\n===== CHUNKS SENT TO THE MODEL =====")
for i, n in enumerate(nodes, 1):
    print(f"\n--- Chunk {i} ({n.metadata.get('filename')}) ---\n{n.text[:500]}")