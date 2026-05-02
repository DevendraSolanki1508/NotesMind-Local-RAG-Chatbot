import chromadb
import ollama
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

# ── Configuration ──────────────────────────────────────────────
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "notesmind"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL   = "mistral"

def load_query_engine():
    print("Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.llm = None

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection    = chroma_client.get_collection(COLLECTION)
    vector_store  = ChromaVectorStore(chroma_collection=collection)
    index         = VectorStoreIndex.from_vector_store(vector_store)

    # ── BM25 nodes from ChromaDB ───────────────────────────────
    result = collection.get(include=["documents", "metadatas"])
    nodes = [
        TextNode(text=doc, metadata=meta)
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]

    # ── Retrievers ─────────────────────────────────────────────
    vector_retriever = index.as_retriever(similarity_top_k=5)
    bm25_retriever   = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5
    )
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=3,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False
    )

    # ── Reranker ───────────────────────────────────────────────
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        top_n=3
    )

    print("✓ Query engine ready!\n")
    return hybrid_retriever, reranker, nodes

def chat(query: str, retriever, reranker, history: list) -> str:
    # Step 1 — retrieve relevant chunks
    nodes = retriever.retrieve(query)

    # Step 2 — rerank
    from llama_index.core.schema import QueryBundle
    nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))

    # Step 3 — build context from top chunks
    context = "\n\n---\n\n".join([n.text for n in nodes])

    # Step 4 — build messages with history
    system_prompt = (
    "You are a helpful assistant answering questions based on document context. "
    "The documents may contain OCR errors, typos, or garbled text from scanning — "
    "do your best to interpret the meaning despite imperfect text. "
    "Answer clearly and concisely based only on what is in the context. "
    "If the context does not contain enough information, say so honestly."
)
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for h in history[-6:]:   # keep last 3 exchanges
        messages.append(h)

    # Add current query with context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    # Step 5 — call Ollama directly
    print("Thinking...")
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    answer = response["message"]["content"]

    # Update history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer, nodes