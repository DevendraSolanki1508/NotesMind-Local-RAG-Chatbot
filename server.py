from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from query_engine import load_query_engine, chat

app = FastAPI()

# Load engine once at startup
print("Loading engine...")
retriever, reranker, nodes = load_query_engine()
history = []
print("Engine ready!")

class Question(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
def ask(q: Question):
    global history
    answer, sources = chat(q.question, retriever, reranker, history)
    source_list = [
        {
            "filename": s.metadata.get("filename", "Unknown"),
            "text": s.text[:300]
        }
        for s in sources
    ]
    return {"answer": answer, "sources": source_list}

@app.post("/clear")
def clear():
    global history
    history = []
    return {"status": "cleared"}

@app.get("/stats")
def stats():
    import chromadb
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        col    = client.get_collection("notesmind")
        return {"chunks": col.count()}
    except:
        return {"chunks": 0}