import streamlit as st
import time
import os
import chromadb

st.set_page_config(page_title="NotesMind", page_icon="🧠", layout="wide")

# ── Global engine store (survives Streamlit reruns) ────────────
@st.cache_resource
def get_engine():
    from query_engine import load_query_engine
    return load_query_engine()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 NotesMind")
    st.caption("Chat with your documents — fully offline")
    st.divider()

    st.subheader("📁 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Add documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "docx"]
    )
    if uploaded_files:
        for file in uploaded_files:
            save_path = os.path.join("./docs", file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s)")
        st.warning("Re-run ingest.py to index new files.")

    st.divider()
    st.subheader("📊 Stats")
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection    = chroma_client.get_collection("notesmind")
        st.metric("Chunks indexed", collection.count())
        results = collection.get(include=["metadatas"])
        docs = set()
        for meta in results["metadatas"]:
            if meta and "filename" in meta:
                docs.add(meta["filename"])
        st.metric("Documents", len(docs))
        if docs:
            st.caption("Files indexed:")
            for d in sorted(docs):
                st.caption(f"• {d}")
    except Exception:
        st.caption("No index found. Run ingest.py first.")

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state["messages"] = []
        st.session_state["history"]  = []
        st.rerun()

# ── Main ───────────────────────────────────────────────────────
st.header("💬 Ask your documents anything")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Sources"):
                for i, src in enumerate(msg["sources"]):
                    fname = src.metadata.get("filename", "Unknown")
                    st.caption(f"**Source {i+1} — {fname}**")
                    st.caption(src.text[:400] + "...")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask anything about your documents..."):

    with st.chat_message("user"):
        st.write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Loading engine + thinking... (first question takes ~1 min)"):
            try:
                from query_engine import chat
                retriever, reranker, nodes = get_engine()
                start  = time.time()
                answer, source_nodes = chat(
                    prompt,
                    retriever,
                    reranker,
                    st.session_state["history"]
                )
                latency = round(time.time() - start, 1)

                st.write(answer)
                st.caption(f"⏱️ {latency}s")

                if source_nodes:
                    with st.expander("📄 Sources"):
                        for i, src in enumerate(source_nodes):
                            fname = src.metadata.get("filename", "Unknown")
                            st.caption(f"**Source {i+1} — {fname}**")
                            st.caption(src.text[:400] + "...")
                            st.divider()

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_nodes
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")