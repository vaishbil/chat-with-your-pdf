
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_graph import build_graph, run_agent

load_dotenv()

AVATARS = {"user": "😎", "assistant": "📎"}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Onest:wght@400..700&display=swap');

:root {
  --ink: #1E1B4B;
  --fog: #ECE9FF;
  --paper: #FFFFFF;
  --hl: #FFE14D;
  --pink: #FF4D8D;
}

/* ---------- type ---------- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp summary, .stApp [data-testid="stMarkdownContainer"] {
  font-family: 'Onest', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp h1, .stApp h2, .stApp h3 {
  font-family: 'Bricolage Grotesque', 'Onest', system-ui, sans-serif !important;
  letter-spacing: -0.03em;
}
.stApp h1 {
  font-weight: 800;
  font-size: 3.4rem;
  line-height: 1;
  padding: 0 0 0.35rem 0;
}
[data-testid="stCaptionContainer"] { font-size: 1.05rem; }

/* ---------- strip the default streamlit chrome ---------- */
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"] {
  background: var(--paper);
  border: 2px solid var(--ink);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.9rem;
}
/* your messages: highlighter-yellow stickers, pinned right */
[data-testid="stChatMessage"]:has(.me) {
  background: var(--hl);
  border-radius: 18px 18px 4px 18px;
  box-shadow: 4px 4px 0 var(--ink);
  margin-left: auto;
  width: fit-content;
  max-width: 80%;
  transform: rotate(-0.6deg);
  flex-direction: row-reverse;
}
[data-testid="stChatMessageAvatarUser"] { display: none; }
.element-container:has(.me) { display: none; }

/* ---------- receipts + "how i got this" ---------- */
[data-testid="stExpander"] details {
  border: 2px dashed var(--ink) !important;
  border-radius: 12px !important;
  background: transparent !important;
}
[data-testid="stExpander"] summary { font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stExpander"] details {
  border-color: rgba(244, 242, 255, 0.5) !important;
}

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"], button[kind="primary"] {
  background: var(--pink) !important;
  color: var(--ink) !important;
  font-weight: 700;
  border: 2px solid var(--ink) !important;
  border-radius: 12px;
  box-shadow: 3px 3px 0 var(--hl);
  transition: transform 0.08s ease, box-shadow 0.08s ease;
}
[data-testid="stBaseButton-primary"]:hover:not(:disabled),
button[kind="primary"]:hover:not(:disabled) {
  transform: translate(-1px, -1px);
  box-shadow: 4px 4px 0 var(--hl);
}
[data-testid="stBaseButton-primary"]:active:not(:disabled),
button[kind="primary"]:active:not(:disabled) {
  transform: translate(3px, 3px);
  box-shadow: 0 0 0 var(--hl);
}
[data-testid="stBaseButton-primary"]:disabled,
button[kind="primary"]:disabled {
  opacity: 0.45;
  box-shadow: none;
}

/* ---------- chat box ---------- */
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: var(--fog) !important; }
[data-testid="stChatInput"] {
  border: 2px solid var(--ink) !important;
  border-radius: 14px !important;
  background: var(--paper) !important;
}

/* ---------- empty state ---------- */
.empty { padding: 2.5rem 0 0 0; }
.empty b {
  display: block;
  font-family: 'Bricolage Grotesque', 'Onest', system-ui, sans-serif;
  font-weight: 700;
  font-size: 2rem;
  letter-spacing: -0.02em;
  margin-bottom: 0.3rem;
}
.empty span { font-size: 1.05rem; opacity: 0.8; }

/* ---------- a11y + small screens ---------- */
:focus-visible { outline: 3px solid var(--pink); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (max-width: 640px) {
  .stApp h1 { font-size: 2.4rem; }
  [data-testid="stChatMessage"]:has(.me) { max-width: 92%; }
}
</style>
"""


@st.cache_resource(show_spinner="loading the embedding model...")
def get_embeddings():
    # Free, runs locally. Downloads ~90MB the first time.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(provider: str):
    if provider == "Gemini (free tier)":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            temperature=0,
            timeout=30,
            max_retries=1,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)


def build_index(uploaded_files, chunk_size: int, chunk_overlap: int):
    docs = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        pages = PyPDFLoader(path).load()
        for p in pages:
            p.metadata["source"] = f.name  # show the real file name in citations
        docs.extend(pages)
        os.unlink(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings()), len(chunks)


def make_sources(hits):
    """Merge all retrieved chunks into ONE source entry."""
    pages = sorted({d.metadata.get("page", 0) + 1 for d in hits})
    names = sorted({d.metadata["source"] for d in hits})
    return [
        {
            "label": f"{', '.join(names)} (pages {', '.join(map(str, pages))})",
            "text": "\n\n---\n\n".join(d.page_content for d in hits),
        }
    ]


def chat_history(messages, last_n: int = 4) -> str:
    """Last few turns as plain text, so the agent can understand follow-up questions."""
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in messages[-last_n:]
    )


def render_extras(sources, trace):
    for s in sources:
        with st.expander(f"receipts: {s['label']}"):
            st.write(s["text"])
    if trace:
        with st.expander("how i got this"):
            for step in trace:
                st.write(f"- {step}")


def show_message(role, text, sources=(), trace=()):
    with st.chat_message(role, avatar=AVATARS[role]):
        if role == "user":
            st.markdown('<span class="me"></span>', unsafe_allow_html=True)
        st.markdown(text)
        render_extras(sources, trace)


# ---------------- UI ----------------
st.set_page_config(page_title="chat with your pdfs", page_icon="📎")
st.markdown(STYLE, unsafe_allow_html=True)

st.title("chat with your pdfs")
st.caption("drop in a pdf, ask questions, get answers with receipts.")

with st.sidebar:
    provider = st.selectbox("model", ["Gemini (free tier)"])
    files = st.file_uploader("your pdfs", type="pdf", accept_multiple_files=True)
    with st.expander("advanced tweaks"):
        k = st.slider(
            "chunks per question", 1, 8, 4,
            help="How many pieces of your pdf the agent reads for each question. "
                 "More gives it more to work with, but also more noise.",
        )
        chunk_size = st.slider(
            "chunk size", 300, 2000, 1000, step=100,
            help="How big each piece of the pdf is, in characters. Smaller is more "
                 "precise, bigger keeps more context. Hit 'read my pdfs' again after changing.",
        )
        chunk_overlap = st.slider(
            "chunk overlap", 0, 400, 150, step=50,
            help="Characters repeated between neighbouring pieces, so a sentence "
                 "doesn't get cut in half.",
        )
    if st.button("read my pdfs", type="primary", disabled=not files):
        with st.spinner("reading, chunking, embedding..."):
            st.session_state.store, n = build_index(files, chunk_size, chunk_overlap)
            st.session_state.doc_names = ", ".join(f.name for f in files)
            st.session_state.messages = []
        st.success(f"read {n} chunks from {len(files)} file(s). ask away.")

if "messages" not in st.session_state:
    st.session_state.messages = []

ready = "store" in st.session_state

if not st.session_state.messages:
    if ready:
        st.markdown(
            f'<div class="empty"><b>all read.</b>'
            f'<span>ask anything about {st.session_state.doc_names}.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="empty"><b>nothing here yet.</b>'
            '<span>add a pdf in the sidebar, hit "read my pdfs", then ask away.</span></div>',
            unsafe_allow_html=True,
        )

for m in st.session_state.messages:
    show_message(m["role"], m["content"], m.get("sources", []), m.get("trace", []))

placeholder = "ask something about your pdf..." if ready else "add a pdf first"
if question := st.chat_input(placeholder, disabled=not ready):
    history = chat_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    show_message("user", question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("thinking..."):
                graph = build_graph(st.session_state.store, get_llm(provider), k)
                state = run_agent(graph, question, history)
        except Exception as e:
            st.session_state.messages.pop()  # drop the failed question so it isn't shown twice
            st.error(
                f"couldn't get an answer from the model: {e}\n\n"
                "the free gemini tier can be slow or busy. try again in a minute."
            )
            st.stop()
        text = state["answer"]
        trace = state["trace"]
        # Only show receipts when the agent actually found relevant context
        sources = make_sources(state["docs"]) if state["relevant"] else []
        st.markdown(text)
        render_extras(sources, trace)
    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources, "trace": trace}
    )