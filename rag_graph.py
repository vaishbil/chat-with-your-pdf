"""Agentic RAG pipeline built with LangGraph.

Flow:
    rewrite -> retrieve -> grade --(relevant)--> generate -> END
                  ^           |
                  |           +--(not relevant, retries left)--> rewrite
                  |           +--(not relevant, no retries)----> give_up -> END
"""
import operator
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

MAX_RETRIES = 1  # how many times the agent may re-search with a new query
NOT_FOUND = "I couldn't find that in the document."

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's question as a short, self-contained search query for "
     "searching inside a document. Use the chat history to resolve words like "
     "'it' or 'that'. Output ONLY the query."),
    ("human", "Chat history:\n{history}\n\nQuestion: {question}"),
])

RETRY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "A previous search query did not find useful passages in the document. "
     "Write a DIFFERENT search query for the question, using different keywords "
     "or synonyms. Stay on the same topic as the chat history and the previous "
     "query; do not broaden to a more general subject. Output ONLY the query."),
    ("human", "Chat history:\n{history}\n\nQuestion: {question}\nPrevious query: {previous}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check search results. Does the context below contain information that "
     "answers the question, at least partially? Reply with exactly one word: "
     "yes or no."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions using ONLY the context below.\n"
     f"If the answer is not in the context, say \"{NOT_FOUND}\"\n"
     "Answer only what the question asks; do not add unrelated parts of the context. "
     "Be concise and accurate.\n\nContext:\n{context}"),
    ("human", "{question}\n\n(Given the conversation so far, this question is about: {search_query})"),
])


class RAGState(TypedDict):
    question: str
    history: str
    search_query: str
    docs: List[Document]
    relevant: bool
    answer: str
    attempts: int
    trace: Annotated[List[str], operator.add]  # steps accumulate across nodes


def to_text(content) -> str:
    """Some Gemini models return a list of blocks instead of a plain string."""
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )


def build_graph(store, llm, k: int = 4):
    def rewrite(state: RAGState):
        if state["attempts"] == 0:
            chain = REWRITE_PROMPT | llm
            reply = chain.invoke({"question": state["question"], "history": state["history"] or "(none)"})
        else:
            chain = RETRY_PROMPT | llm
            reply = chain.invoke({
                "question": state["question"],
                "previous": state["search_query"],
                "history": state["history"] or "(none)",
            })
        query = to_text(reply.content).strip().strip('"') or state["question"]
        return {
            "search_query": query,
            "attempts": state["attempts"] + 1,
            "trace": [f"Rewrote question into search query: \"{query}\""],
        }

    def retrieve(state: RAGState):
        docs = store.similarity_search(state["search_query"], k=k)
        pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
        return {"docs": docs, "trace": [f"Retrieved {len(docs)} chunks from pages {pages}"]}

    def grade(state: RAGState):
        reply = (GRADE_PROMPT | llm).invoke(
            {"question": state["question"], "context": format_context(state["docs"])}
        )
        relevant = to_text(reply.content).strip().lower().startswith("yes")
        verdict = "relevant" if relevant else "NOT relevant"
        return {"relevant": relevant, "trace": [f"Checked results: {verdict}"]}

    def route_after_grade(state: RAGState):
        if state["relevant"]:
            return "generate"
        if state["attempts"] <= MAX_RETRIES:
            return "rewrite"
        return "give_up"

    def generate(state: RAGState):
        reply = (ANSWER_PROMPT | llm).invoke(
            {
                "context": format_context(state["docs"]),
                "question": state["question"],
                "search_query": state["search_query"],
            }
        )
        return {"answer": to_text(reply.content), "trace": ["Generated answer from retrieved context"]}

    def give_up(state: RAGState):
        return {"answer": NOT_FOUND, "trace": ["No relevant context found after retrying - declined to answer"]}

    g = StateGraph(RAGState)
    g.add_node("rewrite", rewrite)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("give_up", give_up)

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"},
    )
    g.add_edge("generate", END)
    g.add_edge("give_up", END)
    return g.compile()


def run_agent(graph, question: str, history: str = ""):
    """Run one question through the graph and return the final state."""
    return graph.invoke({
        "question": question,
        "history": history,
        "search_query": "",
        "docs": [],
        "relevant": False,
        "answer": "",
        "attempts": 0,
        "trace": [],
    })