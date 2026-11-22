"""Evaluate the RAG pipeline on a list of test questions.

Usage:
    python eval.py path/to/your.pdf
    python eval.py path/to/your.pdf --chunk-size 500 --k 3
"""
import argparse
import os
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions using ONLY the context below.\n"
     "If the answer is not in the context, say \"I couldn't find that in the document.\"\n"
     "Be concise and accurate.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def normalize(text: str) -> str:
    return text.lower().replace("\u2019", "'").strip()


def to_text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--questions", default="eval_questions.csv")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between questions")
    args = parser.parse_args()

    print("Loading PDF and building index...")
    pages = PyPDFLoader(args.pdf).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks (chunk_size={args.chunk_size}, k={args.k})")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
        temperature=0, timeout=60, max_retries=2,
    )

    df = pd.read_csv(args.questions)
    answers, correct = [], []
    for i, row in df.iterrows():
        hits = store.similarity_search(row["question"], k=args.k)
        context = "\n\n".join(d.page_content for d in hits)
        try:
            reply = (PROMPT | llm).invoke({"context": context, "question": row["question"]})
            ans = to_text(reply.content)
        except Exception as e:
            ans = f"ERROR: {e}"
        ok = normalize(str(row["expected_keyword"])) in normalize(ans)
        answers.append(ans)
        correct.append(ok)
        print(f"[{i + 1}/{len(df)}] {'PASS' if ok else 'FAIL'}  {row['question']}")
        time.sleep(args.delay)

    df["answer"] = answers
    df["correct"] = correct
    df.to_csv("eval_results.csv", index=False)

    score = sum(correct)
    print(f"\nAccuracy: {score}/{len(df)} = {score / len(df):.0%}")
    print("Details saved to eval_results.csv")


if __name__ == "__main__":
    main()