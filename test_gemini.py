import os, time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = os.getenv("GEMINI_MODEL")
print("Using model:", model)

llm = ChatGoogleGenerativeAI(model=model, timeout=60, max_retries=0)
start = time.time()
print(llm.invoke("Say hello in one word").content)
print("Took", round(time.time() - start, 1), "seconds")