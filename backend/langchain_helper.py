# langchain_helper.py

import os
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Configuration ===
CSV_FILE = "diseases.csv"
FAISS_DIR = "faiss_data"

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "REPLACE_WITH_FALLBACK_API_KEY")
assert GOOGLE_API_KEY, "❌ GEMINI_API_KEY environment variable is not set."

# === Initialize Gemini LLM and Embeddings ===
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)

# === Globals ===
_vectorstore = None
_qa_chain = None


def build_vectorstore(force_rebuild=False):
    """
    Builds or loads FAISS vector index from CSV data.
    If force_rebuild=True, deletes and recreates the index.
    """
    global _vectorstore

    if not force_rebuild and os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        try:
            _vectorstore = FAISS.load_local(
                FAISS_DIR, embedding_model, allow_dangerous_deserialization=True
            )
            print("✅ FAISS loaded from disk.")
            return
        except Exception as e:
            print(f"⚠️ Failed to load FAISS. Rebuilding. Error: {e}")

    # Load CSV
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
    
    loader = CSVLoader(file_path=CSV_FILE, encoding="utf-8")
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} documents from {CSV_FILE}")

    # Optional: split long rows if needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Split into {len(chunks)} chunks.")

    # Create FAISS vector store
    _vectorstore = FAISS.from_documents(chunks, embedding_model)
    _vectorstore.save_local(FAISS_DIR)
    print("✅ FAISS index rebuilt and saved.")


def get_vectorstore():
    """
    Loads the FAISS index into memory if not already loaded.
    """
    global _vectorstore
    if _vectorstore is None:
        build_vectorstore()
    return _vectorstore


def get_qa_chain():
    """
    Initializes and returns the QA chain using LangChain's RetrievalQA.
    It uses a structured medical summary prompt with Gemini.
    """
    global _qa_chain
    if _qa_chain is None:
        vs = get_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": 2, "score_threshold": 0.9999})

        template = '''
🎯 Generate a concise clinical summary using the retrieved context and question.

If no relevant info is found, return:
🚨 "PLEASE TEACH ME ABOUT THIS: No prior medical data found."

---

🔹 **Context:**
{context}

🔹 **Query:**
{question}

---

📋 **Summary:**

- **Symptoms:** [List key 2–3 symptoms]
- **Diagnosis:** [Most likely condition]
- **Differentials:** [If any]
- **Treatment:** [Short plan – meds/intervention/lifestyle]
- **OVERVIEW:**
  | Symptom | Medication | Notes | Specialist |
  |---------|------------|-------|------------|
  | [...]   | [...]      | ...   |............|

- **Urgency:** [Escalation criteria or 'No immediate concern']
- **Follow-up:** [Timeframe + tests if needed]
- **Referral:** [Specialist name or 'Not required']

---

⚠️ *This is an AI-generated aid. Final decisions rest with the attending physician.*
'''
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
    return _qa_chain


def learn(prompt: str, response: str):
    """
    Adds a new prompt-response pair to the CSV + FAISS index.
    Supports live learning without full rebuild.
    """
    global _vectorstore

    # Append to CSV
    df = pd.DataFrame([[prompt, response]], columns=["Prompt", "Response"])
    df.to_csv(CSV_FILE, mode="a", index=False, header=not os.path.exists(CSV_FILE))
    print(f"📝 Appended to {CSV_FILE}")

    # Add to in-memory FAISS
    if _vectorstore is None:
        build_vectorstore()

    new_doc = Document(page_content=f"Prompt: {prompt}\n\nResponse: {response}")
    _vectorstore.add_documents([new_doc])
    _vectorstore.save_local(FAISS_DIR)
    print("✅ FAISS index updated with new entry.")


# Optional: Run directly to rebuild FAISS
if __name__ == "__main__":
    build_vectorstore(force_rebuild=True)
    print("🔧 FAISS rebuilt manually.")
