# 🤖 Conversational RAG Thesis Assistant

A memory-aware Retrieval-Augmented Generation (RAG) assistant built with **LangChain**, **ChromaDB**, and **Llama 3.2** served through the Hugging Face Inference API. It ingests a long PDF (a master's thesis, a research paper) and answers natural-language questions about it across a multi-turn conversation, so follow-up questions like *"and which algorithms did it use?"* resolve against what was already asked.

---

## 🛠️ Tech Stack

| Layer | Choice |
| --- | --- |
| Language model | `meta-llama/Llama-3.2-3B-Instruct` (HF Inference API) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally) |
| Vector store | ChromaDB (in-memory) |
| Orchestration | LangChain — `RunnableWithMessageHistory`, `ChatPromptTemplate` |
| Document processing | `PyPDFLoader` + `RecursiveCharacterTextSplitter` |

---

## 🧭 How it works

1. **Load** — `PyPDFLoader` reads the PDF page by page.
2. **Chunk** — text is split into 1000-character chunks with 200 characters of overlap, so sentences aren't cut mid-thought.
3. **Embed & index** — each chunk is embedded locally with MiniLM and indexed in Chroma.
4. **Retrieve** — each question pulls the top `k=6` most similar chunks.
5. **Answer** — the chunks, the chat history, and the question go into a system prompt that forbids answering outside the retrieved context, then to Llama 3.2.
6. **Remember** — `RunnableWithMessageHistory` keeps per-session history in memory, keyed by `session_id`.

---

## 🚀 Setup

**1. Clone and enter the repo**

```bash
git clone https://github.com/your-username/langchain_project.git
cd langchain_project
```

**2. Create a virtual environment and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your Hugging Face token**

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and request access to the [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) (it is a gated repo).

Then open the notebook and paste the token into **Cell 1**:

```python
HF_TOKEN = "hf_your_token_here"
```

Prefer not to put it in the file? Export it in the shell before launching Jupyter and read it from the environment instead:

```bash
export HF_TOKEN="hf_your_token_here"
```

```python
HF_TOKEN = os.environ["HF_TOKEN"]
```

**4. Add your PDF**

Place the document next to the notebook and point `pdf_path` in **Cell 2** at it:

```python
pdf_path = "Sami_Ercetingoz_Thesis.pdf"
```

PDFs are excluded by `.gitignore`, so the source document is never committed — you supply your own.

**5. Run**

```bash
jupyter notebook conversational_rag_assistant.ipynb
```

Run the cells top to bottom. Cells 0 and 1 (the `pip install` cells) can be skipped once `requirements.txt` is installed in the virtual environment.

---

## 💬 Asking your own questions

Cell 5 shows the pattern. Reuse the same `session_id` to keep the conversation's memory:

```python
config = {"configurable": {"session_id": "user_session_1"}}

response = conversational_rag_chain.invoke(
    {"question": "What evaluation metrics were reported?"},
    config=config,
)
print(response)
```

A new `session_id` starts a fresh conversation with no history.

---

## ⚠️ Notes & limitations

- **NumPy is pinned below 2.0.** `chromadb` and the `scikit-learn` build used here are compiled against the 1.x ABI; upgrading NumPy breaks the import chain.
- **The vector store is in-memory.** It is rebuilt on every kernel restart. To persist it, pass `persist_directory="chroma_db/"` to `Chroma.from_documents` (that path is already gitignored).
- **Never commit a real token.** `.env` is gitignored, but a token pasted directly into the notebook *does* get committed — clear it before pushing.
- The model is a 3B instruct model answering strictly from retrieved context; it will say the information is unavailable rather than guess, which is intended.
