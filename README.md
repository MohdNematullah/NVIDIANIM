
# 🧠 NVIDIA NIM + LangChain + Streamlit: Document-Based Q&A System

Welcome to the **NVIDIA NIM Demo App**! This project demonstrates how to build a powerful **Document Question Answering System** using:

- 🧠 NVIDIA NIM LLMs (`meta/llama3-70b-instruct`)
- 📄 PDF Document Parsing
- 📚 LangChain for chaining LLM logic
- 📊 FAISS for fast vector search
- 🌐 Streamlit for a beautiful, interactive UI

---

## 🚀 Features

✅ Upload & parse multiple PDF documents  
✅ Automatic chunking and vector embeddings with **NVIDIAEmbeddings**  
✅ Semantic document retrieval using **FAISS**  
✅ Answer questions contextually with **Llama 3 70B**  
✅ See similar document chunks in the UI  
✅ Fast and interactive **Streamlit** frontend

---

## 🛠️ Tech Stack

| Tool            | Role                                 |
|-----------------|--------------------------------------|
| `NVIDIA NIM`    | LLM Inferencing via LangChain        |
| `LangChain`     | Prompt templates, chains, retrievers |
| `FAISS`         | Vector store for embeddings          |
| `Streamlit`     | Interactive Web UI                   |
| `PyPDFLoader`   | Load and parse PDFs                  |
| `dotenv`        | Manage API keys securely             |

---

## 🧩 Architecture Overview

```mermaid
flowchart TD
    A[PDF Files] -->|PyPDFDirectoryLoader| B[Documents]
    B --> C[Text Splitter]
    C --> D[Chunks]
    D -->|NVIDIAEmbeddings| E[FAISS Vector Store]
    F[User Question] --> G[Retriever]
    E --> G
    G --> H[Prompt + LLM Response]
    H --> I[Answer + Similar Chunks in UI]
````



## ⚙️ Setup Instructions

### 1. Clone the Repository


git clone https://github.com/MohdNematullah/NVIDIANIM.git
cd nvidia-nim-docqa


### 2. Install Dependencies


pip install -r requirements.txt


### 3. Setup Environment Variables

Create a `.env` file:

NVIDIA_API_KEY=your_nvidia_nim_api_key_here
```

### 4. Add Your PDF Documents

Place all your documents inside a folder (e.g., `D:/nimnvidia/ML&DL`) and update the path in the script accordingly.

### 5. Run the App

streamlit run app.py

---

## 💬 Example Workflow

1. Click "Document Embedding" to load and embed documents.
2. Ask a question like:

   > *"What is backpropagation in deep learning?"*
3. Get a contextual answer from your PDFs.
4. Explore the most relevant document chunks in the **Document Similarity Search** section.


---

## 📌 To-Do / Future Enhancements

* [ ] Add document upload via UI
* [ ] Support more file types (DOCX, TXT)
* [ ] Enable response streaming
* [ ] Cache embeddings for faster reloads

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgments

* [LangChain](https://www.langchain.com/)
* [NVIDIA NIM](https://developer.nvidia.com/nim)
* [Streamlit](https://streamlit.io/)

```

