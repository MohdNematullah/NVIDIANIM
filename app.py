import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# Initialize NVIDIA LLM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Title
st.title("NVIDIA NIM DEMO")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Embedding documents function
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("D:/nimnvidia/ML&DL")  # Use forward slashes or raw string
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# UI Input
prompt1 = st.text_input("Enter Your Question From Documents")

# Button for embedding
if st.button("Document Embedding"):
    vector_embedding()
    st.success("FAISS Vector Store DB is ready using Nvidia Embeddings")

# Perform retrieval and answer
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    elapsed = time.process_time() - start

    # Display response
    st.subheader("Answer:")
    st.write(response.get('answer', 'No answer found.'))

    st.caption(f"Response time: {elapsed:.2f} seconds")

    # Display context chunks
    with st.expander("Document Similarity Search"):
        context_docs = response.get('context', [])
        if isinstance(context_docs, list):
            for i, doc in enumerate(context_docs):
                st.write(doc.page_content)
                st.write("----------------------------")
