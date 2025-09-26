import streamlit as st
import os
import fitz  # PyMuPDF
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Helper Functions ---

def get_text_from_files(docs):
    """Extracts and cleans text from a list of uploaded files."""
    text = ""
    for doc in docs:
        file_extension = os.path.splitext(doc.name)[1]
        if file_extension == ".pdf":
            with fitz.open(stream=doc.read(), filetype="pdf") as pdf_doc:
                for page in pdf_doc:
                    text += page.get_text()
        elif file_extension == ".docx":
            doc_reader = Document(doc)
            for para in doc_reader.paragraphs:
                text += para.text + "\n"
        elif file_extension == ".txt":
            text += doc.getvalue().decode("utf-8")
    
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if "GUVIHCL" not in line and "Skill Up. Level Up" not in line]
    cleaned_text = "\n".join(cleaned_lines)
    
    # Enhanced cleaning to remove artifacts from PDF conversion
    cleaned_text = cleaned_text.replace('"', '')
    
    return cleaned_text

def get_text_chunks(text):
    """Splits a long text into larger, more coherent chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        # Increased chunk size to better capture context in structured documents
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_qa_chain(vectorstore, model_path):
    """Creates a stateless RetrievalQA chain with an improved retriever."""
    custom_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer from the context, just say that the answer is not available in the document.
    Do not use any other information. Keep the answer concise and to the point.
    
    Context: {context}
    Question: {question}
    
    Helpful Answer:
    """
    CUSTOM_PROMPT = PromptTemplate.from_template(custom_template)

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,
        n_batch=512,
        temperature=0.3,
        max_tokens=512,
        stop=["Question:"] 
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # Using default similarity search with a more focused number of chunks
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    return qa_chain

# --- Streamlit App UI ---

def main():
    st.set_page_config(page_title="Chat with Your Docs", page_icon="📚")

    # Initialize session state variables
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    st.header("Chat with Your Documents 📚")
    st.write("Upload your files, ask questions, and get context-aware answers.")

    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDFs, DOCX, or TXT files here",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )
        
        model_path = st.text_input(
            "Enter the path to your Llama 2 model file", 
            "llama-2-7b-chat.Q4_K_M.gguf"
        )

        current_file_names = sorted([file.name for file in uploaded_files])
        if current_file_names and current_file_names != st.session_state.processed_files:
            st.info("New documents detected. Click 'Process Documents' to update.")
            if st.session_state.qa_chain:
                st.session_state.qa_chain = None
                st.session_state.chat_history = []


        if st.button("Process Documents"):
            if uploaded_files:
                if os.path.exists(model_path):
                    with st.spinner("Processing documents..."):
                        raw_text = get_text_from_files(uploaded_files)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.qa_chain = get_qa_chain(vectorstore, model_path)
                        st.session_state.chat_history = []
                        st.session_state.processed_files = current_file_names
                        st.success("Documents processed. Ready to chat!")
                else:
                    st.error(f"Model file not found at: {model_path}")
            else:
                st.warning("Please upload at least one document.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_question := st.chat_input("Ask a question about your documents:"):
        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        if st.session_state.qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.qa_chain({"query": user_question})
                    answer = response.get("result", "Sorry, I could not generate a response.")
                    # Add assistant response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.markdown(answer)
        else:
            st.warning("Please upload and process your documents before asking a question.")

if __name__ == '__main__':
    main()

