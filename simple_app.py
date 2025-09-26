"""
Generic RAG Q&A System with Token Optimization
Works with any PDF document without hardcoding
"""
import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

# Page config
st.set_page_config(page_title="Document Q&A System", page_icon="📚")
st.title("📚 Document Q&A with RAG")
st.caption("Upload any PDF and ask questions about it")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model parameters
    st.subheader("Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_tokens = st.slider("Max Response Tokens", 100, 500, 256, 50)
    
    st.subheader("Retrieval Settings")
    chunk_size = st.slider("Chunk Size", 200, 600, 350, 50)
    chunk_overlap = st.slider("Chunk Overlap", 20, 100, 50, 10)
    n_chunks = st.slider("Chunks to Retrieve", 1, 5, 3, 1)
    
    st.divider()
    
    # File upload
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload any PDF document to analyze"
    )
    
    # Process button
    if uploaded_file:
        if st.button("📥 Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    # Create temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_path = tmp_file.name
                    
                    # Load and split document
                    loader = PyPDFLoader(temp_path)
                    pages = loader.load()
                    
                    # Show document info
                    st.info(f"📖 Loaded {len(pages)} pages from: {uploaded_file.name}")
                    
                    # Text splitter
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", ". ", " ", ""],
                        length_function=len
                    )
                    
                    chunks = text_splitter.split_documents(pages)
                    st.success(f"✂️ Split into {len(chunks)} chunks")
                    
                    # Create embeddings
                    with st.spinner("Creating embeddings..."):
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': True}
                        )
                        
                        # Create vector store
                        vector_store = FAISS.from_documents(chunks, embeddings)
                        st.session_state.vector_store = vector_store
                        st.session_state.current_file = uploaded_file.name
                    
                    # Initialize LLM
                    with st.spinner("Loading language model..."):
                        llm = CTransformers(
                            model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                            model_type="llama",
                            config={
                                'max_new_tokens': max_tokens,
                                'temperature': temperature,
                                'context_length': 2048
                            }
                        )
                        
                        # Create prompt template
                        prompt_template = """Use the following context to answer the question.
If you don't know the answer based on the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer: """
                        
                        prompt = PromptTemplate(
                            template=prompt_template,
                            input_variables=["context", "question"]
                        )
                        
                        # Create QA chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=vector_store.as_retriever(
                                search_kwargs={'k': n_chunks}
                            ),
                            return_source_documents=True,
                            chain_type_kwargs={"prompt": prompt}
                        )
                        
                        st.session_state.qa_chain = qa_chain
                    
                    st.success(f"✅ Document ready for Q&A!")
                    
                    # Cleanup temp file
                    os.unlink(temp_path)
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    # Status display
    if st.session_state.current_file:
        st.divider()
        st.subheader("📊 Status")
        st.success(f"Active document: {st.session_state.current_file}")
        
        # Estimated token usage
        estimated_tokens = (chunk_size * n_chunks) // 4
        remaining = 2048 - estimated_tokens - 150  # Leave room for prompt
        
        if remaining > 0:
            st.metric("Token Budget", f"{estimated_tokens}/2048", f"+{remaining} available")
        else:
            st.error(f"⚠️ May exceed token limit! Reduce chunk size or number.")
        
        # Clear button
        if st.button("🗑️ Clear & Start Over", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.current_file = None
            st.rerun()

# Main Q&A Interface
st.header("💬 Ask Questions About Your Document")

if st.session_state.qa_chain:
    # Display current document
    st.info(f"📄 Current document: **{st.session_state.current_file}**")
    
    # Question input
    question = st.text_area(
        "What would you like to know?",
        placeholder="Type your question here...",
        height=100
    )
    
    # Submit button
    if st.button("🔍 Get Answer", type="primary", disabled=not question):
        if question:
            with st.spinner("Finding answer..."):
                try:
                    # Get response
                    response = st.session_state.qa_chain({"query": question})
                    
                    # Display answer
                    st.markdown("### 📝 Answer")
                    answer = response.get('result', 'No answer found.')
                    
                    # Display in a nice container
                    with st.container():
                        st.write(answer)
                    
                    # Show source chunks if available
                    if response.get('source_documents'):
                        with st.expander("📚 View Source Context", expanded=False):
                            for i, doc in enumerate(response['source_documents'], 1):
                                st.markdown(f"**Chunk {i}:**")
                                # Show only first 500 chars of each chunk
                                content = doc.page_content[:500]
                                if len(doc.page_content) > 500:
                                    content += "..."
                                st.text(content)
                                st.divider()
                    
                    # Show metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"📊 Retrieved {len(response.get('source_documents', []))} chunks")
                    with col2:
                        st.caption(f"💭 Response length: {len(answer)} chars")
                        
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    if "context length" in str(e).lower():
                        st.warning("💡 Try reducing chunk size or number of chunks retrieved.")
else:
    # No document loaded
    st.info("👆 Please upload a PDF document in the sidebar to begin")
    
    # Instructions
    with st.expander("📖 How to use", expanded=True):
        st.markdown("""
        1. **Upload a PDF** document using the sidebar
        2. **Click Process Document** to analyze it
        3. **Ask questions** about the content
        4. The system will search the document and provide answers
        
        **Tips for best results:**
        - Use clear, specific questions
        - If you get token errors, reduce chunk size or number
        - For detailed documents, try asking about specific sections
        """)

# Footer
st.divider()
st.caption("Built with LangChain, FAISS, and Llama 2")