"""
Context-Aware Question Answering System with RAG
Using Llama 2, LangChain, and Streamlit
FULLY FIXED VERSION - No errors, No infinite loops
"""

import streamlit as st
import os
import sys
import tempfile
import shutil
from datetime import datetime
import json
import time
from typing import List, Dict, Any, Optional
import hashlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core dependencies with error handling
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import CTransformers
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    DOCX_SUPPORT = True
except ImportError as e:
    st.error(f"Missing required dependency: {e}")
    st.info("Install with: pip install langchain-community langchain-core")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        word-wrap: break-word;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stSpinner > div {
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state with all required variables
def init_session_state():
    """Initialize all session state variables with proper defaults"""
    defaults = {
        'chat_history': [],
        'vector_store': None,
        'qa_chain': None,
        'documents_processed': [],
        'total_chunks': 0,
        'model_loaded': False,
        'model_instance': None,
        'embeddings': None,
        'processing_time': 0.0,
        'last_query': "",
        'query_count': 0,
        'processing_query': False,
        'current_query_hash': None,
        'error_count': 0,
        'initialization_complete': False,
        'show_sources': False,
        'response_times': [],
        'failed_queries': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Configuration class
class RAGConfig:
    """Configuration for RAG system"""
    
    # Model settings
    MODEL_NAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    MODEL_PATHS = [
        MODEL_NAME,
        f"./{MODEL_NAME}",
        f"models/{MODEL_NAME}",
        f"../{MODEL_NAME}",
        str(Path.home() / "models" / MODEL_NAME)
    ]
    
    # Embedding settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Text splitting settings
    CHUNK_SIZE = 500  # Smaller chunks for resumes
    CHUNK_OVERLAP = 50  # Less overlap
    MIN_CHUNK_SIZE = 50  # Lower minimum
    
    # Retrieval settings
    RETRIEVAL_K = 5  # Increased for better coverage
    SEARCH_TYPE = "similarity"  # Changed to similarity for better results
    FETCH_K = 20  # Increased fetch
    LAMBDA_MULT = 0.5
    SCORE_THRESHOLD = 0.3  # Lowered threshold
    
    # LLM settings
    MAX_NEW_TOKENS = 1024
    TEMPERATURE = 0.1
    CONTEXT_LENGTH = 4096
    TOP_P = 0.95
    REPETITION_PENALTY = 1.1
    
    # Vector DB settings
    VECTOR_DB_DIR = "vector_db"
    VECTOR_DB_PATH = os.path.join(VECTOR_DB_DIR, "faiss")
    AUTO_SAVE = True
    
    # File upload settings
    MAX_FILE_SIZE = 200 * 1024 * 1024
    ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx']
    
    # Performance settings
    BATCH_SIZE = 100
    CACHE_SIZE = 100
    
    # Anti-hallucination prompt
    PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer the question using the context provided below.

Context:
{context}

Question: {question}

Instructions:
- If the answer can be found in the context, provide it clearly and directly
- Include specific details and numbers when available
- If the information is not in the context, say "I cannot find this information in the provided documents"

Answer: """

    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        errors = []
        
        if cls.CHUNK_SIZE <= cls.CHUNK_OVERLAP:
            errors.append("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
        
        if cls.RETRIEVAL_K > cls.FETCH_K:
            errors.append("RETRIEVAL_K cannot be greater than FETCH_K")
        
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 1:
            errors.append("TEMPERATURE must be between 0 and 1")
        
        if cls.MAX_NEW_TOKENS < 1 or cls.MAX_NEW_TOKENS > 4096:
            errors.append("MAX_NEW_TOKENS must be between 1 and 4096")
        
        return errors

class DocumentProcessor:
    """Document loading and processing"""
    
    @staticmethod
    def validate_file(file_path: str) -> bool:
        """Validate file before processing"""
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
        
        file_size = os.path.getsize(file_path)
        if file_size > RAGConfig.MAX_FILE_SIZE:
            logger.error(f"File too large: {file_size} bytes")
            return False
        
        if file_size == 0:
            logger.error(f"File is empty: {file_path}")
            return False
        
        return True
    
    @staticmethod
    def load_documents(file_paths: List[str]) -> List[Document]:
        """Load documents with error handling"""
        documents = []
        failed_files = []
        
        for file_path in file_paths:
            if not DocumentProcessor.validate_file(file_path):
                failed_files.append(file_path)
                continue
            
            try:
                file_extension = os.path.splitext(file_path)[1].lower()
                loader = None
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_extension == '.txt':
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            loader = TextLoader(file_path, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    if not loader:
                        raise Exception("Could not decode text file")
                elif file_extension == '.docx' and DOCX_SUPPORT:
                    loader = Docx2txtLoader(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_extension}")
                    failed_files.append(file_path)
                    continue
                
                if loader:
                    docs = loader.load()
                    
                    for i, doc in enumerate(docs):
                        doc.metadata.update({
                            'source_file': os.path.basename(file_path),
                            'file_path': file_path,
                            'file_type': file_extension,
                            'doc_index': i,
                            'total_docs': len(docs),
                            'processed_at': datetime.now().isoformat()
                        })
                    
                    documents.extend(docs)
                    logger.info(f"Successfully loaded {file_path}: {len(docs)} documents")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                failed_files.append(file_path)
        
        if failed_files:
            st.warning(f"Failed to load {len(failed_files)} file(s): {', '.join([os.path.basename(f) for f in failed_files])}")
        
        return documents
    
    @staticmethod
    def process_documents(documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        chunks = []
        for doc in documents:
            cleaned_text = doc.page_content.strip()
            if not cleaned_text:
                continue
            
            doc.page_content = cleaned_text
            doc_chunks = text_splitter.split_documents([doc])
            
            doc_chunks = [
                chunk for chunk in doc_chunks 
                if len(chunk.page_content.strip()) >= RAGConfig.MIN_CHUNK_SIZE
            ]
            
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    'chunk_id': f"{doc.metadata.get('doc_index', 0)}_{i}",
                    'chunk_index': i,
                    'chunk_total': len(doc_chunks),
                    'chunk_size': len(chunk.page_content)
                })
            
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

class VectorStoreManager:
    """Vector store operations"""
    
    @staticmethod
    @st.cache_resource
    def create_embeddings():
        """Create embedding model"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=RAGConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': RAGConfig.BATCH_SIZE
                }
            )
            logger.info("Embeddings model created successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
    
    @staticmethod
    def create_vector_store(documents: List[Document], embeddings) -> Optional[FAISS]:
        """Create FAISS vector store"""
        if not documents:
            logger.warning("No documents to create vector store")
            return None
        
        try:
            if len(documents) > RAGConfig.BATCH_SIZE:
                vector_store = None
                for i in range(0, len(documents), RAGConfig.BATCH_SIZE):
                    batch = documents[i:i + RAGConfig.BATCH_SIZE]
                    if vector_store is None:
                        vector_store = FAISS.from_documents(batch, embeddings)
                    else:
                        batch_store = FAISS.from_documents(batch, embeddings)
                        vector_store.merge_from(batch_store)
                    
                    progress = min((i + RAGConfig.BATCH_SIZE) / len(documents), 1.0)
                    st.progress(progress, text=f"Processing documents: {int(progress * 100)}%")
            else:
                vector_store = FAISS.from_documents(documents, embeddings)
            
            logger.info(f"Vector store created with {len(documents)} documents")
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            st.error(f"Error creating vector store: {str(e)}")
            return None
    
    @staticmethod
    def save_vector_store(vector_store: FAISS, path: str) -> bool:
        """Save vector store"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            return False
    
    @staticmethod
    def load_vector_store(path: str, embeddings) -> Optional[FAISS]:
        """Load vector store"""
        try:
            if not os.path.exists(path):
                logger.warning(f"Vector store path does not exist: {path}")
                return None
            
            vector_store = FAISS.load_local(
                path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {path}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None

class LLMManager:
    """LLM operations"""
    
    @staticmethod
    @st.cache_resource
    def load_llm(model_path: str = None) -> Optional[CTransformers]:
        """Load Llama 2 model"""
        paths_to_try = RAGConfig.MODEL_PATHS if model_path is None else [model_path]
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    logger.info(f"Attempting to load model from: {path}")
                    
                    file_size = os.path.getsize(path)
                    if file_size < 1000000:
                        logger.warning(f"Model file seems too small: {file_size} bytes")
                        continue
                    
                    llm = CTransformers(
                        model=path,
                        model_type="llama",
                        config={
                            'max_new_tokens': RAGConfig.MAX_NEW_TOKENS,
                            'temperature': RAGConfig.TEMPERATURE,
                            'top_p': RAGConfig.TOP_P,
                            'repetition_penalty': RAGConfig.REPETITION_PENALTY,
                            'context_length': RAGConfig.CONTEXT_LENGTH,
                            'gpu_layers': 0,
                            'threads': os.cpu_count() // 2
                        }
                    )
                    
                    test_response = llm("Test")
                    if test_response:
                        logger.info(f"Model loaded successfully from: {path}")
                        return llm
                    
                except Exception as e:
                    logger.error(f"Failed to load model from {path}: {e}")
                    continue
        
        st.error("❌ Llama 2 model not found!")
        st.info(f"""
        **To fix this issue:**
        
        1. Download the model file: `{RAGConfig.MODEL_NAME}`
        2. Place it in one of these locations:
           - Current directory: `./`
           - Models folder: `./models/`
           - Home directory: `~/models/`
        3. Restart the application
        """)
        
        return None
    
    @staticmethod
    def create_qa_chain(llm, vector_store) -> Optional[RetrievalQA]:
        """Create RetrievalQA chain"""
        if not llm or not vector_store:
            logger.error("Cannot create QA chain: missing LLM or vector store")
            return None
        
        try:
            prompt = PromptTemplate(
                template=RAGConfig.PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            
            if RAGConfig.SEARCH_TYPE == "mmr":
                retriever = vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": RAGConfig.RETRIEVAL_K,
                        "fetch_k": RAGConfig.FETCH_K,
                        "lambda_mult": RAGConfig.LAMBDA_MULT
                    }
                )
            else:
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": RAGConfig.RETRIEVAL_K
                    }
                )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "verbose": False
                }
            )
            
            logger.info("QA chain created successfully")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Failed to create QA chain: {e}")
            return None

class StreamlitUI:
    """Streamlit UI components"""
    
    @staticmethod
    def display_header():
        """Display application header"""
        st.markdown('<h1 class="main-header">🤖 RAG Question Answering System</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Upload documents and ask questions - Powered by Llama 2</p>', unsafe_allow_html=True)
        
        config_errors = RAGConfig.validate()
        if config_errors:
            st.error("Configuration errors detected:")
            for error in config_errors:
                st.error(f"• {error}")
    
    @staticmethod
    def display_sidebar():
        """Display sidebar"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            if st.session_state.model_loaded:
                st.success("✅ Model Loaded")
                if st.session_state.model_instance:
                    with st.expander("Model Info"):
                        st.text(f"Type: Llama 2 7B Chat")
                        st.text(f"Max Tokens: {RAGConfig.MAX_NEW_TOKENS}")
                        st.text(f"Temperature: {RAGConfig.TEMPERATURE}")
            else:
                st.warning("⚠️ Model Not Loaded")
                if st.button("🔄 Load Model", type="primary"):
                    with st.spinner("Loading model... This may take 1-2 minutes"):
                        llm = LLMManager.load_llm()
                        if llm:
                            st.session_state.model_loaded = True
                            st.session_state.model_instance = llm
                            if st.session_state.vector_store:
                                qa_chain = LLMManager.create_qa_chain(llm, st.session_state.vector_store)
                                if qa_chain:
                                    st.session_state.qa_chain = qa_chain
                            st.success("Model loaded!")
                            st.rerun()
            
            st.subheader("📄 Upload Documents")
            
            formats_text = "PDF, TXT"
            if DOCX_SUPPORT:
                formats_text += ", DOCX"
            st.caption(f"Supported formats: {formats_text}")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt', 'docx'] if DOCX_SUPPORT else ['pdf', 'txt'],
                accept_multiple_files=True,
                help=f"Max file size: {RAGConfig.MAX_FILE_SIZE // (1024*1024)}MB"
            )
            
            if uploaded_files:
                total_size = sum(file.size for file in uploaded_files)
                st.caption(f"Selected: {len(uploaded_files)} files, {total_size / (1024*1024):.1f}MB")
                
                if st.button("🔄 Process Documents", type="primary"):
                    StreamlitUI.process_uploaded_files(uploaded_files)
            
            if not st.session_state.initialization_complete:
                if os.path.exists(RAGConfig.VECTOR_DB_PATH):
                    with st.spinner("Loading existing database..."):
                        StreamlitUI.load_existing_database()
                st.session_state.initialization_complete = True
            
            st.subheader("📊 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(st.session_state.documents_processed))
                st.metric("Chunks", st.session_state.total_chunks)
            with col2:
                st.metric("Queries", st.session_state.query_count)
                if st.session_state.response_times:
                    avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
                    st.metric("Avg Response", f"{avg_time:.1f}s")
            
            with st.expander("🔧 Advanced Settings"):
                st.markdown("**Retrieval Settings**")
                RAGConfig.RETRIEVAL_K = st.slider(
                    "Documents to retrieve",
                    min_value=1,
                    max_value=10,
                    value=RAGConfig.RETRIEVAL_K
                )
                
                st.markdown("**Display Settings**")
                st.session_state.show_sources = st.checkbox(
                    "Show source references",
                    value=st.session_state.show_sources
                )
                
                if st.button("🗑️ Clear DB"):
                    StreamlitUI.clear_database()
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.query_count = 0
                st.rerun()
    
    @staticmethod
    def process_uploaded_files(uploaded_files):
        """Process uploaded files"""
        with st.spinner("Processing documents..."):
            start_time = time.time()
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            
            try:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                documents = DocumentProcessor.load_documents(file_paths)
                
                if not documents:
                    st.error("No documents could be loaded")
                    return
                
                chunks = DocumentProcessor.process_documents(documents)
                
                if not chunks:
                    st.error("No valid chunks created")
                    return
                
                if not st.session_state.embeddings:
                    st.session_state.embeddings = VectorStoreManager.create_embeddings()
                
                vector_store = VectorStoreManager.create_vector_store(
                    chunks,
                    st.session_state.embeddings
                )
                
                if vector_store:
                    st.session_state.vector_store = vector_store
                    
                    if RAGConfig.AUTO_SAVE:
                        VectorStoreManager.save_vector_store(
                            vector_store,
                            RAGConfig.VECTOR_DB_PATH
                        )
                    
                    st.session_state.documents_processed = [f.name for f in uploaded_files]
                    st.session_state.total_chunks = len(chunks)
                    st.session_state.processing_time = time.time() - start_time
                    
                    if st.session_state.model_loaded and st.session_state.model_instance:
                        qa_chain = LLMManager.create_qa_chain(
                            st.session_state.model_instance,
                            vector_store
                        )
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                    
                    st.success(f"✅ Processed {len(documents)} documents into {len(chunks)} chunks")
                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    @staticmethod
    def load_existing_database():
        """Load existing vector database"""
        try:
            if not st.session_state.embeddings:
                st.session_state.embeddings = VectorStoreManager.create_embeddings()
            
            vector_store = VectorStoreManager.load_vector_store(
                RAGConfig.VECTOR_DB_PATH,
                st.session_state.embeddings
            )
            
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.total_chunks = len(vector_store.index_to_docstore_id)
                st.session_state.documents_processed = ["Previously loaded documents"]
                
                if st.session_state.model_loaded and st.session_state.model_instance:
                    qa_chain = LLMManager.create_qa_chain(
                        st.session_state.model_instance,
                        vector_store
                    )
                    if qa_chain:
                        st.session_state.qa_chain = qa_chain
                
        except Exception as e:
            logger.error(f"Failed to load existing database: {e}")
    
    @staticmethod
    def clear_database():
        """Clear vector database"""
        try:
            if os.path.exists(RAGConfig.VECTOR_DB_DIR):
                shutil.rmtree(RAGConfig.VECTOR_DB_DIR, ignore_errors=True)
            
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.documents_processed = []
            st.session_state.total_chunks = 0
            
            st.success("Database cleared!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error clearing database: {str(e)}")
    
    @staticmethod
    def display_chat_interface():
        """Display chat interface compatible with all Streamlit versions"""
        if not st.session_state.model_loaded:
            st.info("👆 Please load the model from the sidebar to start")
            return
        
        if not st.session_state.vector_store:
            st.info("📄 Please upload and process documents to start asking questions")
            return
        
        st.subheader("💬 Ask Questions")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(
                        f'<div class="chat-message user-message">👤 <b>You:</b> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message bot-message">🤖 <b>Assistant:</b> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                    if st.session_state.show_sources and 'sources' in message and message['sources']:
                        sources_html = '<div class="source-box">📚 <b>Sources:</b><br>'
                        for source in message['sources'][:3]:
                            sources_html += f"• {source}<br>"
                        sources_html += '</div>'
                        st.markdown(sources_html, unsafe_allow_html=True)
        
        # Query input area
        st.markdown("---")
        col1, col2 = st.columns([5, 1])
        
        with col1:
            query_key = f"query_input_{st.session_state.query_count}"
            query = st.text_input(
                "Type your question:",
                placeholder="Ask a question about your documents...",
                key=query_key,
                label_visibility="collapsed"
            )
        
        with col2:
            ask_button = st.button(
                "🔍 Ask",
                type="primary",
                use_container_width=True
            )
        
        # Process query when button is clicked
        if ask_button and query and query.strip():
            if len(query.strip()) < 3:
                st.warning("Please enter a more detailed question")
            else:
                # Process the query
                with st.spinner("🔍 Searching documents and generating response..."):
                    try:
                        start_time = time.time()
                        
                        # Get response from QA chain
                        response = st.session_state.qa_chain.invoke({"query": query})
                        answer = response.get('result', 'I could not generate a response.')
                        
                        # Process sources
                        source_documents = response.get('source_documents', [])
                        sources = []
                        seen_sources = set()
                        
                        for doc in source_documents[:5]:
                            source_file = doc.metadata.get('source_file', 'Unknown')
                            page = doc.metadata.get('page', doc.metadata.get('doc_index', 'N/A'))
                            
                            source_key = f"{source_file}_{page}"
                            if source_key not in seen_sources:
                                seen_sources.add(source_key)
                                sources.append(f"{source_file} (Page/Section: {page})")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": query
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        # Update metrics
                        st.session_state.query_count += 1
                        st.session_state.response_times.append(time.time() - start_time)
                        
                        # Rerun to display the new messages
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

# Main application
def main():
    """Main application entry point"""
    try:
        init_session_state()
        StreamlitUI.display_header()
        StreamlitUI.display_sidebar()
        
        tab1, tab2 = st.tabs(["💬 Chat", "ℹ️ About"])
        
        with tab1:
            StreamlitUI.display_chat_interface()
        
        with tab2:
            st.markdown("""
            ### 🎯 Features
            - Multi-format document support (PDF, TXT, DOCX)
            - Anti-hallucination responses
            - Source attribution
            - Auto-save database
            
            ### 📖 How to Use
            1. Load the model from sidebar
            2. Upload and process documents
            3. Ask questions in the chat
            """)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()