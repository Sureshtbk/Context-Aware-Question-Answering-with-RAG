# app.py - Fixed version without infinite loop
"""
Streamlit interface for the RAG Document Q&A System - Fixed
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import time
from typing import List, Dict, Tuple
import json
import shutil

# Import the RAG system
from rag_system import RAGPipeline, Config

# Page configuration
st.set_page_config(
    page_title="📚 Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-weight: 600;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 4px solid #4CAF50;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .metric-box {
        text-align: center;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("🚀 Initializing RAG System..."):
            config = Config()
            
            # Check model file
            if not os.path.exists(config.LLAMA_MODEL_PATH):
                st.error(f"⚠️ Model file not found: {config.LLAMA_MODEL_PATH}")
                st.info("Please ensure 'llama-2-7b-chat.Q4_K_M.gguf' is in the current directory")
                st.stop()
            
            try:
                st.session_state.rag_pipeline = RAGPipeline(config)
            except Exception as e:
                st.error(f"Failed to initialize: {str(e)}")
                st.stop()
    
    if 'indexed_documents' not in st.session_state:
        st.session_state.indexed_documents = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'total_chunks' not in st.session_state:
        st.session_state.total_chunks = 0
    
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0


def main():
    """Main application"""
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Upload your document")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Limit 200MB per file • PDF, DOCX, TXT"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Process", type="primary", disabled=not uploaded_files):
                process_documents(uploaded_files)
        
        with col2:
            if st.button("Clear All"):
                clear_all()
        
        # Show indexed documents
        if st.session_state.indexed_documents:
            st.success(f"### ✅ {len(st.session_state.indexed_documents)} Documents Indexed")
            st.info(f"**Total Chunks:** {st.session_state.total_chunks}")
            
            with st.expander("Document Details"):
                for doc in st.session_state.indexed_documents:
                    st.write(f"• {doc['name']} ({doc['chunks']} chunks)")
        else:
            st.info("No documents indexed yet. Upload and process documents to begin.")
        
        # Advanced Settings
        with st.expander("Advanced Settings"):
            st.slider("Sources to retrieve", 1, 10, 4, key="top_k")
            st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, key="temperature")
    
    # Main content area
    st.title("Context-Aware Question Answering with RAG")
    st.caption("Used by Llama 2 model and RAG Technology")
    
    # Check if documents are indexed
    if not st.session_state.indexed_documents:
        st.markdown("""
        ### Getting Started
        1. Upload your documents using the sidebar
        2. Click "Process" to index them
        3. Start asking questions!
        
        The system will search through your documents and provide accurate, source-backed answers.
        """)
        return
    
    # Query interface
    st.markdown("---")
    st.subheader("Ask Your Question")
    
    # Query input
    query = st.text_input(
        "What would you like to know?",
        placeholder="Enter your question about the documents...",
        label_visibility="collapsed"
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Search", type="primary", disabled=not query):
            process_query(query)
    
    # Sample questions
    st.markdown("####  Try these questions:")
    
    sample_questions = [
        "What is the main purpose of this document?",
        "Summarize the key points",
        "What are the technical requirements?",
        "List the main features"
    ]
    
    cols = st.columns(len(sample_questions))
    for i, q in enumerate(sample_questions):
        with cols[i]:
            if st.button(q, key=f"sample_{i}"):
                process_query(q)
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader(" Recent Conversations")
        
        for item in reversed(st.session_state.chat_history[-3:]):
            with st.container():
                # Question
                st.markdown(f"** Question:** {item['question']}")
                
                # Answer
                st.markdown(f"""
                <div class="answer-box">
                {item['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Sources
                if item['sources']:
                    with st.expander(f" View {len(item['sources'])} Sources"):
                        for src in item['sources']:
                            st.markdown(f"""
                            <div class="source-card">
                            <b>{src['source']}</b> - Page {src['page']}<br>
                            <small>{src['preview']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ {item['response_time']:.2f}s")
                with col2:
                    st.caption(f"📄 {len(item['sources'])} sources")
                with col3:
                    st.caption(f"🕐 {item['timestamp']}")
                
                st.markdown("---")


def process_documents(uploaded_files):
    """Process uploaded documents without infinite loop"""
    if not uploaded_files:
        return
    
    # Use a placeholder for updates
    placeholder = st.empty()
    
    with placeholder.container():
        progress_bar = st.progress(0)
        status = st.info(" Starting document processing...")
        
        temp_dir = tempfile.mkdtemp()
        total = len(uploaded_files)
        successful = 0
        
        try:
            for i, file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / total
                progress_bar.progress(progress)
                status.info(f"📄 Processing: {file.name}")
                
                # Save temp file
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, 'wb') as f:
                    f.write(file.getbuffer())
                
                # Process document
                try:
                    chunks = st.session_state.rag_pipeline.ingest_document(temp_path)
                    
                    # Update session state
                    st.session_state.indexed_documents.append({
                        'name': file.name,
                        'chunks': chunks,
                        'timestamp': time.strftime("%H:%M:%S")
                    })
                    st.session_state.total_chunks += chunks
                    successful += 1
                    
                except Exception as e:
                    st.error(f"Error with {file.name}: {str(e)}")
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Show final status
            if successful > 0:
                status.success(f"✅ Processing Complete\n• Successful: {successful} documents\n• Failed: {total - successful} documents\n• Total Chunks: {st.session_state.total_chunks}")
            else:
                status.error("❌ No documents were processed successfully")
            
            # Clean progress bar after 2 seconds
            time.sleep(2)
            progress_bar.empty()
            
        except Exception as e:
            status.error(f"Critical error: {str(e)}")
        
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Clear the placeholder after processing
    time.sleep(1)
    placeholder.empty()


def process_query(query: str):
    """Process user query"""
    if not query:
        return
    
    with st.spinner(f"🔍 Searching for: {query}"):
        try:
            start_time = time.time()
            
            # Get answer
            answer, sources = st.session_state.rag_pipeline.query(query)
            
            # Calculate metrics
            response_time = time.time() - start_time
            st.session_state.query_count += 1
            
            # Add to history
            st.session_state.chat_history.append({
                'question': query,
                'answer': answer,
                'sources': sources,
                'timestamp': time.strftime("%H:%M:%S"),
                'response_time': response_time
            })
            
            # Display results
            st.success("✅ Answer found!")
            
            # Show answer
            st.markdown("### Answer")
            st.markdown(f"""
            <div class="answer-box">
            {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources
            if sources:
                st.markdown("### Sources")
                cols = st.columns(min(len(sources), 2))
                for i, src in enumerate(sources):
                    with cols[i % 2]:
                        st.info(f"**{src['source']}**\nPage {src['page']}")
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Time", f"{response_time:.2f}s")
            with col2:
                st.metric("Sources Used", len(sources))
            with col3:
                st.metric("Query #", st.session_state.query_count)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")


def clear_all():
    """Clear all documents and history"""
    try:
        st.session_state.rag_pipeline.clear_index()
        st.session_state.indexed_documents = []
        st.session_state.total_chunks = 0
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.success("✅ All data cleared!")
    except Exception as e:
        st.error(f"Error clearing: {str(e)}")


if __name__ == "__main__":
    main()