# rag_system.py - Fixed version with updated imports
"""
Complete RAG System using Llama 2 for Accurate Document Q&A
Fixed: Using langchain-community for LlamaCpp
"""

import os
import hashlib
import pickle
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import re

# Document Processing
import fitz  # PyMuPDF
from docx import Document as DocxDocument

# ML/NLP Libraries
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Vector Store
import chromadb
from chromadb.config import Settings

# LLM - Updated import
try:
    from langchain_community.llms import LlamaCpp
except ImportError:
    # Fallback for older versions
    from langchain.llms import LlamaCpp

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Text Processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Utilities
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Model paths
    LLAMA_MODEL_PATH: str = "./llama-2-7b-chat.Q4_K_M.gguf"
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"  # Better than MiniLM
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Chunking parameters
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 100
    
    # Retrieval parameters
    TOP_K_RETRIEVAL: int = 10  # Retrieve more initially
    TOP_K_RERANKED: int = 4    # Keep top 4 after reranking
    
    # LLM parameters
    CONTEXT_WINDOW: int = 2048
    MAX_OUTPUT_TOKENS: int = 512
    TEMPERATURE: float = 0.1
    
    # Vector DB
    VECTOR_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "documents"
    
    # Processing
    BATCH_SIZE: int = 32


class DocumentProcessor:
    """Handles document loading and text extraction"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.txt'}
    
    def extract_text(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text and metadata from document"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        metadata = {
            'source': os.path.basename(file_path),
            'file_type': file_ext
        }
        
        if file_ext == '.pdf':
            text = self._extract_from_pdf(file_path)
        elif file_ext == '.docx':
            text = self._extract_from_docx(file_path)
        else:  # .txt
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Clean text
        text = self._clean_text(text)
        metadata['char_count'] = len(text)
        
        return text, metadata
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        text = ""
        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf):
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
        return text
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = DocxDocument(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()


class SmartTextSplitter:
    """Intelligent text splitting that preserves context"""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_documents(self, text: str, metadata: Dict) -> List[Document]:
        """Split text into chunks while preserving metadata"""
        # First split by pages if PDF
        if '[Page' in text:
            chunks = self._split_by_pages(text, metadata)
        else:
            chunks = self._simple_split(text, metadata)
        
        return chunks
    
    def _split_by_pages(self, text: str, metadata: Dict) -> List[Document]:
        """Split PDF text while preserving page information"""
        chunks = []
        pages = re.split(r'\[Page \d+\]', text)
        page_numbers = re.findall(r'\[Page (\d+)\]', text)
        
        for i, page_text in enumerate(pages[1:], 1):  # Skip empty first split
            if page_text.strip():
                page_num = page_numbers[i-1] if i-1 < len(page_numbers) else i
                page_chunks = self.splitter.split_text(page_text)
                
                for j, chunk in enumerate(page_chunks):
                    chunk_metadata = {
                        **metadata,
                        'page': int(page_num),
                        'chunk_index': j,
                        'chunk_total': len(page_chunks)
                    }
                    chunks.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        return chunks
    
    def _simple_split(self, text: str, metadata: Dict) -> List[Document]:
        """Simple splitting for non-PDF documents"""
        text_chunks = self.splitter.split_text(text)
        chunks = []
        
        for i, chunk in enumerate(text_chunks):
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'chunk_total': len(text_chunks)
            }
            chunks.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        return chunks


class EnhancedRetriever:
    """Advanced retriever with semantic search and reranking"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Initialize reranker
        print("Loading reranking model...")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)
        
        # Initialize ChromaDB
        print("Initializing vector database...")
        self.chroma_client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.chroma_client.get_collection(config.COLLECTION_NAME)
    
    def index_documents(self, documents: List[Document], batch_size: int = 32):
        """Index documents in the vector store"""
        print(f"Indexing {len(documents)} chunks...")
        
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i + batch_size]
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            # Generate embeddings
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            
            # Generate IDs
            ids = [self._generate_id(text) for text in texts]
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents with reranking"""
        if k is None:
            k = self.config.TOP_K_RETRIEVAL
        
        # Get query embedding
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        if not results['documents'][0]:
            return []
        
        # Prepare documents for reranking
        documents = []
        for i in range(len(results['documents'][0])):
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=results['metadatas'][0][i] if results['metadatas'] else {}
            )
            documents.append(doc)
        
        # Rerank documents
        reranked_docs = self._rerank(query, documents)
        
        return reranked_docs[:self.config.TOP_K_RERANKED]
    
    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder"""
        if not documents:
            return []
        
        # Prepare pairs for reranking
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Sort by scores
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_scores]
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for document chunk"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def clear_index(self):
        """Clear the vector store"""
        self.chroma_client.delete_collection(self.config.COLLECTION_NAME)
        self.collection = self.chroma_client.create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


class OptimizedLlamaQA:
    """Optimized Llama 2 for question answering"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Callback manager for streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        print("Loading Llama 2 model...")
        self.llm = LlamaCpp(
            model_path=config.LLAMA_MODEL_PATH,
            n_ctx=config.CONTEXT_WINDOW,
            n_batch=512,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=0.95,
            n_threads=8,
            callback_manager=callback_manager,
            verbose=False
        )
    
    def generate_answer(self, query: str, context: List[Document]) -> Tuple[str, List[Dict]]:
        """Generate answer using retrieved context"""
        
        if not context:
            return "I couldn't find relevant information to answer your question.", []
        
        # Prepare context string
        context_str = self._format_context(context)
        
        # Create prompt
        prompt = self._create_prompt(query, context_str)
        
        # Generate answer
        answer = self.llm(prompt)
        
        # Extract sources
        sources = self._extract_sources(context)
        
        return answer.strip(), sources
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            context_part = f"[Source {i}: {source}, Page {page}]\n{doc.page_content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create optimized prompt for Llama 2"""
        prompt = f"""[INST] <<SYS>>
You are a helpful assistant that answers questions based ONLY on the provided context. 
Your answers must be accurate, specific, and directly based on the information given.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite the source and page number when providing information.
<</SYS>>

Context:
{context}

Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Be specific and accurate
3. Cite sources with [Source X, Page Y] format
4. If information is not in context, say "The provided documents do not contain this information"
5. Keep the answer concise but complete

Answer: [/INST]"""
        
        return prompt
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict]:
        """Extract source information from documents"""
        sources = []
        seen = set()
        
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            key = f"{source}_{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    'source': source,
                    'page': page,
                    'preview': doc.page_content[:200] + "..."
                })
        
        return sources


class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.text_splitter = SmartTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        self.retriever = EnhancedRetriever(self.config)
        self.qa_system = OptimizedLlamaQA(self.config)
        
        print("RAG Pipeline initialized successfully!")
    
    def ingest_document(self, file_path: str):
        """Process and index a single document"""
        print(f"Processing document: {file_path}")
        
        # Extract text
        text, metadata = self.doc_processor.extract_text(file_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(text, metadata)
        
        # Index chunks
        self.retriever.index_documents(chunks)
        
        print(f"Successfully indexed {len(chunks)} chunks from {file_path}")
        return len(chunks)
    
    def ingest_multiple_documents(self, file_paths: List[str]):
        """Process and index multiple documents"""
        total_chunks = 0
        
        for file_path in file_paths:
            try:
                chunks = self.ingest_document(file_path)
                total_chunks += chunks
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Total chunks indexed: {total_chunks}")
        return total_chunks
    
    def query(self, question: str) -> Tuple[str, List[Dict]]:
        """Answer a question using RAG"""
        print(f"\nQuery: {question}")
        
        # Retrieve relevant documents
        print("Retrieving relevant documents...")
        relevant_docs = self.retriever.retrieve(question)
        
        if not relevant_docs:
            return "No relevant information found in the documents.", []
        
        print(f"Found {len(relevant_docs)} relevant chunks after reranking")
        
        # Generate answer
        print("Generating answer...")
        answer, sources = self.qa_system.generate_answer(question, relevant_docs)
        
        return answer, sources
    
    def clear_index(self):
        """Clear the document index"""
        self.retriever.clear_index()
        print("Document index cleared")


# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Example: Ingest documents
    # documents = ["path/to/doc1.pdf", "path/to/doc2.docx"]
    # rag.ingest_multiple_documents(documents)
    
    # Example: Query
    # answer, sources = rag.query("What is the main topic discussed in the document?")
    # print(f"Answer: {answer}")
    # print(f"Sources: {sources}")