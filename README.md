🌟 Project Overview
This project implements a state-of-the-art Retrieval-Augmented Generation (RAG) system that combines the power of Large Language Models (LLMs) with intelligent document retrieval to provide accurate, context-aware answers from uploaded documents. Unlike traditional search engines or standalone LLMs, this system grounds its responses in actual document content, eliminating hallucinations and ensuring factual accuracy.
🔍 Problem Statement
Challenge
Traditional document search and question-answering systems face significant limitations:

Keyword-based searches often miss semantically relevant content
Standalone LLMs may hallucinate or provide outdated information
Manual document review is time-consuming and inefficient
Context preservation is lost in conventional search systems

Objective
Build an intelligent RAG system that:

Processes multiple document formats (PDF, DOCX, TXT)
Understands semantic meaning beyond keywords
Provides accurate answers with source citations
Maintains context across document chunks
Operates efficiently with local resources
 Solution Approach
Step-by-Step Implementation
Step 1: Data Ingestion ✅

Accepts PDF, DOCX, and TXT documents
Extracts text using PyMuPDF for PDFs and python-docx for Word files
Preserves document structure and metadata

Step 2: Text Embeddings & Indexing ✅

Converts text into high-dimensional vector embeddings
Uses SentenceTransformers (all-mpnet-base-v2) for semantic understanding
Stores vectors in ChromaDB for efficient similarity search

Step 3: Query Processing & Retrieval ✅

Transforms user queries into embeddings
Performs cosine similarity search in vector space
Implements two-stage retrieval with reranking for precision

Step 4: Response Generation ✅

Integrates Llama 2 (7B parameters) for natural language generation
Uses prompt engineering to ensure factual, grounded responses
Includes source attribution for transparency

Step 5: Evaluation & Optimization ✅

Implements accuracy metrics and response time tracking
Provides confidence scoring for answers
Includes comprehensive test suite

📦 Installation
Prerequisites

Python 3.8 or higher
8GB RAM minimum (16GB recommended)
5GB disk space for model storage
Git for version control

Step-by-Step Setup

Clone the Repository

bashgit clone https://github.com/yourusername/rag-qa-system.git
cd rag-qa-system

Create Virtual Environment

bashpython -m venv rag_env

# Windows
rag_env\Scripts\activate

# Linux/macOS
source rag_env/bin/activate

Install Dependencies

bashpip install -r requirements.txt

Download Llama 2 Model

bash# Option 1: Using wget (Linux/macOS)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# Option 2: Manual download
# Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# Download: llama-2-7b-chat.Q4_K_M.gguf (4.08 GB)
# Place in project root directory

Verify Installation

bashpython test_rag.py

Launch Application

bashstreamlit run app.py
Navigate to http://localhost:8501 in your browser.

📖 Usage Guide
Quick Start

Upload Documents

Click "Browse files" in the sidebar
Select PDF, DOCX, or TXT files
Multiple files can be uploaded simultaneously


Process Documents

Click "Process" button
Wait for indexing to complete (progress bar shows status)
Check "Indexed Documents" section for confirmation


Ask Questions

Type your question in the main interface
Click "Search" or press Enter
View answer with source citations


Review Sources

Expand source cards to see relevant excerpts
Check confidence scores for reliability
Export chat history if needed



Example Questions

"What is the main purpose of this document?"
"Summarize the key findings in the report"
"What are the technical requirements mentioned?"
"List all stakeholders and their roles"
"What are the project deliverables and timelines?"
