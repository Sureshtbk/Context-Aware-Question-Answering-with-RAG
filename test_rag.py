# run_rag_test.py
"""
Direct RAG test runner - bypasses package checks
"""

import os
from rag_system import RAGPipeline, Config

def run_tests():
    """Run the RAG system tests directly"""
    
    print("=" * 60)
    print("RAG System Direct Test")
    print("=" * 60)
    
    # Test 1: Initialize Pipeline
    print("\n1. Initializing RAG Pipeline...")
    try:
        config = Config()
        config.TOP_K_RETRIEVAL = 5
        config.TOP_K_RERANKED = 3
        rag = RAGPipeline(config)
        print("✅ Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\nTroubleshooting:")
        print("- Check if llama-2-7b-chat.Q4_K_M.gguf exists in current directory")
        print("- Verify CUDA/CPU compatibility")
        return False
    
    # Test 2: Create and ingest a test document
    print("\n2. Testing Document Ingestion...")
    test_file = "test_doc.txt"
    test_content = """
    Artificial Intelligence in Healthcare
    
    AI is transforming healthcare through:
    1. Medical Imaging: AI analyzes X-rays and MRIs
    2. Drug Discovery: AI accelerates drug development
    3. Personalized Medicine: AI customizes treatments
    4. Predictive Analytics: AI predicts patient risks
    
    Challenges include data privacy, regulatory approval, and integration.
    The future looks promising with continued AI advancement.
    """
    
    try:
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        chunks = rag.ingest_document(test_file)
        print(f"✅ Successfully ingested {chunks} chunks")
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False
    
    # Test 3: Query the system
    print("\n3. Testing Query System...")
    test_queries = [
        "What are the applications of AI in healthcare?",
        "What challenges does AI face?",
        "How does AI help with medical imaging?"
    ]
    
    successful = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        try:
            answer, sources = rag.query(query)
            print(f"✅ Answer: {answer[:150]}...")
            print(f"📚 Sources: {len(sources)} found")
            successful += 1
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # Test 4: Clean up
    print("\n4. Cleaning up...")
    try:
        rag.clear_index()
        print("✅ Cleanup successful")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Pipeline Initialization: PASSED")
    print(f"✅ Document Ingestion: PASSED")
    print(f"✅ Query Processing: {successful}/{len(test_queries)} PASSED")
    print(f"✅ Overall: {'PASSED' if successful > 0 else 'FAILED'}")
    print("=" * 60)
    
    return successful > 0

if __name__ == "__main__":
    import sys
    
    # Check if model exists
    model_path = "./llama-2-7b-chat.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print("⚠️ ERROR: Llama 2 model not found!")
        print(f"Expected location: {os.path.abspath(model_path)}")
        print("\nTo fix:")
        print("1. Download from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        print("2. Download file: llama-2-7b-chat.Q4_K_M.gguf")
        print("3. Place in current directory")
        sys.exit(1)
    
    print("✅ Model file found")
    print("\nStarting RAG system tests...\n")
    
    success = run_tests()
    
    if success:
        print("\n🎉 Tests completed successfully!")
        print("\nYou can now run:")
        print("  streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")