#!/usr/bin/env python3
"""
Test script to verify TextChunker functionality
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import TextChunker

def test_text_chunker():
    """Test the TextChunker class with sample text"""
    
    # Initialize chunker with default parameters
    chunker = TextChunker(chunk_size=300, overlap=50)
    
    # Sample text for testing
    sample_text = """
    This is a sample news article about artificial intelligence and machine learning. 
    The technology has advanced significantly over the past decade, with applications 
    in healthcare, finance, transportation, and many other industries. Machine learning 
    algorithms can now process vast amounts of data and make predictions with high accuracy. 
    Deep learning, a subset of machine learning, has revolutionized fields like computer vision 
    and natural language processing. Companies are investing heavily in AI research and development, 
    leading to breakthrough innovations. However, there are also concerns about the ethical implications 
    of AI, including bias in algorithms and job displacement. Governments and organizations worldwide 
    are working on establishing frameworks for responsible AI development and deployment. The future 
    of AI holds both tremendous promise and significant challenges that society must address.
    """
    
    print("Testing TextChunker...")
    print(f"Original text length: {len(sample_text.split())} words")
    print(f"Chunk size: {chunker.chunk_size}")
    print(f"Overlap: {chunker.overlap}")
    print("-" * 50)
    
    # Test chunking
    chunks = chunker.split_text(sample_text)
    
    print(f"Number of chunks created: {len(chunks)}")
    print("-" * 50)
    
    # Display each chunk
    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        print(f"Chunk {i} ({word_count} words):")
        print(f"'{chunk[:100]}...'")  # Show first 100 characters
        print("-" * 30)
    
    # Test edge cases
    print("\nTesting edge cases:")
    
    # Empty text
    empty_chunks = chunker.split_text("")
    print(f"Empty text chunks: {empty_chunks}")
    
    # Very short text
    short_chunks = chunker.split_text("This is a short text.")
    print(f"Short text chunks: {short_chunks}")
    
    # Text with only whitespace
    whitespace_chunks = chunker.split_text("   \n\t   ")
    print(f"Whitespace text chunks: {whitespace_chunks}")

if __name__ == "__main__":
    test_text_chunker()
