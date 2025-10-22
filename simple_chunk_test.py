#!/usr/bin/env python3
"""
Simple test for TextChunker without pandas dependency
"""

def clean_text(text: str) -> str:
    """Clean a string by removing extra spaces, URLs, and non-alphanumeric characters."""
    import re
    text = re.sub(r'\s+', ' ', text)          
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

class TextChunker:
    """Découpe un texte long en plusieurs morceaux (chunks) pour le passage à l'embedding."""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        :param chunk_size: nombre de mots par chunk
        :param overlap: nombre de mots partagés entre deux chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> list:
        """
        Découpe le texte en plusieurs chunks avec chevauchement.
        :param text: texte à découper
        :return: liste de chunks
        """
        if not isinstance(text, str) or not text.strip():
            return []

        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])

            if len(chunk.split()) > 10:  # éviter les mini-chunks
                chunks.append(chunk)

            if end >= len(words):
                break  # fin propre

            start += self.chunk_size - self.overlap  # avance avec chevauchement

        return chunks

def test_chunker():
    """Test the TextChunker functionality"""
    
    chunker = TextChunker(chunk_size=100, overlap=20)  # Smaller chunks for testing
    
    # Sample news text
    sample_text = """
    Artificial intelligence has become one of the most transformative technologies of our time. 
    From machine learning algorithms that power recommendation systems to deep learning models 
    that enable autonomous vehicles, AI is reshaping industries across the globe. The healthcare 
    sector has particularly benefited from AI innovations, with diagnostic tools that can detect 
    diseases earlier and more accurately than traditional methods. In finance, AI algorithms 
    are used for fraud detection, algorithmic trading, and risk assessment. The transportation 
    industry is being revolutionized by autonomous vehicles and smart traffic management systems. 
    However, the rapid advancement of AI also brings challenges, including concerns about job 
    displacement, algorithmic bias, and the need for robust ethical frameworks. Governments 
    and organizations worldwide are grappling with how to regulate AI while fostering innovation. 
    The future of AI holds immense potential, but it requires careful consideration of its 
    societal implications and responsible development practices.
    """
    
    print("=== TextChunker Test ===")
    print(f"Original text: {len(sample_text.split())} words")
    print(f"Chunk size: {chunker.chunk_size}")
    print(f"Overlap: {chunker.overlap}")
    print("=" * 50)
    
    chunks = chunker.split_text(sample_text)
    
    print(f"Number of chunks created: {len(chunks)}")
    print("=" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        words = chunk.split()
        print(f"Chunk {i}: {len(words)} words")
        print(f"Content: '{chunk[:80]}...'")
        print("-" * 40)
    
    # Test overlap verification
    if len(chunks) > 1:
        print("\n=== Overlap Verification ===")
        for i in range(len(chunks) - 1):
            current_words = chunks[i].split()
            next_words = chunks[i + 1].split()
            
            # Find common words at the end of current chunk and start of next chunk
            overlap_found = 0
            for j in range(min(20, len(current_words))):  # Check last 20 words
                if j < len(next_words) and current_words[-(j+1)] == next_words[j]:
                    overlap_found += 1
            
            print(f"Chunks {i+1} and {i+2}: {overlap_found} overlapping words")

if __name__ == "__main__":
    test_chunker()
