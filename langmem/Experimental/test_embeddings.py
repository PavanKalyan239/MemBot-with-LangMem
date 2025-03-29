from sentence_transformers import SentenceTransformer

# Load model (downloads ~420MB first time)
model = SentenceTransformer('all-mpnet-base-v2')

# Test it
text = "I like Python"
embedding = model.encode(text)
print(f"Embedding shape: {embedding.shape}")  # Should be (768,)
print(f"First few values: {embedding[:5]}")