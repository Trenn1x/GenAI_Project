import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data
documents = [
    "Generative AI helps automate complex tasks by generating human-like text.",
    "Vector databases allow for efficient search and retrieval of embedded data.",
    "Retrieval-Augmented Generation combines data retrieval with language generation."
]

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

# Create a FAISS index
d = doc_vectors.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(doc_vectors).astype('float32'))

# Define a simple summarization function
def summarize_text(text):
    # Split the text into sentences
    sentences = text.split(". ")
    # Return the longest sentence as a "summary"
    summary = max(sentences, key=len)
    return summary

def process_query(query):
    # Convert query to vector
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    # Search for the closest document
    D, I = index.search(query_vector, k=1)
    closest_doc_idx = I[0][0]

    if closest_doc_idx != -1:
        closest_document = documents[closest_doc_idx]
        print(f"Closest document: {closest_document}")

        # Generate a summary
        summary = summarize_text(closest_document)
        print(f"Generated Summary: {summary}")
    else:
        print("No relevant documents found.")

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    process_query(user_query)

