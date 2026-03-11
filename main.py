from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

sentences = [
    "Na mojej stronie znajdziesz informacje o mnie, blogi, mój GitHub oraz LinkedIn",
    "Blogi znajdują się na górnym pasku menu pod nazwą 'Blog'",
    "Uwielbiam jeść pizzę",
    "Strona została napisana w Django"
]

sentences_embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

index = sentences_embeddings.shape[1]
db_index = faiss.IndexFlatIP(index)
db_index.add(sentences_embeddings)
print(f"Numbers of vectors in database: {db_index.ntotal}")

query = "Gdzie znajdę twoje artykuły i wpisy?"

query_embeddings = model.encode(
    [query],
    normalize_embeddings=True
)


distances, indexes = db_index.search(query_embeddings, k=2)

print(f"Query: {query}")
print("The best results: ")
for i in range(2):
    nr_of_sentence = indexes[0][i]
    certainty = distances[0][i]
    print(f"{i+1}, {sentences[nr_of_sentence]} and certainty {certainty}")

