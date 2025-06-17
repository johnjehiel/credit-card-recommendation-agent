import os
import numpy as np
import faiss
import pandas as pd
import google.generativeai as genai
from collections import defaultdict
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()
# Gemini config for embeddings
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_name = "models/text-embedding-004"

# Load card description dataset
df = pd.read_csv("data/credit_card_data_updated.csv")
card_descriptions = dict(zip(df["name"], df["description"]))

# Cross-Encoder reranker
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def chunk_text(text, chunk_size=1):
    sentences = text.split("; ")
    return ["; ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]


def get_gemini_embeddings(text_list):
    embeddings = []
    for text in text_list:
        response = genai.embed_content(
            model=model_name,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings.append(np.array(response["embedding"], dtype=np.float32))
    return np.vstack(embeddings)


# Initialization of FAISS index
chunk_texts = []
chunk_name_mapping = {}
for card_name, desc in card_descriptions.items():
    chunks = chunk_text(desc)
    for chunk in chunks:
        idx = len(chunk_texts)
        chunk_texts.append(chunk)
        chunk_name_mapping[idx] = card_name

chunk_embeddings = get_gemini_embeddings(chunk_texts)
faiss.normalize_L2(chunk_embeddings)
dim = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(chunk_embeddings)


def retrieve_from_faiss(user_query: str, top_k=15):
    query_embedding = genai.embed_content(
        model=model_name,
        content=user_query,
        task_type="RETRIEVAL_QUERY"
    )["embedding"]
    query_embedding = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    D, I = faiss_index.search(np.expand_dims(query_embedding, axis=0), top_k * 5)
    similarity_scores = D[0]

    card_similarity = defaultdict(float)
    card_dict = {card["name"]: card for card in df.to_dict(orient="records")}
    unique_cards = {}

    for i, chunk_idx in enumerate(I[0]):
        if chunk_idx == -1:
            continue
        card_name = chunk_name_mapping[chunk_idx]
        if card_name not in unique_cards or similarity_scores[i] > card_similarity[card_name]:
            card_similarity[card_name] = similarity_scores[i]
            unique_cards[card_name] = {
                "name": card_name,
                "description": card_dict[card_name]["description"],
                "similarity": similarity_scores[i]
            }

    ordered = sorted(unique_cards.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]
    return ordered


def rerank_cards(query: str, cards: list, top_n: int = 5):
    pairs = [[query, c["description"]] for c in cards]
    scores = cross_encoder.predict(pairs)
    if len(scores) == 0 or (max(scores) - min(scores)) == 0:
        return cards[:top_n]
    norm_scores = (np.array(scores) - np.min(scores)) / (np.max(scores) - np.min(scores))
    sorted_cards = sorted(zip(norm_scores, cards), key=lambda x: x[0], reverse=True)
    return [c for _, c in sorted_cards[:top_n]]


# Unified retrieval call

def retrieve_ranked_cards(query):
    candidates = retrieve_from_faiss(query)
    top_cards = rerank_cards(query, candidates, top_n=5)
    return top_cards
