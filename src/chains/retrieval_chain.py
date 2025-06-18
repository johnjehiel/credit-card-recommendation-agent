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
def enhance_query(user_query: str) -> str:
    """
    Enhance the user query to make it more suitable for retrieval.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
        You are an AI assistant that refines user queries to make them optimized for credit card search and retrieval.
        
        User Query: "{user_query}"
        
        Your task is to rewrite this query to:
        1. Make it more direct and specific for credit card search
        2. Include all relevant features and requirements
        3. Add any implied criteria that would be helpful for retrieval
        4. Maintain the original intent and purpose
        
        Respond with ONLY the rewritten query - no explanations or additional text.
    """
    
    response = model.generate_content(prompt)
    enhanced_query = response.text.strip()
    
    # Remove quotation marks if present
    if enhanced_query.startswith('"') and enhanced_query.endswith('"'):
        enhanced_query = enhanced_query[1:-1]
    
    print(f"Enhanced query: {enhanced_query}")
    return enhanced_query


def generate_multi_queries(direct_query: str, n: int = 3) -> list:
    """
    Generate multiple focused subqueries from a user query for better coverage.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
    The following is a detailed credit card search query:
    "{direct_query}"
    
    Generate {n} distinct subqueries that collectively cover all the important features from the original query.
    Each subquery should emphasize a different combination of the features (e.g., lounge access, travel insurance, low foreign transaction fees, hotel discounts, etc.).
    
    Make sure all features from the original query are represented across the {n} queries.
    Output only the subqueries, one per line. Do not include any explanations, numbering, or formatting.
    """
    
    response = model.generate_content(prompt)
    queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
    
    print(f"Generated sub-queries: {queries}")
    return queries

def retrieve_ranked_cards(query, top_n=5):
    """
    Enhanced retrieval pipeline:
    1. Enhance the query for better search
    2. Generate multiple subqueries for broader coverage
    3. Retrieve candidates from each query
    4. Rerank using cross-encoder
    """
    # Step 1: Enhance the original query
    enhanced_query = enhance_query(query)
    
    # Step 2: Generate multiple queries for better coverage
    multi_queries = generate_multi_queries(enhanced_query)
    all_queries = [enhanced_query] + multi_queries
    
    # Step 3: Collect results from all queries
    all_candidates = []
    for sub_query in all_queries:
        candidates = retrieve_from_faiss(sub_query)
        all_candidates.extend(candidates)
    
    # Step 4: Remove duplicates
    seen = set()
    unique_candidates = []
    for card in all_candidates:
        if card["name"] not in seen:
            seen.add(card["name"])
            unique_candidates.append(card)
    
    # Step 5: Final reranking with cross-encoder
    top_cards = rerank_cards(enhanced_query, unique_candidates, top_n=top_n)
    return top_cards
