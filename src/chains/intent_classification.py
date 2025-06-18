import os
import google.generativeai as genai
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Load credit card data for specific card detection
credit_card_df = pd.read_csv("data/credit_card_data_updated.csv")

def classify_query_intent(query: str):
    """
    Classify the user's query intent into retrieve, specific, or no_retrieval.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
        You are a smart financial assistant.
        
        ### User's Query:
        {query}
        
        ### Task:
        Classify the user's intent into one of the following categories:
        1. "retrieve" → If the user is asking for card suggestions, recommendations, or showing cards (e.g., "suggest a card", "need a travel card") OR if they mention their lifestyle, income, spending, or needs (e.g., travel, shopping, fuel, rewards, luxury).
        2. "specific" → If the user is asking about a particular credit card by name (even if the word "card" is not used). Examples: "Tell me about HDFC Regalia", "Is SBI Elite good?".
        3. "no_retrieval" → If the user is asking a general financial question not related to credit card recommendations (e.g., "What is APR?", "How to improve credit score?").
        
        Format your response as a JSON with the following structure:
        ```json
        {{
            "intent": "retrieve|specific|no_retrieval",
            "confidence": 0.0-1.0,
            "card_name": "name of the specific card (only if intent is 'specific')",
            "response": "direct answer (only if intent is 'no_retrieval')"
        }}
        ```
        
        Ensure the JSON is valid and properly formatted with no extra text.
    """
    
    raw_response = model.generate_content(prompt).text.strip()
    
    # Clean any markdown formatting if present
    if raw_response.startswith("```"):
        raw_response = raw_response.strip("`").strip()
    if raw_response.startswith("json"):
        raw_response = raw_response[4:].strip()
    
    try:
        result = json.loads(raw_response)
        return result
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {raw_response}")
        # Return a default response if parsing fails
        return {
            "intent": "retrieve",
            "confidence": 0.5,
            "card_name": None,
            "response": None
        }

def find_fd_intent(query: str) -> bool:
    """
    Determine if query is about first-deposit (FD) cards for beginners/students.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
        You are a helpful assistant. A user has asked the following question or made the following request:
        
        "{query}"
        
        Determine ONLY whether this query is likely about:
        - First-Deposit (FD) backed credit cards
        - Credit cards for beginners or students
        - Cards that don't require a credit score
        - Cards suitable for users with low income
        - Cards for people who are new to credit cards
        - Cards for people with no or low credit score
        
        Respond with just "true" or "false" - no explanation, no extra words.
    """
    
    response = model.generate_content(prompt)
    result = response.text.strip().lower()
    return result == "true"

def find_specific_card(query: str) -> dict:
    """
    Identify if a specific card is mentioned in the query.
    """
    lowered_query = query.lower()
    
    for _, row in credit_card_df.iterrows():
        card_name = row["name"]
        if card_name.lower() in lowered_query:
            return {
                "name": card_name,
                "description": row["description"]
            }
    
    return None

def generate_card_specific_response(query: str, card_info: dict) -> str:
    """
    Generate a response for a query about a specific card.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
        You are a helpful financial assistant. A user has asked about a specific credit card.
        
        Card Name: {card_info.get('name')}
        Description: {card_info.get('description')}
        
        User's Question: {query}
        
        Please provide a concise, relevant answer using the above card context.
        Focus on answering the specific question without unnecessary details.
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()
