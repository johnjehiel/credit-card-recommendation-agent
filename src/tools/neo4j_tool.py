from langchain.tools import BaseTool
from neo4j import GraphDatabase, Driver
import google.generativeai as genai
import os
import pandas as pd
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class Neo4jRetrievalTool(BaseTool):
    name: str = "neo4j_card_retriever"
    description: str = "Tool to query credit card recommendations from Neo4j using Gemini-generated Cypher"
    driver: Optional[Driver] = None
    eligibility_df: Optional[pd.DataFrame] = None

    def __init__(self):
        super().__init__()
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))
        )
        # Load eligibility data
        self.eligibility_df = pd.read_csv("data/cards_eligibility_updated.csv")

    def generate_cypher(self, user_query: str, query_intent: bool, include_cobranded: bool) -> str:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        context_flags = f"""
        Contextual Flags:
        - FD Card intent: {query_intent}
        - Include co-branded cards: {include_cobranded}
        """

        with open("src/prompts/cypher_template.txt", encoding="utf-8") as f:
            prompt = f.read()

        cypher_prompt = prompt.format(context_flags=context_flags, user_query=user_query)
        cypher_code = model.generate_content(cypher_prompt).text.strip()

        return cypher_code.strip("`").replace("cypher", "").strip()

    def _run(self, query_text: str, query_intent: bool = False, include_cobranded: bool = True, 
             use_eligibility: bool = False, min_income: float = 0, min_cibil: int = 0, 
             age: int = 25, min_joining_fee: int = 0, max_joining_fee: int = 100000,
             min_annual_fee: int = 0, max_annual_fee: int = 100000):
        """
        Run the Neo4j retrieval tool with eligibility filtering
        
        Parameters:
        - query_text: User's question about credit cards
        - query_intent: Whether to focus on first-time/student cards
        - include_cobranded: Whether to include co-branded cards
        - use_eligibility: Whether to apply eligibility filtering
        - min_income: Minimum income (LPA)
        - min_cibil: Minimum CIBIL score
        - age: User's age
        - min_joining_fee/max_joining_fee: Range for joining fee
        - min_annual_fee/max_annual_fee: Range for annual fee
        """
        cypher = self.generate_cypher(query_text, query_intent, include_cobranded)
        print("Generated Cypher:\n", cypher)
        cards = []
        
        with self.driver.session() as session:
            result = session.run(cypher)
            cards = [record["c"] for record in result]
        
        # Apply eligibility filtering if enabled
        if use_eligibility and self.eligibility_df is not None:
            filtered_cards = self.eligibility_filter(
                cards, min_income, min_cibil, age, 
                min_joining_fee, max_joining_fee, 
                min_annual_fee, max_annual_fee
            )
            return filtered_cards
        
        return cards
    
    def eligibility_filter(self, cards: List[Dict[str, Any]], user_income: float, user_cibil: int, 
                          user_age: int, min_joining_fee: int, max_joining_fee: int,
                          min_annual_fee: int, max_annual_fee: int) -> List[Dict[str, Any]]:
        """Filter cards based on eligibility criteria"""
        eligible_cards = []
        print("Applying eligibility filter")
        
        for card in cards:
            card_name = card.get("name", "")
            
            # Find matching card in eligibility data
            card_eligibility = self.eligibility_df[self.eligibility_df['Name'] == card_name]
            
            if card_eligibility.empty:
                # If no eligibility data found, include the card
                eligible_cards.append(card)
                continue
                
            # Get eligibility criteria for the card
            min_age = card_eligibility['Minimum Age'].values[0]
            max_age = card_eligibility['Maximum Age'].values[0]
            min_income_required = card_eligibility['Minimum Income (LPA)'].values[0]
            min_credit_score = card_eligibility['Minimum Credit Score'].values[0]
            joining_fee = card_eligibility['Joining fee'].values[0]
            annual_fee = card_eligibility['Annual fee'].values[0]
            
            # Check if user meets eligibility criteria
            if (user_income >= min_income_required and
                user_cibil >= min_credit_score and
                min_age <= user_age <= max_age and
                min_joining_fee <= joining_fee <= max_joining_fee and
                min_annual_fee <= annual_fee <= max_annual_fee):
                eligible_cards.append(card)
        
        return eligible_cards
