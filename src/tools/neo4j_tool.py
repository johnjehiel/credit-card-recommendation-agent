from langchain.tools import BaseTool
from neo4j import GraphDatabase, Driver
import google.generativeai as genai
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Neo4jRetrievalTool(BaseTool):
    name: str = "neo4j_card_retriever"
    description: str = "Tool to query credit card recommendations from Neo4j using Gemini-generated Cypher"
    driver: Optional[Driver] = None

    def __init__(self):
        super().__init__()
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))
        )

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

    def _run(self, query_text: str, query_intent: bool = False, include_cobranded: bool = True):
        cypher = self.generate_cypher(query_text, query_intent, include_cobranded)
        print("Generated Cypher:\n", cypher)
        with self.driver.session() as session:
            result = session.run(cypher)
            return [record["c"] for record in result]
