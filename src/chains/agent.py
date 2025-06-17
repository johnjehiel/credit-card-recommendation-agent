import os
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool  # Add this import
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any

from src.tools.neo4j_tool import Neo4jRetrievalTool
from src.chains.retrieval_chain import retrieve_ranked_cards
from dotenv import load_dotenv

load_dotenv()

# Gemini Chat LLM (LangChain wrapper)
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    convert_system_message_to_human=True
)

# Memory store for agent (keeps chat history)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input"
)

# Wrapper function for Gradio to call
def run_agent_pipeline(query: str, query_intent: bool = False, include_cobranded: bool = True):
    """Run the agent pipeline to get credit card recommendations."""
    
    # Initialize tools
    neo4j_tool = Neo4jRetrievalTool()
    
    def retrieval_tool(query_text: str) -> List[Dict[str, Any]]:
        """Get credit card recommendations using vector search."""
        cards = retrieve_ranked_cards(query_text)
        return cards
    
    # Create a proper Tool object instead of using a dictionary
    vector_tool = Tool(
        name="vector_card_retriever",
        description="Use this to retrieve credit cards using vector search based on descriptions.",
        func=retrieval_tool
    )
    
    tools = [neo4j_tool, vector_tool]

    system_prompt = SystemMessage(
    content="""
    You are a financial card assistant. You have access to tools that let you:
    - retrieve credit cards from a knowledge graph (Neo4j)
    - retrieve cards from a vector search index (FAISS)

    Instructions:
    - ALWAYS use the tools to fetch card results. NEVER make up cards.
    - For low credit score / student queries, prefer graph-based FD cards.
    - Summarize and explain cards only from the tool outputs.
    - If the graph returns empty or low results, use the vector retriever.
    - Do not assume card benefits unless they are explicitly present.
        """
    )
    
    # Initialize agent
    agent_executor = initialize_agent(
        tools,
        gemini_llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_prompt}
    )
    
    # Process combined input
    combined_input = f"User query: {query}"
    result = agent_executor.invoke({
        "input": combined_input,  # This is the key that memory will use
        "query_intent": query_intent,
        "include_cobranded": include_cobranded
    })
    
    return result["output"]