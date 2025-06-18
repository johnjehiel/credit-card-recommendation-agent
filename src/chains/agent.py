import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any

from src.tools.neo4j_tool import Neo4jRetrievalTool
from src.chains.retrieval_chain import retrieve_ranked_cards
from src.chains.intent_classification import (
    classify_query_intent, 
    find_fd_intent, 
    find_specific_card, 
    generate_card_specific_response
)

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
def run_agent_pipeline(query: str, query_intent_manual: bool = False, include_cobranded: bool = True,
                      use_eligibility: bool = False, min_income: float = 5.0, min_cibil: int = 750, 
                      age: int = 25, min_joining_fee: int = 0, max_joining_fee: int = 100000,
                      min_annual_fee: int = 0, max_annual_fee: int = 100000):
    """
    Run the agent pipeline to get credit card recommendations.
    
    Parameters:
    - query: User's question about credit cards
    - query_intent_manual: Manual override for FD card intent (beginners/students)
    - include_cobranded: Whether to include co-branded cards
    - use_eligibility: Whether to apply eligibility filtering
    - min_income: Minimum income (LPA)
    - min_cibil: Minimum CIBIL score
    - age: User's age
    - min_joining_fee/max_joining_fee: Range for joining fee
    - min_annual_fee/max_annual_fee: Range for annual fee
    """
    # Step 1: Auto-detect query intent (specific card, general retrieve, or no-retrieval)
    intent_result = classify_query_intent(query)
    intent_type = intent_result.get("intent", "retrieve")
    print(f"Detected intent: {intent_type}")
    
    # Step 2: Handle specific card requests
    if intent_type == "specific":
        # Find the specific card mentioned
        card_name = intent_result.get("card_name")
        card = None
        
        if card_name:
            card = find_specific_card(query)
        
        if card:
            # Generate specific response about this card
            response = generate_card_specific_response(query, card)
            return response
    
    # Step 3: Handle no-retrieval (general financial questions)
    if intent_type == "no_retrieval":
        response = intent_result.get("response")
        if response:
            return response
    
    # Step 4: For retrieval intents, prepare tools and detect FD card intent
    # Auto-detect if the query is about FD cards (for beginners/students)
    query_intent = query_intent_manual or find_fd_intent(query)
    print(f"FD Card intent: {query_intent}")
    
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
      # Process combined input with eligibility info if applicable
    eligibility_info = ""
    if use_eligibility:
        eligibility_info = f"""
        Eligibility filters applied:
        - Minimum Income: {min_income} LPA
        - Minimum CIBIL Score: {min_cibil}
        - Age: {age}
        - Joining Fee Range: ₹{min_joining_fee} - ₹{max_joining_fee}
        - Annual Fee Range: ₹{min_annual_fee} - ₹{max_annual_fee}
        """
    
    intent_info = ""
    if query_intent:
        intent_info = "\nThis query is for a first-time credit card user or student with limited credit history."
    
    cobranded_info = ""
    if not include_cobranded:
        cobranded_info = "\nExclude co-branded cards (airline, retail, etc.)"
    
    combined_input = f"User query: {query}{intent_info}{cobranded_info}\n{eligibility_info}"
    
    result = agent_executor.invoke({
        "input": combined_input,  # This is the key that memory will use
        "query_intent": query_intent,
        "include_cobranded": include_cobranded,
        "use_eligibility": use_eligibility,
        "min_income": min_income,
        "min_cibil": min_cibil,
        "age": age,
        "min_joining_fee": min_joining_fee,
        "max_joining_fee": max_joining_fee,
        "min_annual_fee": min_annual_fee,
        "max_annual_fee": max_annual_fee
    })
    
    return result["output"]