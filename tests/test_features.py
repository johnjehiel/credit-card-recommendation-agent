import os
from dotenv import load_dotenv
from src.chains.intent_classification import (
    classify_query_intent, find_fd_intent, find_specific_card, 
    generate_card_specific_response
)
from src.chains.retrieval_chain import (
    enhance_query, generate_multi_queries, retrieve_ranked_cards
)

# Load environment variables
load_dotenv()

def test_intent_classification():
    print("\n=== Testing Intent Classification ===")
    
    # Test query intent classification
    print("\nTesting query intent classification:")
    test_queries = [
        "What are the best credit cards for travel?",
        "Tell me about the SBI Card ELITE",
        "How do I improve my credit score?"
    ]
    
    for query in test_queries:
        result = classify_query_intent(query)
        print(f"Query: {query}")
        print(f"Intent: {result.get('intent', 'unknown')}")
        print(f"Confidence: {result.get('confidence', 0)}")
        if result.get('intent') == 'specific':
            print(f"Card Name: {result.get('card_name', 'unknown')}")
        if result.get('intent') == 'no_retrieval':
            print(f"Response: {result.get('response', 'none')}")
        print("---")
    
    # Test FD card intent detection
    print("\nTesting FD card intent detection:")
    fd_queries = [
        "I'm a college student looking for my first credit card",
        "Best travel rewards credit cards with airport lounge access"
    ]
    
    for query in fd_queries:
        is_fd = find_fd_intent(query)
        print(f"Query: {query}")
        print(f"Is FD card intent: {is_fd}")
        print("---")
    
    # Test specific card detection
    print("\nTesting specific card detection:")
    card_queries = [
        "Tell me about the Cashback SBI Card",
        "Is the BPCL SBI Card Octane good for fuel benefits?"
    ]
    
    for query in card_queries:
        card = find_specific_card(query)
        print(f"Query: {query}")
        if card:
            print(f"Detected card: {card['name']}")
            response = generate_card_specific_response(query, card)
            print(f"Response: {response[:100]}...")
        else:
            print("No specific card detected")
        print("---")

def test_enhanced_retrieval():
    print("\n=== Testing Enhanced Retrieval ===")
    
    # Test query enhancement
    print("\nTesting query enhancement:")
    original_query = "I travel a lot and want something with good rewards"
    enhanced = enhance_query(original_query)
    print(f"Original: {original_query}")
    print(f"Enhanced: {enhanced}")
    
    # Test multi-query generation
    print("\nTesting multi-query generation:")
    queries = generate_multi_queries(enhanced)
    print("Generated queries:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
    
    # Test retrieval and ranking
    print("\nTesting retrieval and ranking:")
    cards = retrieve_ranked_cards(original_query, top_n=3)
    print(f"Retrieved {len(cards)} cards:")
    for card in cards:
        print(f"- {card['name']}")

if __name__ == "__main__":
    print("Running tests for the credit card recommender...")
    test_intent_classification()
    test_enhanced_retrieval()
    print("\nAll tests completed.")
