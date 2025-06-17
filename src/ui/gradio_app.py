import gradio as gr
from src.chains.agent import run_agent_pipeline
from dotenv import load_dotenv

load_dotenv()

chat_history = []

with gr.Blocks(title="Agentic Credit Card Recommender") as demo:
    gr.Markdown("""
    # 🧠 Credit Card Recommender
    Personalized credit card suggestions using Neo4j Knowledge Graph and Gemini-powered LLM Agent
    """)

    with gr.Row():
        query_input = gr.Textbox(label="What kind of card are you looking for?", placeholder="e.g. Best cards for online shopping with fuel benefits")
        fd_checkbox = gr.Checkbox(label="Are you a beginner / student (FD card)?", value=False)
        cobrand_checkbox = gr.Checkbox(label="Include Co-branded Cards", value=True)

    run_button = gr.Button("🔍 Get Recommendations")
    output = gr.Textbox(lines=15, label="Recommendations", interactive=False)

    with gr.Accordion("Ask Follow-up Questions", open=False):
        chatbox = gr.Chatbot(type="messages")  # Changed to use messages format
        followup_input = gr.Textbox(label="Your follow-up question")
        followup_submit = gr.Button("Submit")

    # Wrapper logic
    def recommend(query, fd_intent, include_cobranded):
        global chat_history
        chat_history = []  # Clear chat history
        result = run_agent_pipeline(query, fd_intent, include_cobranded)
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": result})
        return result, chat_history

    def continue_chat(message):
        global chat_history
        context = "\n".join([f"User: {m['content']}" if m['role'] == 'user' else f"Bot: {m['content']}" for m in chat_history])
        full_message = f"{context}\nUser: {message}\nBot:"
        from src.chains.agent import gemini_llm
        response = gemini_llm.invoke(full_message)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response.content})
        return chat_history

    run_button.click(
        fn=recommend,
        inputs=[query_input, fd_checkbox, cobrand_checkbox],
        outputs=[output, chatbox]
    )

    followup_submit.click(
        fn=continue_chat,
        inputs=[followup_input],
        outputs=[chatbox]
    )

    followup_input.submit(
        fn=continue_chat,
        inputs=[followup_input],
        outputs=[chatbox]
    )

if __name__ == "__main__":
    demo.launch(share=True)