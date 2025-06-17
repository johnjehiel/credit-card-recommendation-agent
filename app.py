from dotenv import load_dotenv
from src.ui.gradio_app import demo

if __name__ == "__main__":
    load_dotenv()  # Load .env credentials
    demo.launch(share=True)