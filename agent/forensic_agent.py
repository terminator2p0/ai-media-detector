import os
from dotenv import load_dotenv
import streamlit as st

# Standard LangChain v1.0 Imports
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# Import your orchestrator
from models.inference_orchestrator import MediaForensicsOrchestrator

load_dotenv()
# Initialize the engine once
orchestrator = MediaForensicsOrchestrator()

# --- 1. SECURE KEY RETRIEVAL ---
# Check Streamlit secrets first (Cloud), then fall back to os.environ (Local)
if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
else:
    google_api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. TOOLS ---
@tool
def analyze_image(file_path: str):
    """Analyzes an image file (JPG/PNG) for deepfakes."""
    return orchestrator.scan_image(file_path)

@tool
def analyze_text(content: str):
    """Analyzes text for AI generation/LLM markers."""
    return orchestrator.scan_text(content)

@tool
def analyze_audio(file_path: str):
    """Analyzes audio for voice cloning/synthetic speech."""
    return orchestrator.scan_audio(file_path)

@tool
def analyze_video(file_path: str):
    """Analyzes an MP4/MOV video file. It samples frames to detect facial manipulation or AI generation."""
    return orchestrator.scan_video(file_path)

tools = [analyze_image, analyze_text, analyze_audio, analyze_video]

# --- 3. THE BRAIN (Gemini 2.5 Flash) ---
# Pass the retrieved google_api_key directly to the model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=google_api_key,
    temperature=0,
    convert_system_message_to_human=True 
)

# --- 4. CREATE THE AGENT (v1.0 Syntax) ---
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a Senior Forensic AI Investigator. Use your tools to analyze media. "
        "Summarize results professionally with confidence scores."
    )
)

# --- 5. RUN ---
if __name__ == "__main__":
    query = (
        "I found this text in a suspect email: 'Leverage our synergistic deep-learning paradigms.' "
        "Also, check the voice memo at 'data/raw/audio/test_audio.wav'."
    )
    
    # LangChain v1.0 standardized input dict
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    print("\n--- 📜 FINAL FORENSIC REPORT ---")
    print(response["messages"][-1].content)