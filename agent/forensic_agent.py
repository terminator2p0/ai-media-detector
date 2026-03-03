import os
from dotenv import load_dotenv

# Standard LangChain v1.0 Imports
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# Import your orchestrator
from models.inference_orchestrator import MediaForensicsOrchestrator

load_dotenv()
orchestrator = MediaForensicsOrchestrator()

# --- 1. TOOLS ---
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

# --- 2. THE BRAIN (Gemini 1.5 Pro) ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    convert_system_message_to_human=True 
)

# --- 3. CREATE THE AGENT (v1.0 Syntax) ---
# In 2026, 'system_prompt' is the standard keyword
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a Senior Forensic AI Investigator. Use your tools to analyze media. "
        "Summarize results professionally with confidence scores."
    )
)

# --- 4. RUN ---
if __name__ == "__main__":
    query = (
        "I found this text in a suspect email: 'Leverage our synergistic deep-learning paradigms.' "
        "Also, check the voice memo at 'data/raw/audio/test_audio.wav'."
    )
    
    # LangChain v1.0 uses the standardized input dict
    # The 'create_agent' runtime handles the loop and formatting automatically
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    # The result is returned in a standard content block
    print("\n--- 📜 FINAL FORENSIC REPORT ---")
    print(response["messages"][-1].content)