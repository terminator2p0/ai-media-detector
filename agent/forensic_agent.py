"""Forensic AI Agent — powered by Gemini 2.5 Flash.

Two modes of use:
  1. Streamlit integration — call generate_forensic_report() from app.py.
     Gemini receives the scan results + media metadata and produces a
     professional forensic analysis report explaining WHY something was flagged.

  2. Standalone agent — run this file directly.  A LangChain tool-calling agent
     can autonomously decide which scan tools to invoke and chain multi-modal
     analyses together (e.g. video frames + audio from the same file).

Both paths fall back gracefully if the GOOGLE_API_KEY is missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _get_api_key() -> str | None:
    key = os.getenv("GOOGLE_API_KEY")
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return key


# ---------------------------------------------------------------------------
# 1. Forensic report generator (used by Streamlit UI)
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """\
You are a Senior Forensic AI Investigator. You receive the results of automated
neural scans on suspect media (image, video, audio, or text) and produce a
professional forensic analysis report.

Your report must include:
1. **Executive Summary** — one-paragraph verdict with confidence assessment.
2. **Technical Analysis** — explain WHAT artifacts or patterns led to the verdict.
   For images/video: mention frequency-domain anomalies, GAN fingerprints,
   inconsistent lighting/shadows, blending seams, temporal coherence issues.
   For audio: mention spectral artifacts, unnatural prosody, voice cloning markers.
   For text: mention statistical patterns, perplexity signatures, repetitive structure.
3. **Confidence Calibration** — how reliable is this specific score? What factors
   could raise or lower confidence (e.g., compression, short sample, domain mismatch)?
4. **Investigative Recommendations** — next steps for a human analyst.

Be precise and professional. Cite the model's confidence score and frame your
analysis around it. If confidence is low (<60%), emphasize uncertainty.
Do NOT fabricate technical details — if you cannot infer a specific artifact type,
say so and recommend manual review.
"""


def generate_forensic_report(
    scan_result: dict,
    media_type: str,
    file_name: str | None = None,
    extra_context: str = "",
) -> str:
    """Call Gemini to produce a forensic analysis report from scan results.

    Returns the report text, or a fallback message if the API key is missing.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "_Forensic report unavailable — add GOOGLE_API_KEY to your Streamlit secrets "
            "or .env file to enable Gemini-powered analysis reports._"
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    # Build prompt as a single human message — Gemini handles system instructions
    # best when they are included in the human turn rather than as a SystemMessage.
    full_prompt = (
        f"{REPORT_SYSTEM_PROMPT}\n\n"
        f"---\n"
        f"Media type: {media_type}\n"
        f"File: {file_name or 'N/A'}\n"
        f"Scan results: {scan_result}\n"
    )
    if extra_context:
        full_prompt += f"Additional context: {extra_context}\n"
    full_prompt += "\nProduce your forensic analysis report now."

    # Try models in order from most capable to most available
    model_candidates = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    last_error = None
    for model_name in model_candidates:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
                convert_system_message_to_human=True,
            )
            response = llm.invoke([HumanMessage(content=full_prompt)])
            return response.content
        except Exception as exc:
            last_error = exc
            continue

    return (
        f"_Forensic report generation failed. "
        f"Check that your GOOGLE_API_KEY has access to Gemini models. "
        f"Error: {type(last_error).__name__}: {str(last_error)[:200]}_"
    )


# ---------------------------------------------------------------------------
# 2. Multi-modal analysis — analyze video's visual + audio tracks together
# ---------------------------------------------------------------------------

def analyze_multimodal(video_path: str, orchestrator) -> dict:
    """Run both visual and audio scans on a video file and return combined results."""
    import subprocess
    import tempfile

    visual_result = orchestrator.scan_video(video_path)

    audio_result = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", audio_path, "-y"],
            capture_output=True, timeout=60,
        )
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            audio_result = orchestrator.scan_audio(audio_path)
        os.unlink(audio_path)
    except Exception:
        pass

    combined = {
        "visual_analysis": visual_result,
        "audio_analysis": audio_result or "Audio extraction failed or no audio track",
    }

    if isinstance(visual_result, dict) and isinstance(audio_result, dict):
        v_conf = visual_result.get("average_confidence", visual_result.get("confidence", 50))
        a_conf = audio_result.get("confidence", 50)
        avg = (v_conf + a_conf) / 2
        v_fake = "AI" in visual_result.get("prediction", "") or "Deepfake" in visual_result.get("prediction", "")
        a_fake = "AI" in audio_result.get("prediction", "")

        if v_fake and a_fake:
            combined["combined_verdict"] = "AI-Generated/Deepfake"
            combined["combined_confidence"] = round(avg, 2)
        elif v_fake or a_fake:
            combined["combined_verdict"] = "Partially Manipulated (visual and audio disagree)"
            combined["combined_confidence"] = round(avg, 2)
        else:
            combined["combined_verdict"] = "Authentic"
            combined["combined_confidence"] = round(avg, 2)

    return combined


# ---------------------------------------------------------------------------
# 3. Standalone LangChain tool-calling agent
# ---------------------------------------------------------------------------

def build_agent():
    """Build a LangChain tool-calling agent with forensic scan tools."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import AgentExecutor, create_tool_calling_agent

    from models.inference_orchestrator import MediaForensicsOrchestrator

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set. Add it to .env or Streamlit secrets.")

    orchestrator = MediaForensicsOrchestrator()

    @tool
    def scan_image(file_path: str) -> dict:
        """Analyze an image file (JPG/PNG) for AI-generation artifacts."""
        return orchestrator.scan_image(file_path)

    @tool
    def scan_text(content: str) -> dict:
        """Analyze text for LLM-generation markers."""
        return orchestrator.scan_text(content)

    @tool
    def scan_audio(file_path: str) -> dict:
        """Analyze an audio file for synthetic speech / voice cloning."""
        return orchestrator.scan_audio(file_path)

    @tool
    def scan_video(file_path: str) -> dict:
        """Analyze a video by sampling frames for deepfake detection."""
        return orchestrator.scan_video(file_path)

    @tool
    def multimodal_video_scan(file_path: str) -> dict:
        """Analyze a video's visual frames AND audio track, then cross-reference."""
        return analyze_multimodal(file_path, orchestrator)

    tools = [scan_image, scan_text, scan_audio, scan_video, multimodal_video_scan]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", REPORT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)


if __name__ == "__main__":
    executor = build_agent()
    query = (
        "I found this text in a suspect email: 'Leverage our synergistic deep-learning paradigms "
        "to unlock unprecedented value across enterprise verticals.' "
        "Also analyze the image at 'data/feedback_loop/real/' if any exist."
    )
    response = executor.invoke({"input": query})
    print("\n--- FORENSIC REPORT ---")
    print(response["output"])
