import streamlit as st
import os
import tempfile
from models.inference_orchestrator import MediaForensicsOrchestrator

# Force Page Config for the "Cybersecurity" Look
st.set_page_config(page_title="Forensic AI Detector", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_orchestrator():
    return MediaForensicsOrchestrator()

orchestrator = load_orchestrator()

st.title("🛡️ Multi-Modal Forensic AI Investigator")
st.markdown("---")

# --- 1. UPLOAD SECTION ---
st.sidebar.header("📁 Upload Suspect Media")
uploaded_file = st.sidebar.file_uploader(
    "Upload Image, Video, or Audio", 
    type=['mp4', 'mov', 'jpg', 'jpeg', 'png', 'wav', 'mp3']
)

# --- 2. MAIN ANALYSIS LOGIC ---
if uploaded_file is not None:
    # Save uploaded file to a temporary location so the Orchestrator can read the path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Media Preview")
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'video':
            st.video(uploaded_file)
        elif file_type == 'image':
            st.image(uploaded_file, use_container_width=True)
        elif file_type == 'audio':
            st.audio(uploaded_file)

    with col2:
        st.subheader("Forensic Analysis")
        if st.button("🚀 Run Neural Scan"):
            with st.spinner("Analyzing artifacts on MPS GPU..."):
                
                result = None
                if file_type == 'video':
                    result = orchestrator.scan_video(tmp_path)
                elif file_type == 'image':
                    result = orchestrator.scan_image(tmp_path)
                elif file_type == 'audio':
                    result = orchestrator.scan_audio(tmp_path)

                if result:
                    # Styling the output
                    pred = result.get("prediction", "Unknown")
                    conf = result.get("average_confidence", result.get("confidence", 0))
                    color = "red" if "AI" in pred or "Deepfake" in pred else "green"
                    
                    st.markdown(f"### Verdict: :{color}[{pred}]")
                    st.metric("Confidence Score", f"{conf}%")
                    st.progress(conf / 100)
                    
                    if "frames_scanned" in result:
                        st.caption(f"Spatial Analysis: {result['frames_scanned']} frames processed.")
                
    # Cleanup temp file after analysis
    os.remove(tmp_path)

else:
    st.info("Please upload a media file in the sidebar to begin the forensic investigation.")

# --- 3. TEXT SCANNER (Independent) ---
st.markdown("---")
st.subheader("📝 Textual Analysis (LLM Detection)")
user_text = st.text_area("Paste text (Email, Transcript, Post):", placeholder="Enter text here...")
if st.button("Scan Text"):
    if user_text.strip():
        text_result = orchestrator.scan_text(user_text)
        st.write("### Analysis Results")
        st.json(text_result)
    else:
        st.warning("Please enter some text first.")