import streamlit as st
import os
import tempfile
import json
import shutil
import hashlib
from datetime import datetime
from models.inference_orchestrator import MediaForensicsOrchestrator


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Forensic AI Detector", 
    page_icon="🛡️", 
    layout="wide"
)

# --- INITIALIZATION ---
@st.cache_resource
def load_orchestrator():
    return MediaForensicsOrchestrator()

orchestrator = load_orchestrator()

# --- CONSTANTS ---
FEEDBACK_DIR = "data/feedback_loop"
os.makedirs(os.path.join(FEEDBACK_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "fake"), exist_ok=True)

# Initialize session state variables
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_file_path' not in st.session_state:
    st.session_state.last_file_path = None
if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None

# --- HELPER FUNCTIONS ---
def get_file_hash(file_path):
    """Generates a unique MD5 hash for deduplication."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# --- UI HEADER ---
st.title("🛡️ Multi-Modal Forensic AI Investigator")
st.markdown("---")

# --- SIDEBAR: UPLOAD ---
st.sidebar.header("📁 Upload Suspect Media")
uploaded_file = st.sidebar.file_uploader(
    "Upload Image, Video, or Audio", 
    type=['mp4', 'mov', 'jpg', 'jpeg', 'png', 'wav', 'mp3']
)

if st.sidebar.button("♻️ Clear Session"):
    st.session_state.last_result = None
    st.session_state.last_file_name = None
    st.rerun()

# --- MAIN ANALYSIS LOGIC ---
if uploaded_file is not None:
    t_suffix = os.path.splitext(uploaded_file.name)[1]
    
    # Save file only if it's a new upload
    if st.session_state.last_file_name != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=t_suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.last_file_path = tmp_file.name
            st.session_state.last_file_name = uploaded_file.name
            st.session_state.last_result = None 

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Media Preview")
        file_type = uploaded_file.type.split('/')[0]
        if file_type == 'video':
            st.video(uploaded_file)
        elif file_type == 'image':
            st.image(uploaded_file, width="stretch")
        elif file_type == 'audio':
            st.audio(uploaded_file)

    with col2:
        st.subheader("Forensic Analysis")
        if st.button("🚀 Run Neural Scan"):
            with st.spinner("Analyzing artifacts on MPS GPU..."):
                result = None
                path = st.session_state.last_file_path
                
                if file_type == 'video':
                    result = orchestrator.scan_video(path)
                elif file_type == 'image':
                    result = orchestrator.scan_image(path)
                elif file_type == 'audio':
                    result = orchestrator.scan_audio(path)
                
                st.session_state.last_result = result

        # Display Results and Feedback UI
        if st.session_state.last_result:
            res = st.session_state.last_result
            pred = res.get("prediction", "Unknown")
            conf = res.get("average_confidence", res.get("confidence", 0))
            color = "red" if ("AI" in pred or "Deepfake" in pred) else "green"
            
            st.markdown(f"### Verdict: :{color}[{pred}]")
            st.metric("Confidence Score", f"{conf}%")
            st.progress(conf / 100)

            # --- DEDUPLICATED FEEDBACK LOOP ---
            st.write("---")
            st.write("### 🤖 Self-Supervision: Was this correct?")
            f_col1, f_col2 = st.columns(2)
            
            if f_col1.button("✅ Yes, Correct"):
                st.success("Great! Data point validated.")
                st.session_state.last_result = None
            
            if f_col2.button("❌ No, Incorrect"):
                # 1. Determine the "True" Label
                target_label = "real" if ("AI" in pred or "Deepfake" in pred) else "fake"
                
                # 2. Check for Duplicates via Hashing
                file_hash = get_file_hash(st.session_state.last_file_path)
                dest_dir = os.path.join(FEEDBACK_DIR, target_label)
                
                # Scan directory for existing hash
                is_duplicate = any(f.startswith(file_hash) for f in os.listdir(dest_dir))
                
                if is_duplicate:
                    st.info("System Note: This artifact is already in the feedback loop. Skipping duplicate.")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"{file_hash}_{timestamp}{t_suffix}"
                    dest_path = os.path.join(dest_dir, new_filename)
                    
                    if os.path.exists(st.session_state.last_file_path):
                        shutil.copy(st.session_state.last_file_path, dest_path)
                        
                        metadata = {
                            "filename": st.session_state.last_file_name,
                            "hash": file_hash,
                            "model_prediction": pred,
                            "true_label": target_label,
                            "timestamp": timestamp
                        }
                        with open(f"{dest_path}.json", "w") as f:
                            json.dump(metadata, f)
                        
                        st.warning(f"Logged as False {pred}. Target: {target_label.upper()}")
                
                st.session_state.last_result = None

else:
    st.info("Please upload a media file in the sidebar to begin.")

# --- 3. TEXT SCANNER ---
st.markdown("---")
st.subheader("📝 Textual Analysis (LLM Detection)")
user_text = st.text_area("Paste text (Email, Transcript, Post):", height=150)
if st.button("Scan Text"):
    if user_text.strip():
        text_result = orchestrator.scan_text(user_text)
        st.write("### Analysis Results")
        st.json(text_result)
    else:
        st.warning("Please enter some text first.")