import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime

import streamlit as st

import db
from agent.forensic_agent import generate_forensic_report, analyze_multimodal
from models.inference_orchestrator import MediaForensicsOrchestrator


st.set_page_config(
    page_title="Forensic AI Detector",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_orchestrator():
    db.init_db()
    return MediaForensicsOrchestrator()


orchestrator = load_orchestrator()

FEEDBACK_DIR = "data/feedback_loop"
os.makedirs(os.path.join(FEEDBACK_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "fake"), exist_ok=True)

MODEL_VERSION = os.environ.get("MODEL_VERSION", "efficientnet_b4_v1")

for key, default in [
    ("last_result", None),
    ("last_file_path", None),
    ("last_file_name", None),
    ("last_file_type", None),
    ("last_file_hash", None),
    ("last_prediction_id", None),
    ("feedback_recorded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def persist_media(src_path: str, true_label: str, file_hash: str, suffix: str) -> str:
    """Copy media into the labeled feedback directory; return the destination path.

    Skips the copy when an identical hash already lives in that directory.
    """
    dest_dir = os.path.join(FEEDBACK_DIR, true_label)
    os.makedirs(dest_dir, exist_ok=True)
    existing = next((f for f in os.listdir(dest_dir) if f.startswith(file_hash) and not f.endswith(".json")), None)
    if existing:
        return os.path.join(dest_dir, existing)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = os.path.join(dest_dir, f"{file_hash}_{timestamp}{suffix}")
    shutil.copy(src_path, dest_path)
    with open(dest_path + ".json", "w") as fp:
        json.dump(
            {
                "hash": file_hash,
                "true_label": true_label,
                "timestamp": timestamp,
            },
            fp,
        )
    return dest_path


def reset_session():
    if st.session_state.last_file_path and os.path.exists(st.session_state.last_file_path):
        try:
            os.remove(st.session_state.last_file_path)
        except OSError:
            pass
    for key in ("last_result", "last_file_path", "last_file_name", "last_file_type",
                "last_file_hash", "last_prediction_id", "feedback_recorded"):
        st.session_state[key] = None if key != "feedback_recorded" else False


st.title("🛡️ Multi-Modal Forensic AI Investigator")
st.markdown("---")

st.sidebar.header("📁 Upload Suspect Media")
uploaded_file = st.sidebar.file_uploader(
    "Upload Image, Video, or Audio",
    type=["mp4", "mov", "jpg", "jpeg", "png", "wav", "mp3"],
)

if st.sidebar.button("♻️ Clear Session"):
    reset_session()
    st.rerun()

with st.sidebar.expander("📊 Dataset Stats", expanded=False):
    s = db.stats()
    st.metric("Total scans logged", s["total_predictions"])
    st.metric("Audited samples", s["total_feedback"])
    if s["audited_accuracy"] is not None:
        st.metric("Audited accuracy", f"{s['audited_accuracy']}%")
    if s["predictions_by_type"]:
        st.caption("Scans by media type")
        st.json(s["predictions_by_type"])
    if s["feedback_by_label"]:
        st.caption("Confirmed labels")
        st.json(s["feedback_by_label"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]

    if st.session_state.last_file_name != uploaded_file.name:
        reset_session()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            st.session_state.last_file_path = tmp.name
        st.session_state.last_file_name = uploaded_file.name
        st.session_state.last_file_type = uploaded_file.type.split("/")[0]
        st.session_state.last_file_hash = file_md5(st.session_state.last_file_path)

    col1, col2 = st.columns([1, 1])
    file_type = st.session_state.last_file_type

    with col1:
        st.subheader("Media Preview")
        if file_type == "video":
            st.video(uploaded_file)
        elif file_type == "image":
            st.image(uploaded_file, width="stretch")
        elif file_type == "audio":
            st.audio(uploaded_file)

    with col2:
        st.subheader("Forensic Analysis")
        if st.button("🚀 Run Neural Scan"):
            with st.spinner(f"Analyzing artifacts on {orchestrator.device}..."):
                path = st.session_state.last_file_path
                if file_type == "video":
                    result = orchestrator.scan_video(path)
                elif file_type == "image":
                    result = orchestrator.scan_image(path)
                elif file_type == "audio":
                    result = orchestrator.scan_audio(path)
                else:
                    result = {"error": f"Unsupported file type: {file_type}"}

                st.session_state.last_result = result
                st.session_state.feedback_recorded = False

                if isinstance(result, dict) and "error" not in result:
                    pred = result.get("prediction", "Unknown")
                    conf = result.get("average_confidence", result.get("confidence"))
                    st.session_state.last_prediction_id = db.log_prediction(
                        file_type=file_type,
                        model_prediction=pred,
                        confidence=conf,
                        raw_result=result,
                        file_hash=st.session_state.last_file_hash,
                        file_name=st.session_state.last_file_name,
                        model_version=MODEL_VERSION,
                    )

        res = st.session_state.last_result
        if isinstance(res, str):
            st.error(res)
        elif isinstance(res, dict) and "error" in res:
            st.error(res["error"])
        elif isinstance(res, dict):
            pred = res.get("prediction", "Unknown")
            conf = res.get("average_confidence", res.get("confidence", 0))
            color = "red" if ("AI" in pred or "Deepfake" in pred) else "green"

            st.markdown(f"### Verdict: :{color}[{pred}]")
            st.metric("Confidence Score", f"{conf}%")
            st.progress(min(max(conf / 100, 0.0), 1.0))

            # --- Gemini forensic report + multi-modal ---
            report_col1, report_col2 = st.columns(2)
            with report_col1:
                if st.button("🔬 Generate Forensic Report"):
                    with st.spinner("Gemini is analyzing the evidence..."):
                        report = generate_forensic_report(
                            scan_result=res,
                            media_type=file_type,
                            file_name=st.session_state.last_file_name,
                        )
                        st.session_state["forensic_report"] = report

            with report_col2:
                if file_type == "video" and st.button("🎬 Multi-Modal Scan (Video + Audio)"):
                    with st.spinner("Cross-referencing visual and audio tracks..."):
                        mm_result = analyze_multimodal(st.session_state.last_file_path, orchestrator)
                        st.session_state["multimodal_result"] = mm_result

            if st.session_state.get("forensic_report"):
                with st.expander("📜 Forensic Report (Gemini)", expanded=True):
                    st.markdown(st.session_state["forensic_report"])

            if st.session_state.get("multimodal_result"):
                with st.expander("🎬 Multi-Modal Analysis", expanded=True):
                    mm = st.session_state["multimodal_result"]
                    if "combined_verdict" in mm:
                        v_color = "red" if "AI" in mm["combined_verdict"] or "Deepfake" in mm["combined_verdict"] else "green"
                        st.markdown(f"**Combined Verdict:** :{v_color}[{mm['combined_verdict']}]")
                        st.metric("Combined Confidence", f"{mm['combined_confidence']}%")
                    st.json(mm)

            st.write("---")
            st.write("### 🤖 Self-Supervision: Was this correct?")

            if st.session_state.feedback_recorded:
                st.success("Feedback recorded for this scan.")
            else:
                f_col1, f_col2 = st.columns(2)

                model_says_fake = "AI" in pred or "Deepfake" in pred

                if f_col1.button("✅ Yes, Correct"):
                    true_label = "fake" if model_says_fake else "real"
                    stored = persist_media(
                        st.session_state.last_file_path,
                        true_label,
                        st.session_state.last_file_hash,
                        suffix,
                    )
                    db.log_feedback(
                        prediction_id=st.session_state.last_prediction_id,
                        true_label=true_label,
                        was_correct=True,
                        file_hash=st.session_state.last_file_hash,
                        stored_media_path=stored,
                    )
                    st.session_state.feedback_recorded = True
                    st.success(f"Validated. Saved to {true_label}/ for future training.")
                    st.rerun()

                if f_col2.button("❌ No, Incorrect"):
                    true_label = "real" if model_says_fake else "fake"
                    stored = persist_media(
                        st.session_state.last_file_path,
                        true_label,
                        st.session_state.last_file_hash,
                        suffix,
                    )
                    db.log_feedback(
                        prediction_id=st.session_state.last_prediction_id,
                        true_label=true_label,
                        was_correct=False,
                        file_hash=st.session_state.last_file_hash,
                        stored_media_path=stored,
                    )
                    st.session_state.feedback_recorded = True
                    st.warning(f"Logged correction. True label: {true_label.upper()}.")
                    st.rerun()
else:
    st.info("Please upload a media file in the sidebar to begin.")

st.markdown("---")
st.subheader("📝 Textual Analysis (LLM Detection)")
user_text = st.text_area("Paste text (Email, Transcript, Post):", height=150)

if "last_text_result" not in st.session_state:
    st.session_state.last_text_result = None
    st.session_state.last_text_prediction_id = None
    st.session_state.last_text_hash = None
    st.session_state.text_feedback_recorded = False

if st.button("Scan Text"):
    if user_text.strip():
        text_result = orchestrator.scan_text(user_text)
        st.session_state.last_text_result = text_result
        st.session_state.text_feedback_recorded = False
        if isinstance(text_result, dict) and "error" not in text_result:
            text_hash = hashlib.md5(user_text.strip().encode("utf-8")).hexdigest()
            st.session_state.last_text_hash = text_hash
            st.session_state.last_text_prediction_id = db.log_prediction(
                file_type="text",
                model_prediction=text_result.get("prediction", "Unknown"),
                confidence=text_result.get("ai_probability"),
                raw_result=text_result,
                file_hash=text_hash,
                file_name=None,
                model_version="tmr-roberta",
            )
    else:
        st.warning("Please enter some text first.")

if st.session_state.last_text_result:
    st.write("### Analysis Results")
    st.json(st.session_state.last_text_result)

    if isinstance(st.session_state.last_text_result, dict) and "error" not in st.session_state.last_text_result:
        if st.button("🔬 Text Forensic Report", key="text_report"):
            with st.spinner("Gemini is analyzing the text..."):
                report = generate_forensic_report(
                    scan_result=st.session_state.last_text_result,
                    media_type="text",
                    extra_context=f"First 200 chars of input: {user_text[:200]}",
                )
                st.session_state["text_forensic_report"] = report

        if st.session_state.get("text_forensic_report"):
            with st.expander("📜 Text Forensic Report (Gemini)", expanded=True):
                st.markdown(st.session_state["text_forensic_report"])
        if st.session_state.text_feedback_recorded:
            st.success("Feedback recorded.")
        else:
            t_col1, t_col2 = st.columns(2)
            pred = st.session_state.last_text_result.get("prediction", "")
            model_says_ai = "AI" in pred
            if t_col1.button("✅ Correct", key="text_correct"):
                true_label = "ai" if model_says_ai else "human"
                db.log_feedback(
                    prediction_id=st.session_state.last_text_prediction_id,
                    true_label=true_label,
                    was_correct=True,
                    file_hash=st.session_state.last_text_hash,
                )
                st.session_state.text_feedback_recorded = True
                st.rerun()
            if t_col2.button("❌ Incorrect", key="text_incorrect"):
                true_label = "human" if model_says_ai else "ai"
                db.log_feedback(
                    prediction_id=st.session_state.last_text_prediction_id,
                    true_label=true_label,
                    was_correct=False,
                    file_hash=st.session_state.last_text_hash,
                )
                st.session_state.text_feedback_recorded = True
                st.rerun()

with st.expander("🕓 Recent Scans", expanded=False):
    rows = db.recent_predictions(limit=15)
    if not rows:
        st.caption("No scans logged yet.")
    else:
        st.dataframe(rows, width="stretch")
