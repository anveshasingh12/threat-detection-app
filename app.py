import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import time
from streamlit_lottie import st_lottie
import requests
import base64

# Constants
MODEL_PATH = "best.pt"
ALARM_SOUND_PATH ="alarm.mp3"
LOCAL_TEST_IMAGE = r"C:\Users\anvesha singh\Documents\SIXray\test\images\009008_jpg.rf.8f9d287571d5d48f46a87116a4a82d56"
HIGH_RISK_ITEMS = {"gun", "knife", "wrench", "pliers", "scissors"}

# Load model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_scan = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_t9gkkhz4.json")

# Play alarm with base64 as fallback
def play_alarm_base64(audio_file_path):
    try:
        with open(audio_file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            st.markdown(f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("🔇 Alarm sound file not found.")
    except Exception as e:
        st.warning(f"🔇 Alarm error: {e}")

# UI Configuration
st.set_page_config(page_title="AI Security Scanner", layout="wide")

st.markdown("""
<style>
body {
    background: #f3f4f7;
    font-family: 'Segoe UI', sans-serif;
}
.main-header {
    text-align: center;
    padding: 1rem 0;
}
.main-title {
    font-size: 3rem;
    font-weight: bold;
    color: #1f3b4d;
}
.main-subtitle {
    font-size: 1.2rem;
    color: #555;
}
.status-indicator {
    height: 12px;
    width: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
}
.status-online {
    background-color: #00cc66;
}
.detection-item {
    background-color: #fff;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.high-risk {
    border-left: 6px solid #ff4b4b;
}
.safe {
    border-left: 6px solid #4caf50;
}
.alert-safe {
    padding: 1rem;
    background: #eafbe7;
    color: #1c6f26;
    border-radius: 8px;
    font-weight: bold;
}
.alert-high-risk {
    padding: 1rem;
    background: #ffdddd;
    color: #b30000;
    border-radius: 8px;
    font-weight: bold;
    font-size: 1.2rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ AI SECURITY SCANNER</div>
    <div class="main-subtitle">Advanced Threat Detection System</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔍 System Status")
    st.markdown('<span class="status-indicator status-online"></span>**AI Model**: Online', unsafe_allow_html=True)
    st.markdown('<span class="status-indicator status-online"></span>**Scanner**: Ready', unsafe_allow_html=True)
    st.markdown("### ⚠️ Monitored Threats")
    for threat in HIGH_RISK_ITEMS:
        st.markdown(f"🔹 **{threat.title()}**")

# Lottie Animation
st_lottie(lottie_scan, height=200)

# Image Upload Section
st.markdown("### 📤 Upload X-Ray Image or Use Test Image")
option = st.radio("Choose input method:", ["📁 Upload Image", "🖼️ Use Local Test Image"])
uploaded_file = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png", "bmp"])
elif option == "🖼️ Use Local Test Image":
    if os.path.exists(LOCAL_TEST_IMAGE):
        uploaded_file = open(LOCAL_TEST_IMAGE, "rb")
    else:
        st.error("❌ Local test image not found.")

# Detection Block
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, use_column_width=True)
    with col2:
        st.markdown("### 📋 Image Info")
        st.info(f"""
        **Format:** {image.format or 'N/A'}
        **Size:** {image.size[0]} × {image.size[1]} px
        **Mode:** {image.mode}
        """)

    if st.button("🔍 Begin Security Scan"):
        st.markdown("""<div style="text-align:center;"><h4>🔎 Scanning in progress...</h4></div>""", unsafe_allow_html=True)
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1)

        results = model(img_array, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            st.markdown("""<div class="alert-safe">✅ No threats detected!</div>""", unsafe_allow_html=True)
        else:
            detections = []
            annotated_img = results[0].plot()
            high_risk = False

            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = model.names.get(cls_id, f"ID {cls_id}")
                conf = box.conf[0].item()
                detections.append((class_name, conf))
                if class_name in HIGH_RISK_ITEMS:
                    high_risk = True

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(annotated_img, caption="📌 Detection Result", use_column_width=True)
                save_path = "annotated_result.jpg"
                cv2.imwrite(save_path, annotated_img)
                st.success(f"📸 Annotated image saved as `{save_path}`")

            with col2:
                st.markdown("### 🎯 Detected Items")
                for name, conf in detections:
                    icon = "🚨" if name in HIGH_RISK_ITEMS else "✅"
                    style = "high-risk" if name in HIGH_RISK_ITEMS else "safe"
                    st.markdown(f"""
                    <div class="detection-item {style}">
                        <strong>{icon} {name.title()}</strong><br>
                        <small>Confidence: {conf:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)

            if high_risk:
                st.markdown("""<div class="alert-high-risk">🚨 SECURITY ALERT: HIGH-RISK ITEM DETECTED!</div>""", unsafe_allow_html=True)
                try:
                    st.audio(ALARM_SOUND_PATH, autoplay=True)
                except:
                    play_alarm_base64(ALARM_SOUND_PATH)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; padding: 2rem 0;">
    🛡️ <strong>AI Security Scanner</strong> — Powered by YOLOv8<br>
    <small>Built with ❤️ and Streamlit</small>
</div>
""", unsafe_allow_html=True)


