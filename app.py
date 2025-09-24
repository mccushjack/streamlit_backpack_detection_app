# Streamlit Backpack Detection App using YOLOv8

import streamlit as st
from streamlit.components.v1 import html
import torch
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
import time

# Set page configuration with dark theme
st.set_page_config(
    page_title="Backpack Detection",
    page_icon="🎒",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Sidebar toggle (keep this):
with st.sidebar:
    mode = st.radio("Theme", ["Dark", "Light"], horizontal=True)

#    :root { --primary:#0C62FB; --radius:10px; }

def apply_theme(mode: str):
    common = """
    :root { --primary:#0C62FB; --radius:10px; }

    /* Generic buttons */
    .stButton > button {
      background: var(--primary) !important;
      color:#fff !important;
      border:0 !important;
      border-radius: var(--radius) !important;
    }

    /* ==== File Uploader: surfaces + texts ==== */
    /* Dropzone */
    div[data-testid="stFileUploader"] section {
      background: var(--input-bg) !important;
      border: 1px solid var(--border) !important;
      border-radius: var(--radius) !important;
    }
    /* All helper/limit texts */
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
      color: var(--text) !important; opacity: .92;
    }
    /* >>> Uploaded file row: filename + actions <<< */
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] {
      color: var(--text) !important;
    }

    /* File  downloader internals (drop text + limits + button) */
    div[data-testid="stDownloadButton"] p,
    div[data-testid="stDownloadButton"] span,
    div[data-testid="stDownloadButton"] [data-testid="stMarkdownContainer"] {
      color: var(--text) !important; opacity: .92;
    }
    div[data-testid="stDownloadButton"] button {
      background: var(--primary) !important; color:#fff !important; border:0 !important;
      border-radius: var(--radius) !important;
    }

    /* The 'Browse files' button inside the uploader */
    div[data-testid="stFileUploader"] button {
      background: var(--primary) !important; color:#fff !important; border:0 !important;
      border-radius: var(--radius) !important;
    }

    /* Labels (e.g., 'Choose input type:') + Radio options */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--text) !important; }
    div[data-testid="stRadio"] [role="radiogroup"] * { color: var(--text) !important; }
    """

    dark = """
    :root { --text:#FFFFFF; --bg:#111111; --bg2:#222222; --input-bg:#222222; --border:#444444; }
    [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
    [data-testid="stHeader"], [data-testid="stDecoration"] { background: var(--bg) !important; color: var(--text) !important; }
    section[data-testid="stSidebar"] { background: var(--bg2); color: var(--text); }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    h1,h2,h3,h4,h5,h6 { color: var(--text) !important; }
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[role="combobox"] { background: var(--input-bg) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
    """

    light = """
    :root { --text:#0B0B0C; --bg:#FFFFFF; --bg2:#F6F8FA; --input-bg:#FFFFFF; --border:#DDE1E5; }
    [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
    [data-testid="stHeader"], [data-testid="stDecoration"] { background: var(--bg) !important; color: var(--text) !important; }
    section[data-testid="stSidebar"] { background: var(--bg2); color: var(--text); }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    h1,h2,h3,h4,h5,h6 { color: var(--text) !important; }
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[role="combobox"] { background: var(--input-bg) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
    """

    st.markdown(f"<style>{common + (dark if mode=='Dark' else light)}</style>", unsafe_allow_html=True)

apply_theme(mode)


# Pick the right logo for the background
LOGO_DARK_BG  = "assets/slalom-white.png"  # shows on dark bg
LOGO_LIGHT_BG = "assets/slalom-blue.png"  # shows on light bg
logo_src = LOGO_DARK_BG if mode == "Dark" else LOGO_LIGHT_BG

# Header
col_logo, col_title = st.columns([1, 5])
with col_logo:
    # Newer Streamlit: width="stretch" is the preferred way
    st.image(logo_src, width="stretch")
with col_title:
    st.markdown("<h1 style='margin-bottom:0;'>Backpack Detection with YOLO v11</h1>", unsafe_allow_html=True)
    st.caption("AI-Powered Solutions & Services")

# Title
#st.title("Backpack Detection with YOLOv8")
st.subheader("Upload an image or video to detect backpacks")

# Initialize the YOLO model
@st.cache_resource
def load_model():
    # Load YOLOv8 model (will download if not present)
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple GPU
    model = YOLO("yolo11l.pt") 
    return model

model = load_model()

# Upload file section
upload_option = st.radio("Choose input type:", ["Image", "Video"])

if upload_option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process image
        with st.spinner("Detecting backpacks..."):
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Perform backpack detection (filter for class 'backpack')
            results = model(image)
            
            # Filter for backpacks (class 24 in COCO dataset)
            backpack_class_id = 24
            
            # Process results
            result_image = image.copy()
            backpack_count = 0
            
            # Draw bounding boxes for detected backpacks
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == backpack_class_id:
                        backpack_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Draw bounding box
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add label
                        label = f"Backpack: {conf:.2f}"
                        cv2.putText(result_image, label, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Convert from BGR to RGB for display
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Display result
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
            with col2:
                st.image(result_image, caption="Detected Backpacks", use_container_width=True)
            
            st.success(f"Found {backpack_count} backpack(s) in the image.")

else:  # Video option
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Save input
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

        with st.spinner("Processing video for backpack detection..."):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error opening video file")
            else:
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps    = cap.get(cv2.CAP_PROP_FPS)
                # Fallback when FPS is missing/0
                fps = 30 if (not fps or fps <= 0) else fps
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

                # Writer (prefer H.264/AVC)
                temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # try H.264/AVC
                out = cv2.VideoWriter(temp_output_file, fourcc, fps, (width, height))
                if not out.isOpened():
                    # Fallback to mp4v; convert later if needed
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_output_file, fourcc, fps, (width, height))

                progress_bar = st.progress(0)
                processed_frames = 0
                total_backpacks = 0
                backpack_class_id = 24

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)
                    frame_backpacks = 0
                    for r in results:
                        for b in r.boxes:
                            if int(b.cls[0]) == backpack_class_id:
                                frame_backpacks += 1
                                x1, y1, x2, y2 = map(int, b.xyxy[0])
                                conf = float(b.conf[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                                cv2.putText(frame, f"Backpack: {conf:.2f}", (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                    cv2.putText(frame, f"Backpacks: {frame_backpacks}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    total_backpacks += frame_backpacks

                    out.write(frame)  # <-- critical: write each processed frame

                    processed_frames += 1
                    if frame_count:
                        progress_bar.progress(min(1.0, processed_frames / frame_count))

                cap.release(); out.release()

        # If the file was written with mp4v and doesn't play, convert to H.264 (optional):
        # import subprocess, os
        # h264_path = temp_output_file.replace(".mp4", "_h264.mp4")
        # subprocess.run(["ffmpeg", "-y", "-i", temp_output_file, "-vcodec", "libx264", "-an", h264_path], check=False)
        # display_path = h264_path if os.path.exists(h264_path) else temp_output_file
        display_path = temp_output_file

        st.video(display_path)
        st.caption(f"Frames: {processed_frames} • Total detections (sum over frames): {total_backpacks}")
        with open(display_path, "rb") as f:
            st.download_button("Download processed video", f, file_name="backpacks.mp4")
