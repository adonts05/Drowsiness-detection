import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from pygame import mixer
import time

# Configuration
ALARM_PATH = r"utils/alarm.WAV"
FACE_MODEL_PATH = r"models/yolov8n-face.pt"
DEFAULT_EAR_THRESHOLD = 0.21
DEFAULT_MAR_THRESHOLD = 0.85
CONSEC_FRAMES = 10
YAWN_CONSEC_FRAMES = 20
ALARM_COOLDOWN = 5

# Landmark indices
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14, 78, 308]

# Custom CSS styling
st.markdown("""
<style>
    .header {
        color: #2E86C1;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        box-shadow: 5px 5px 10px #d9d9d9, -5px -5px 10px #ffffff;
        margin-bottom: 2rem;
    }
    .metric-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        box-shadow: 3px 3px 6px #d9d9d9, -3px -3px 6px #ffffff;
        margin-bottom: 1rem;
    }
    .status-alert {
        color: #E74C3C !important;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.5; }
    }
    .video-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize models and mixer once
@st.cache_resource
def load_resources():
    mixer.init()
    mixer.music.load(ALARM_PATH)
    face_model = YOLO(FACE_MODEL_PATH)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return face_model, face_mesh

face_model, face_mesh = load_resources()

# App Header
st.markdown('<div class="header"><h1>🚗 Real-time Drowsiness Detection System</h1><h4>Computer Vision-powered Driver Monitoring</h4></div>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    st.session_state.ear_threshold = st.slider("EAR Threshold", 0.1, 0.5, DEFAULT_EAR_THRESHOLD, 0.01)
    st.session_state.mar_threshold = st.slider("MAR Threshold", 0.5, 1.5, DEFAULT_MAR_THRESHOLD, 0.01)
    st.session_state.consec_frames = st.slider("Consecutive Frames for Alarm", 1, 30, CONSEC_FRAMES)
    
    st.markdown("---")
    st.header("📈 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Alarms", st.session_state.get('total_alarms', 0))
    with col2:
        st.metric("Yawn Detections", st.session_state.get('total_yawns', 0))
    
    st.markdown("---")
    st.markdown("**Landmark Indices**")
    st.code(f"Eyes: {LEFT_EYE_INDICES} | {RIGHT_EYE_INDICES}")
    st.code(f"Mouth: {MOUTH_INDICES}")

# Main content layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### 🎥 Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Real-time Metrics")
    
    status_container = st.empty()
    ear_container = st.empty()
    mar_container = st.empty()
    
    st.markdown("### 📉 Counters")
    with st.container():
        st.markdown("Eye Closure Progress")
        eye_progress = st.progress(0)
        
        st.markdown("Yawn Detection Progress")
        yawn_progress = st.progress(0)

# Session state initialization
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_alarm' not in st.session_state:
    st.session_state.last_alarm = 0
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'yawn_counter' not in st.session_state:
    st.session_state.yawn_counter = 0
if 'total_alarms' not in st.session_state:
    st.session_state.total_alarms = 0
if 'total_yawns' not in st.session_state:
    st.session_state.total_yawns = 0

# Control buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start Detection", use_container_width=True):
        st.session_state.running = True
with col2:
    if st.button("⏹️ Stop Detection", use_container_width=True):
        st.session_state.running = False
        st.session_state.counter = 0
        st.session_state.yawn_counter = 0

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_points):
    vertical = np.linalg.norm(mouth_points[0] - mouth_points[1])
    horizontal = np.linalg.norm(mouth_points[2] - mouth_points[3])
    return vertical / horizontal

def process_frame(frame):
    height, width = frame.shape[:2]
    results = face_model(frame, verbose=False, conf=0.7)

    status = "Normal"
    ear = 0
    mar = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_roi = frame[y1:y2, x1:x2]

        if face_roi.size == 0:
            continue

        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(rgb_face)

        if results_face.multi_face_landmarks:
            landmarks = results_face.multi_face_landmarks[0]
            face_width = x2 - x1
            face_height = y2 - y1

            left_eye = np.array([(landmarks.landmark[i].x * face_width + x1,
                                landmarks.landmark[i].y * face_height + y1)
                               for i in LEFT_EYE_INDICES])

            right_eye = np.array([(landmarks.landmark[i].x * face_width + x1,
                                  landmarks.landmark[i].y * face_height + y1)
                                 for i in RIGHT_EYE_INDICES])

            mouth_points = np.array([(landmarks.landmark[i].x * face_width + x1,
                                     landmarks.landmark[i].y * face_height + y1)
                                    for i in MOUTH_INDICES])

            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)
            ear = (ear_left + ear_right) / 2.0
            mar = mouth_aspect_ratio(mouth_points)

            # Drowsiness detection
            if ear < st.session_state.ear_threshold:
                st.session_state.counter += 1
                if st.session_state.counter >= st.session_state.consec_frames:
                    if time.time() - st.session_state.last_alarm > ALARM_COOLDOWN:
                        mixer.music.play()
                        st.session_state.last_alarm = time.time()
                        st.session_state.total_alarms += 1
                    status = "DROWSY! 🚨"
            else:
                st.session_state.counter = max(0, st.session_state.counter - 1)

            # Yawn detection
            if mar > st.session_state.mar_threshold:
                st.session_state.yawn_counter += 1
                if st.session_state.yawn_counter >= YAWN_CONSEC_FRAMES:
                    status = "YAWNING! 😮"
                    st.session_state.total_yawns += 1
            else:
                st.session_state.yawn_counter = max(0, st.session_state.yawn_counter - 2)

    # Draw annotations
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, f"Status: {status}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if "Normal" in status else (255,0,0), 2)
    
    return frame, ear, mar, status

# Main loop
def main():
    cap = cv2.VideoCapture(0)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break

        frame = cv2.resize(frame, (640, 480))
        processed_frame, ear, mar, status = process_frame(frame)

        # Update UI components
        with video_placeholder.container():
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.image(processed_frame)
            st.markdown('</div>', unsafe_allow_html=True)

        # Update metrics
        status_color = "#2ECC71" if "Normal" in status else "#E74C3C"
        status_container.markdown(f"""
        <div class="metric-box">
            <h4 style="color: {status_color}; margin:0;">STATUS</h4>
            <h2 style="color: {status_color}; margin:0;">{status}</h2>
        </div>
        """, unsafe_allow_html=True)

        ear_color = "#E74C3C" if ear < st.session_state.ear_threshold else "#2ECC71"
        ear_container.markdown(f"""
        <div class="metric-box">
            <h4 style="color: {ear_color}; margin:0;">EAR (Eye Aspect Ratio)</h4>
            <h2 style="color: {ear_color}; margin:0;">{ear:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

        mar_color = "#E74C3C" if mar > st.session_state.mar_threshold else "#2ECC71"
        mar_container.markdown(f"""
        <div class="metric-box">
            <h4 style="color: {mar_color}; margin:0;">MAR (Mouth Aspect Ratio)</h4>
            <h2 style="color: {mar_color}; margin:0;">{mar:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Update progress bars
        eye_progress.progress(min(st.session_state.counter/st.session_state.consec_frames, 1.0))
        yawn_progress.progress(min(st.session_state.yawn_counter/YAWN_CONSEC_FRAMES, 1.0))

    cap.release()

if st.session_state.running:
    main()

st.markdown("---")
st.markdown("👨💻 Developed by Anush Adonts | 📧 a.adonts05@gmail.com")
