import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from pygame import mixer
import time

# Configuration
ALARM_PATH = r"utils/alarm.WAV"
FACE_MODEL_PATH =  r"models/yolov8n-face.pt"
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.85
CONSEC_FRAMES = 10
YAWN_CONSEC_FRAMES = 20
ALARM_COOLDOWN = 5

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #1a1a1a;}
    .stMarkdown h1 {color: #4a90e2;}
    .metric-box {padding: 15px; border-radius: 10px; margin: 10px 0;}
    .alarm-active {background-color: #ff4b4b !important; animation: pulse 1s infinite;}
    @keyframes pulse {0% {transform: scale(1);} 50% {transform: scale(1.05);} 100% {transform: scale(1);}}
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
st.title("🚗 Real-time Drowsiness Detection System")
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    ear_threshold = st.slider("EAR Threshold", 0.1, 0.5, EAR_THRESHOLD, 0.01)
    mar_threshold = st.slider("MAR Threshold", 0.5, 1.5, MAR_THRESHOLD, 0.01)
    st.markdown("---")
    st.subheader("ℹ️ Instructions")
    st.markdown("""
    1. Ensure proper lighting
    2. Face the camera directly
    3. Remove sunglasses
    4. Maintain neutral expression
    """)

# Main Content
col1, col2 = st.columns([1, 2])

with col1:
    # Status Panel
    st.subheader("📊 Real-time Metrics")
    status_container = st.container()
    
    with status_container:
        metric1, metric2 = st.columns(2)
        with metric1:
            ear_placeholder = st.empty()
        with metric2:
            mar_placeholder = st.empty()
        
        status_placeholder = st.empty()
        st.markdown("---")
        st.markdown("**Detection Log**")
        log_placeholder = st.empty()

with col2:
    # Video Feed
    st.subheader("📸 Live Camera Feed")
    frame_placeholder = st.empty()
    st.markdown("---")
    
    # Controls
    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        start_btn = st.button("▶️ Start Detection", type="primary")
    with control_col2:
        stop_btn = st.button("⏹️ Stop Detection", type="secondary")
    with control_col3:
        if st.button("ℹ️ Help"):
            st.toast("Ensure your face is visible in the camera feed!", icon="👤")

# Session state initialization
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_alarm' not in st.session_state:
    st.session_state.last_alarm = 0
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'yawn_counter' not in st.session_state:
    st.session_state.yawn_counter = 0

# Landmark indices
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14, 78, 308]

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
    
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
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
                if ear < EAR_THRESHOLD:
                    st.session_state.counter += 1
                    if st.session_state.counter >= CONSEC_FRAMES:
                        if time.time() - st.session_state.last_alarm > ALARM_COOLDOWN:
                            mixer.music.play()
                            st.session_state.last_alarm = time.time()
                        status = "DROWSY!"
                else:
                    st.session_state.counter = max(0, st.session_state.counter - 1)
                    
                # Yawn detection
                if mar > MAR_THRESHOLD:
                    st.session_state.yawn_counter += 1
                    if st.session_state.yawn_counter >= YAWN_CONSEC_FRAMES:
                        status = "YAWNING!"
                else:
                    st.session_state.yawn_counter = max(0, st.session_state.yawn_counter - 2)
    
    # Draw annotations
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, f"Status: {status}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if status == "Normal" else (255,0,0), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if mar < MAR_THRESHOLD else (255,0,0), 2)
    
    return frame, ear, mar, status

def update_ui(status, ear, mar):
    ear_color = "#ff4b4b" if ear < EAR_THRESHOLD else "#4a90e2"
    mar_color = "#ff4b4b" if mar > MAR_THRESHOLD else "#4a90e2"
    
    ear_html = f"""
    <div class="metric-box" style="border: 2px solid {ear_color};">
        <h3>👁️ EAR (Eye)</h3>
        <h2 style="color: {ear_color};">{ear:.2f}</h2>
        <p>Threshold: {EAR_THRESHOLD:.2f}</p>
    </div>
    """
    mar_html = f"""
    <div class="metric-box" style="border: 2px solid {mar_color};">
        <h3>👄 MAR (Mouth)</h3>
        <h2 style="color: {mar_color};">{mar:.2f}</h2>
        <p>Threshold: {MAR_THRESHOLD:.2f}</p>
    </div>
    """
    
    ear_placeholder.markdown(ear_html, unsafe_allow_html=True)
    mar_placeholder.markdown(mar_html, unsafe_allow_html=True)
    
    status_color = "#ff4b4b" if status != "Normal" else "#4a90e2"
    status_class = "alarm-active" if status != "Normal" else ""
    status_html = f"""
    <div class="metric-box {status_class}" style="border: 2px solid {status_color};">
        <h3>🚨 Current Status</h3>
        <h2 style="color: {status_color};">{status}</h2>
    </div>
    """
    status_placeholder.markdown(status_html, unsafe_allow_html=True)
    
    if status != "Normal":
        log_placeholder.markdown(f"`{time.strftime('%H:%M:%S')}` - {status} detected")

def main():
    cap = cv2.VideoCapture(0)
    
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break
            
        frame = cv2.resize(frame, (640, 480))
        processed_frame, ear, mar, status = process_frame(frame)
        
        frame_placeholder.image(processed_frame, use_column_width=True)
        update_ui(status, ear, mar)
        
    cap.release()

# Button handlers
if start_btn:
    st.session_state.running = True
    main()

if stop_btn:
    st.session_state.running = False
    st.session_state.counter = 0
    st.session_state.yawn_counter = 0
    status_placeholder.markdown("""
    <div class="metric-box" style="border: 2px solid #cccccc;">
        <h3>🚨 Current Status</h3>
        <h2 style="color: #cccccc;">System Stopped</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("👨💻 Developed by Anush Adonts | 📧 a.adonts05@gmail.com")

