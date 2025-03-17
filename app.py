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
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.85
CONSEC_FRAMES = 10  # Number of consecutive frames for drowsiness
YAWN_CONSEC_FRAMES = 20
ALARM_COOLDOWN = 5  # seconds

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

# Streamlit app
st.title("Real-time Drowsiness Detection")
st.markdown("---")

# Detection parameters
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14, 78, 308]

# Session state initialization
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_alarm' not in st.session_state:
    st.session_state.last_alarm = 0
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'yawn_counter' not in st.session_state:
    st.session_state.yawn_counter = 0

# Control buttons
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("Start Detection")
with col2:
    stop_btn = st.button("Stop Detection")

# Status indicators
status_text = st.empty()
ear_text = st.empty()
mar_text = st.empty()
frame_placeholder = st.empty()

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
        
        # Update display
        frame_placeholder.image(processed_frame)
        ear_text.write(f"**EAR:** `{ear:.2f}`")
        mar_text.write(f"**MAR:** `{mar:.2f}`")
        status_text.write(f"**Status:** `{status}`")
        
    cap.release()

# Button handlers
if start_btn:
    st.session_state.running = True
    main()

if stop_btn:
    st.session_state.running = False
    st.session_state.counter = 0
    st.session_state.yawn_counter = 0
    status_text.write("**Status:** `Stopped`")

st.markdown("---")
st.info("ℹ️ Keep your face visible in the camera feed. System will alert if drowsiness or yawning is detected.")
