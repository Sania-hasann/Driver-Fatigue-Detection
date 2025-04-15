import streamlit as st
import time
import pprint
import pygame
import os
import glob
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters
from PIL import ImageGrab 
from parser import get_args  

pygame.mixer.init()
sound = None  
SCREENSHOT_DIR = "screenshots"
if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# Variable to track the last time a screenshot was taken
last_screenshot_time = 0
is_webcam_open = False
cap = None

# --- Custom CSS for a more cohesive look ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f2f6; /* Light gray sidebar background */
        color: #333; /* Dark gray sidebar text */
    }
    [data-testid="stSidebar"] h1 {
        color: #2e3192; /* Consistent main blue */
        font-size: 28px;
        font-weight: bold;
        padding: 20px 15px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
        text-align: center; /* Center the sidebar heading */
    }
    .sidebar-title {
        color: #555;
        font-size: 1.1em;
        padding: 10px 15px;
    }
    [data-testid="stSidebar"] .st-selectbox > label {
        color: #555;
        font-size: 1em;
        margin-top: 15px;
        margin-left: 15px;
    }
    [data-testid="stSidebar"] .st-selectbox select {
        background-color: white;
        color: #333;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 8px 12px;
        font-size: 1em;
        margin: 5px 15px;
    }
    [data-testid="stSidebar"] .st-selectbox select:focus {
        outline: none;
        border-color: #007bff;
    }
    .st-title {
        color: #2e3192; /* Consistent main blue */
        font-size: 36px;
        margin-bottom: 30px;
        text-align: center; /* Center the main title */
    }
    .button-container {
        display: flex;
        gap: 10px; /* Space between buttons */
        justify-content: center; /* Center the buttons horizontally */
        margin-bottom: 20px; /* Space below buttons */
    }
    .st-button {
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
    }
    .st-button:first-child button {
        background-color: #28a745;
        color: white;
        border: 1px solid #28a745;
    }
    .st-button:last-child button {
        background-color: #dc3545;
        color: white;
        border: 1px solid #dc3545;
    }
    .st-empty {
        border: 2px dashed #ccc;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        color: #777;
        font-style: italic;
    }
    .st-info {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .st-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .st-warning {
        background-color: #fff3cd;
        color: #85640a;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_sound(filepath):
    try:
        return pygame.mixer.Sound(filepath)
    except pygame.error as e:
        st.error(f"Error loading sound file '{filepath}': {e}")
        return None

def capture_screenshot(alert_type):
    """Captures a screenshot of the entire window and saves it."""
    global last_screenshot_time
    current_time = time.time()
    if current_time - last_screenshot_time >= 3:
        try:
            screenshot = ImageGrab.grab()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SCREENSHOT_DIR, f"{alert_type}_{timestamp}.png")
            screenshot.save(filename)
            st.toast(f"Screenshot saved: {filename}", icon="📸")
            last_screenshot_time = current_time
        except Exception as e:
            st.error(f"Error capturing screenshot: {e}")

@st.cache_resource
def load_models(camera_params_path):
    if camera_params_path:
        camera_matrix, dist_coeffs = load_camera_parameters(camera_params_path)
    else:
        camera_matrix, dist_coeffs = None, None

    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )
    eye_det = EyeDet(show_processing=False)  
    head_pose = HeadPoseEst(show_axis=False, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    return detector, eye_det, head_pose

def process_frame(frame, detector, eye_det, head_pose, scorer, frame_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, axis=2)
    gray = np.concatenate([gray, gray, gray], axis=2)
    lms = detector.process(gray).multi_face_landmarks
    alert_messages = []
    if lms:
        landmarks = get_landmarks(lms)
        eye_det.show_eye_keypoints(color_frame=frame, landmarks=landmarks, frame_size=frame_size)
        ear = eye_det.get_EAR(landmarks=landmarks)
        tired, perclos_score = scorer.get_rolling_PERCLOS(time.perf_counter(), ear)
        gaze = eye_det.get_Gaze_Score(frame=gray, landmarks=landmarks, frame_size=frame_size)
        frame_det, roll, pitch, yaw = head_pose.get_pose(frame=frame, landmarks=landmarks, frame_size=frame_size)

        asleep, looking_away, distracted = scorer.eval_scores(
            t_now=time.perf_counter(), ear_score=ear, gaze_score=gaze, head_roll=roll, head_pitch=pitch, head_yaw=yaw
        )

        if frame_det is not None:
            frame = frame_det

        if tired:
            alert_messages.append("TIRED!")
            capture_screenshot("Tired")
        if asleep:
            alert_messages.append("ASLEEP!")
            sound.play(loops=0)
            capture_screenshot("Asleep")
        if looking_away:
            alert_messages.append("LOOKING AWAY!")
            sound.play(loops=0)
            capture_screenshot("LookingAway")
        if distracted:
            alert_messages.append("DISTRACTED!")
            sound.play(loops=0)
            capture_screenshot("Distracted")

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color_alert = (0, 0, 255)
        y_position = 30
        for message in alert_messages:
            cv2.putText(frame, message, (10, y_position), font, font_scale, text_color_alert, font_thickness, cv2.LINE_AA)
            y_position += 30

        font_pose = cv2.FONT_HERSHEY_PLAIN
        font_scale_pose = 1.5
        font_thickness_pose = 1
        text_color_pose = (0, 255, 0)
        x_position_pose = frame.shape[1] - 150
        y_position_pose = 40
        if roll is not None:
            cv2.putText(frame, f"Roll:{roll.round(1)[0]}", (x_position_pose, y_position_pose), font_pose, font_scale_pose, text_color_pose, font_thickness_pose, cv2.LINE_AA)
            y_position_pose += 30
        if pitch is not None:
            cv2.putText(frame, f"Pitch:{pitch.round(1)[0]}", (x_position_pose, y_position_pose), font_pose, font_scale_pose, text_color_pose, font_thickness_pose, cv2.LINE_AA)
            y_position_pose += 30
        if yaw is not None:
            cv2.putText(frame, f"Yaw:{yaw.round(1)[0]}", (x_position_pose, y_position_pose), font_pose, font_scale_pose, text_color_pose, font_thickness_pose, cv2.LINE_AA)

    return frame

def main():
    global sound, cap, is_webcam_open

    args = get_args()  
    sound = load_sound(r'640g_alarm-83662.mp3')

    detector, eye_det, head_pose = load_models(args.camera_params)

    scorer = AttScorer(
        t_now=time.perf_counter(), ear_thresh=args.ear_thresh, gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh, pitch_thresh=args.pitch_thresh, yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh, gaze_thresh=args.gaze_thresh, pose_time_thresh=args.pose_time_thresh,
        verbose=args.verbose if 'verbose' in args else False # Handle potential missing verbose arg
    )

    st.sidebar.title("Driver Fatigue Detection")
    st.sidebar.title("Navigation")
    menu = ["Home", "Screenshots"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("Driver Fatigue Detection")
        
        # Create 3 columns: button - spacer - button
        col1, spacer, col2 = st.columns([1, 3, 1])  

        open_clicked = col1.button("Open Webcam")
        close_clicked = col2.button("Close Webcam")
        
        video_placeholder = st.empty()


        if open_clicked:
            is_webcam_open = True
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                st.error("Cannot open webcam")
                is_webcam_open = False

        if close_clicked:
            is_webcam_open = False
            if cap is not None and cap.isOpened():
                cap.release()
            video_placeholder.empty()

        if is_webcam_open:
            if cap is not None and cap.isOpened():
                prev_time = time.perf_counter()
                while is_webcam_open:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error reading frame from webcam")
                        break
                    frame = cv2.flip(frame, 2) if args.camera == 0 else frame
                    frame_size = frame.shape[1], frame.shape[0]
                    processed_frame = process_frame(frame, detector, eye_det, head_pose, scorer, frame_size)

                    # Calculate FPS (optional, for display)
                    current_time = time.perf_counter()
                    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                    cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)

                    video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                else:
                    st.warning("Webcam is not open.")

    elif choice == "Screenshots":
        st.title("Captured Screenshots")
        image_files = sorted(glob.glob(os.path.join(SCREENSHOT_DIR, "*.png")), key=os.path.getmtime, reverse=True)
        if not image_files:
            st.info("No screenshots captured yet.")
        else:
            num_cols = 3
            for i in range(0, len(image_files), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    index = i + j
                    if index < len(image_files):
                        img_file = image_files[index]
                        try:
                            image = Image.open(img_file)
                            cols[j].image(image, caption=os.path.basename(img_file), use_container_width=True)
                        except Exception as e:
                            cols[j].error(f"Error loading image: {e}")

if __name__ == "__main__":
    main()