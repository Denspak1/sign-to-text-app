import streamlit as st
import numpy as np

# Try importing cv2, but handle failure gracefully
try:
    import cv2
    import mediapipe as mp
    USE_OPENCV = True
except ImportError:
    import mediapipe as mp
    USE_OPENCV = False

st.title("🤟 Sign-to-Text Banking Translator")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gesture_dict = {
    "OPEN_PALM": "Check Balance",
    "PINCH": "Deposit",
    "VICTORY": "Withdraw"
}

def classify_gesture(landmarks):
    # Placeholder: always return OPEN_PALM for now
    return "OPEN_PALM"

if USE_OPENCV:
    # Real-time video with OpenCV
    st.write("Using OpenCV for live video")
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.write("Camera not found")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        text_output = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = classify_gesture(hand_landmarks)
                text_output = gesture_dict.get(gesture, "")

        cv2.putText(img, text_output, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(img, channels="BGR")

else:
    # Fallback: Streamlit camera input
    st.write("OpenCV not available, using Streamlit camera input")
    img_file = st.camera_input("Take a picture")

    if img_file:
        # Convert uploaded image to numpy array
        file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        text_output = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = classify_gesture(hand_landmarks)
                text_output = gesture_dict.get(gesture, "")

        cv2.putText(img, text_output, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        st.image(img, channels="BGR")
