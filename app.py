import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("🤟 Sign-to-Text Banking Translator")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture dictionary (expand later with ML model)
gesture_dict = {
    "OPEN_PALM": "Check Balance",
    "PINCH": "Deposit",
    "VICTORY": "Withdraw"
}

def classify_gesture(landmarks):
    # Placeholder: rule-based or ML model
    # For now, always return OPEN_PALM
    return "OPEN_PALM"

# Streamlit video capture
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

    # Show video + text
    cv2.putText(img, text_output, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    stframe.image(img, channels="BGR")