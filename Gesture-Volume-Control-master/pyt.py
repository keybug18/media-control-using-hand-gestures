import cv2
import numpy as np
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        height, width, _ = image.shape

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmarks
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                pinky_tip = hand_landmarks.landmark[20]

                # Get landmark positions
                thumb_tip_x, thumb_tip_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
                index_tip_x, index_tip_y = int(index_tip.x * width), int(index_tip.y * height)
                pinky_tip_x, pinky_tip_y = int(pinky_tip.x * width), int(pinky_tip.y * height)

                # Calculate hand direction
                if thumb_tip_y < index_tip_y < pinky_tip_y:
                    pyautogui.press('volumeup')
                elif thumb_tip_y > index_tip_y > pinky_tip_y:
                    pyautogui.press('volumedown')
                elif thumb_tip_y < index_tip_y and pinky_tip_y:
                    pyautogui.press('volumemute')

        cv2.imshow('Media Control', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
