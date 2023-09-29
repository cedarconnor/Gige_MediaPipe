import cv2
import mediapipe as mp
from pythonosc import udp_client

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# OSC config
client = udp_client.SimpleUDPClient("127.0.0.1", 8888)

# Capture video with OpenCV
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=8) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks is not None:  # Type hint to avoid IDE warning
            for hand_id, hand_lm in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for landmark_id, lm in enumerate(hand_lm.landmark):  # Renamed 'id' to 'landmark_id'
                    client.send_message(f"/hand/{hand_id}/{landmark_id}/x", lm.x)
                    client.send_message(f"/hand/{hand_id}/{landmark_id}/y", lm.y)
                    client.send_message(f"/hand/{hand_id}/{landmark_id}/z", lm.z)

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
