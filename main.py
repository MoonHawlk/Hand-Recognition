import cv2
import mediapipe as mp

# Initialize hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence = 0.5)

def detect_gesture(hand_landmarks):
    # Get landmark coordinates
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

    # Extract tip landmarks for fingers
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Calculate distance between thumb tip and palm (wrist)
    palm_center = landmarks[0]
    thumb_palm_distance = ((thumb_tip[0] - palm_center[0]) ** 2 + (thumb_tip[1] - palm_center[1]) ** 2) ** 0.5

    # Define thresholds for gesture recognition
    finger_open_threshold = 0.1  # Adjust this value based on your setup
    thumb_palm_open_threshold = 0.15  # Adjust this value based on your setup

    # Detect hand gestures
    if thumb_palm_distance > thumb_palm_open_threshold:
        return "open_hand"

    elif all(landmarks[i][1] > middle_tip[1] for i in range(4)):
        # Closed fist gesture (all fingertips are below the middle finger)
        return "closed_hand"

    elif thumb_tip[1] < middle_tip[1] and thumb_tip[1] < index_tip[1]:
        # Thumbs-up gesture (thumb is lower than both middle and index fingers)
        return "thumbs_up"

    else:
        # Gesture not recognized
        return "unknown"

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        # Convert the image to RGB and process it through the hand tracking model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get gesture from hand landmarks
                gesture = detect_gesture(hand_landmarks)
                
                if gesture == "open_hand":
                    print('Mão aberta')
                elif gesture == "closed_hand":
                    print('Mão meio aberta')
                elif gesture == "thumbs_up":
                    print('Mão fechada')
                elif gesture == "unknown":
                    print('Gesto desconhecido')
                else:
                    print('Mão não identificada')

        cv2.imshow("Hand Gesture Recognition", image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
