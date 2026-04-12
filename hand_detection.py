import cv2
import mediapipe as mp

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Fingertip landmark IDs
TIP_IDS = [4, 8, 12, 16, 20]

def get_finger_states(landmarks):
    fingers = []

    # Thumb — compare x instead of y (thumb goes sideways)
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)  # UP
    else:
        fingers.append(0)  # DOWN

    # Index, Middle, Ring, Pinky — compare y
    for tip in [8, 12, 16, 20]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)  # UP
        else:
            fingers.append(0)  # DOWN

    return fingers  # e.g. [1, 1, 0, 0, 0]


def classify_gesture(fingers):
    # fingers = [Thumb, Index, Middle, Ring, Pinky]
    # 1 = up, 0 = down

    if fingers == [0, 0, 0, 0, 0]:
        return "FIST - STOP"

    elif fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM - HALT"

    elif fingers == [0, 1, 0, 0, 0]:
        return "POINTING - FORWARD"

    elif fingers == [0, 1, 1, 0, 0]:
        return "PEACE - ROTATE"

    elif fingers == [1, 0, 0, 0, 1]:
        return "CALL ME - SPECIAL"

    elif fingers == [0, 0, 0, 0, 1]:
        return "PINKY - BACKWARD"

    elif fingers == [1, 1, 0, 0, 0]:
        return "GUN - PICK UP"

    else:
        return "UNKNOWN"


# Open webcam
cap = cv2.VideoCapture(0)
print("Running... Press Q to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get finger states
            landmarks = hand_landmarks.landmark
            fingers = get_finger_states(landmarks)

            # Classify gesture
            gesture = classify_gesture(fingers)

            # Show finger states on screen
            finger_names = ["T", "I", "M", "R", "P"]
            for i, (name, state) in enumerate(zip(finger_names, fingers)):
                color = (0, 200, 0) if state else (0, 0, 200)
                cv2.putText(frame, name, (20 + i * 35, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show gesture label
    cv2.putText(frame, gesture, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Gesture Recognition - Step 2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()