import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

def get_finger_status(landmarks):
    """
    Checks if each finger is up (1) or down (0).
    Tip IDs: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
    PIP IDs: Thumb=3, Index=6, Middle=10, Ring=14, Pinky=18
    """
    fingers = []

    # Thumb check
    if landmarks[4].y < landmarks[3].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers: Check if tip is above PIP joint
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip, pip in zip(tip_ids, pip_ids):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def classify_gesture(fingers):
    if fingers == [1, 1, 1, 1, 1]:
        return "PLAY/PAUSE", "playpause"
    elif fingers == [0, 0, 0, 0, 0]:
        return "MUTE", "volumemute"
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "NEXT", "nexttrack"
    elif fingers == [1, 0, 0, 0, 0]:
        return "VOL_UP", "volumeup"
    return "NEUTRAL", None

def main():
    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Download 'hand_landmarker.task' before running."
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Unable to access system webcam (index 0).")

    # State machine and debounce setup
    current_candidate = "NEUTRAL"
    candidate_start_time = 0.0
    last_trigger_time = 0.0
    active_latched_gesture = "NEUTRAL"  # Latches event to avoid repeated firing
    last_action_display = "NONE"

    CONFIRMATION_DELAY = 0.5  # Seconds held before recognition
    TRIGGER_COOLDOWN = 1.2    # Seconds lockout between distinct events

    print("[INFO] Camera started. Press 'q' in the window to safely exit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("[WARNING] Frame capture dropped.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            detected_label = "NEUTRAL"
            action_key = None

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        p1 = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                        p2 = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)

                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                    fingers = get_finger_status(hand_landmarks)
                    detected_label, action_key = classify_gesture(fingers)

            now = time.time()

            # Debounce & Latched Event Firing
            if detected_label != "NEUTRAL":
                if detected_label != current_candidate:
                    current_candidate = detected_label
                    candidate_start_time = now
                elif (now - candidate_start_time) >= CONFIRMATION_DELAY:
                    # Fire only if not already latched on this gesture and cooldown passed
                    if (detected_label != active_latched_gesture) and ((now - last_trigger_time) >= TRIGGER_COOLDOWN):
                        active_latched_gesture = detected_label
                        last_trigger_time = now
                        last_action_display = f"{detected_label} -> {action_key}"
                        if action_key:
                            try:
                                pyautogui.press(action_key)
                                print(f"[EVENT] Action triggered: {action_key}")
                            except Exception as e:
                                print(f"[WARN] Keypress simulated error: {e}")
            else:
                current_candidate = "NEUTRAL"
                active_latched_gesture = "NEUTRAL"

            # Overlay HUD
            cv2.rectangle(frame, (10, 10), (360, 110), (20, 20, 20), -1)
            cv2.putText(frame, f"Candidate : {detected_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Latched   : {active_latched_gesture}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' to Exit Safely", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow("Hand Gesture Media Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("[INFO] Releasing camera and closing detector resources...")
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[INFO] Clean exit completed.")

if __name__ == "__main__":
    main()