import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math


CAM_WIDTH = 960
CAM_HEIGHT = 540
BACKGROUND_COLOR = (20, 20, 40)
TEXT_COLOR = (240, 240, 240)
BUBBLE_COLOR = (255, 180, 90)
TRAIL_COLOR = (255, 255, 255)


def spawn_bubble():
    radius = random.randint(30, 55)
    x = random.randint(radius, CAM_WIDTH - radius)
    y = CAM_HEIGHT + radius
    rise_speed = random.uniform(2.5, 4.5)
    wobble_phase = random.uniform(0, np.pi * 2)
    wobble_speed = random.uniform(0.005, 0.012)
    wobble_amplitude = random.uniform(12, 28)
    return {
        "x": x,
        "y": float(y),
        "radius": radius,
        "rise_speed": rise_speed,
        "wobble_phase": wobble_phase,
        "wobble_speed": wobble_speed,
        "wobble_amplitude": wobble_amplitude,
    }


def draw_instructions(canvas, score, best_reaction, cooldown):
    cv2.putText(
        canvas,
        "Bubble Popper - index finger only",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        TEXT_COLOR,
        2,
    )
    cv2.putText(
        canvas,
        "Move your index finger tip into the bubbles to pop them!",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        1,
    )
    cv2.putText(
        canvas,
        "Press Q to exit.",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        1,
    )
    cv2.putText(
        canvas,
        f"Score: {score}",
        (20, CAM_HEIGHT - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        TEXT_COLOR,
        2,
    )
    if best_reaction is not None:
        cv2.putText(
            canvas,
            f"Best reaction: {best_reaction:0.2f}s",
            (CAM_WIDTH - 260, CAM_HEIGHT - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
        )
    if cooldown > 0:
        cv2.putText(
            canvas,
            "Good hit!",
            (CAM_WIDTH // 2 - 80, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            TEXT_COLOR,
            2,
        )


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    bubbles = []
    score = 0
    best_reaction = None
    last_spawn = time.time()
    last_pop_time = None
    pop_feedback_cooldown = 0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        mp_drawing = mp.solutions.drawing_utils

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            ret, frame = cap.read()
            if not ret:
                print("Unable to access webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            finger_tip = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger_tip = (int(tip.x * CAM_WIDTH), int(tip.y * CAM_HEIGHT))
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            now = time.time()
            if now - last_spawn > 1.6:
                bubbles.append(spawn_bubble())
                last_spawn = now

            canvas = np.full((CAM_HEIGHT, CAM_WIDTH, 3), BACKGROUND_COLOR, np.uint8)

            for bubble in bubbles[:]:
                bubble["y"] -= bubble["rise_speed"]
                bubble["wobble_phase"] += bubble["wobble_speed"]
                wobble = math.sin(bubble["wobble_phase"]) * bubble["wobble_amplitude"]
                center = (
                    int(bubble["x"] + wobble),
                    int(bubble["y"]),
                )

                if finger_tip:
                    distance = np.linalg.norm(
                        np.array(finger_tip) - np.array(center)
                    )
                    if distance <= bubble["radius"]:
                        bubbles.remove(bubble)
                        score += 1
                        if last_pop_time is not None:
                            reaction = now - last_pop_time
                            if best_reaction is None or reaction < best_reaction:
                                best_reaction = reaction
                        last_pop_time = now
                        pop_feedback_cooldown = 20
                        continue

                cv2.circle(canvas, center, bubble["radius"], BUBBLE_COLOR, 2)
                cv2.circle(canvas, center, bubble["radius"] - 8, (100, 100, 150), 2)
                highlight_offset = int(bubble["radius"] * 0.5)
                cv2.circle(
                    canvas,
                    (center[0] - highlight_offset, center[1] - highlight_offset),
                    8,
                    (255, 255, 255),
                    -1,
                )

                if center[1] + bubble["radius"] < 0:
                    bubbles.remove(bubble)

            if finger_tip:
                cv2.circle(canvas, finger_tip, 12, TRAIL_COLOR, -1)

            pop_feedback_cooldown = max(0, pop_feedback_cooldown - 1)
            draw_instructions(canvas, score, best_reaction, pop_feedback_cooldown)

            debug_cam = cv2.resize(frame, (240, 135))
            canvas[10 : 10 + 135, CAM_WIDTH - 250 : CAM_WIDTH - 10] = debug_cam

            cv2.imshow("Bubble Pop Game", canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
