import cv2
import numpy as np
import random
import time


CAM_WIDTH = 960
CAM_HEIGHT = 540
TEXT_COLOR = (255, 255, 255)
TARGET_COLOR = (0, 200, 255)
PLAYER_COLOR = (80, 255, 120)

# Predefined HSV range for a vivid green object.
LOWER_GREEN = np.array([40, 80, 80])
UPPER_GREEN = np.array([85, 255, 255])


def random_target():
    radius = random.randint(35, 55)
    return {
        "x": random.randint(radius, CAM_WIDTH - radius),
        "y": random.randint(radius, CAM_HEIGHT - radius),
        "radius": radius,
        "created_at": time.time(),
    }


def draw_overlay(canvas, score, combo, remaining_time):
    cv2.putText(
        canvas,
        "Colour Chase - show a vivid green object to move.",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
    )
    cv2.putText(
        canvas,
        "Hit the glowing orb quickly for combo bonus. Press Q to exit.",
        (20, 65),
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
    cv2.putText(
        canvas,
        f"Combo x{combo}",
        (200, CAM_HEIGHT - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        TEXT_COLOR,
        2,
    )
    cv2.putText(
        canvas,
        f"Target expires in: {remaining_time:0.1f}s",
        (CAM_WIDTH - 260, CAM_HEIGHT - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        1,
    )


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    score = 0
    combo = 1
    target = random_target()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        ret, frame = cap.read()
        if not ret:
            print("Unable to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        player_pos = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 800:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    player_pos = (cx, cy)

        canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        remaining_time = max(0.0, 6.0 - (time.time() - target["created_at"]))
        pulse = 1 + 0.1 * np.sin(time.time() * 6)
        animated_radius = int(target["radius"] * pulse)
        cv2.circle(
            canvas,
            (target["x"], target["y"]),
            animated_radius,
            TARGET_COLOR,
            2,
        )
        cv2.circle(
            canvas,
            (target["x"], target["y"]),
            max(8, animated_radius // 4),
            TARGET_COLOR,
            -1,
        )

        if player_pos:
            cv2.circle(canvas, player_pos, 18, PLAYER_COLOR, -1)
            cv2.circle(canvas, player_pos, 24, PLAYER_COLOR, 2)

            distance = np.linalg.norm(
                np.array(player_pos) - np.array([target["x"], target["y"]])
            )
            if distance < target["radius"]:
                elapsed = time.time() - target["created_at"]
                bonus = max(5, int(40 - elapsed * 6))
                score += bonus * combo
                combo = min(combo + 1, 5)
                target = random_target()
                remaining_time = 6.0

        if remaining_time <= 0:
            combo = 1
            target = random_target()

        draw_overlay(canvas, score, combo, remaining_time)

        debug_cam = cv2.resize(frame, (240, 135))
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored = cv2.resize(mask_colored, (240, 135))
        canvas[10 : 10 + 135, CAM_WIDTH - 250 : CAM_WIDTH - 10] = debug_cam
        canvas[155 : 155 + 135, CAM_WIDTH - 250 : CAM_WIDTH - 10] = mask_colored

        cv2.imshow("Colour Chase Game", canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
