# gun_follow_hand.py
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from pathlib import Path

# --- 設定 ---
GUN_PATH = "C:\kaihatu\gungun.png"  # 銃画像（透過PNG）
CAM_ID = 0
MAX_HANDS = 1
SMOOTH_ALPHA_POS = 0.35   # 位置スムージング係数 (0..1)
SMOOTH_ALPHA_ANGLE = 0.45 # 角度スムージング係数
GUN_SCALE = 0.9           # 手幅に対する銃の相対スケール（調整可）
OFFSET_X_RATIO = 0.0      # ガン画像中心のオフセット（手位置に合わせて微調整）
OFFSET_Y_RATIO = -0.15    # ガンをやや上にずらす（手の位置基準）

# --- MediaPipe Hands 初期化 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

mp_drawing = mp.solutions.drawing_utils

# --- PNG 透過読み込み関数 ---
def load_rgba(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {path}")
    if img.shape[2] == 3:
        # アルファが無ければ追加（不透明）
        b,g,r = cv2.split(img)
        a = np.ones(b.shape, dtype=b.dtype) * 255
        img = cv2.merge((b,g,r,a))
    return img

# --- 透明画像の回転・リサイズ・重ね合わせ ---
def transform_and_overlay(bg, fg_rgba, center, angle_deg, scale):
    """
    bg: BGR background image (H,W,3)
    fg_rgba: RGBA image (h,w,4)
    center: (x,y) 画像上の位置（整数）
    angle_deg: 回転角（度）
    scale: スケール係数（1 = 元画像サイズ）
    """
    h, w = fg_rgba.shape[:2]
    # 回転行列（中心は画像中央）
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, scale)
    # 出力サイズは fg の大きさのまま回転（余白は透過になる）
    rotated = cv2.warpAffine(fg_rgba, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # overlay の左上座標を計算（中心基準）
    cx, cy = center
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = x1 + w
    y2 = y1 + h

    # 布石: 背景の切り出し範囲に合わせる（境界チェック）
    H, W = bg.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x2), min(H, y2)
    if sx1 >= sx2 or sy1 >= sy2:
        return bg  # 完全に画面外

    # fg の切り出し領域
    fx1 = sx1 - x1
    fy1 = sy1 - y1
    fx2 = fx1 + (sx2 - sx1)
    fy2 = fy1 + (sy2 - sy1)

    fg_crop = rotated[fy1:fy2, fx1:fx2]  # RGBA
    bg_crop = bg[sy1:sy2, sx1:sx2]

    # アルファ合成
    fg_bgr = fg_crop[..., :3].astype(float)
    fg_a = fg_crop[..., 3:].astype(float) / 255.0
    bg_part = bg_crop.astype(float)

    out_part = fg_bgr * fg_a + bg_part * (1.0 - fg_a)
    bg[sy1:sy2, sx1:sx2] = out_part.astype(np.uint8)
    return bg

# --- 補助: ベクトル角度計算 ---
def angle_from_points(p_from, p_to):
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    # atan2 の dy,dx は画面座標（y増は下方向）なので符号はそのままで良い
    return math.degrees(math.atan2(dy, dx))

# --- メインループ ---
def main():
    gun_rgba = load_rgba(GUN_PATH)
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    smoothed_pos = None
    smoothed_angle = None

    print("起動: qキーで終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # 鏡表示
        H, W = frame.shape[:2]

        # MediaPipe に渡す
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # デフォルトの描画（ランドマーク表示が欲しければ有効化）
        # if results.multi_hand_landmarks:
        #     for lm in results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand = results.multi_hand_landmarks[0]  # 最初の手を使用
            # 重要ランドマーク: WRIST, INDEX_FINGER_MCP / INDEX_FINGER_TIP
            wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
            index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # ピクセル座標に変換
            wrist_px = (int(wrist.x * W), int(wrist.y * H))
            index_px = (int(index_tip.x * W), int(index_tip.y * H))

            # --- 手の平中心を計算（WRIST と 各指の MCP の平均） ---
            mcp_points = [
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.RING_FINGER_MCP,
                mp_hands.HandLandmark.PINKY_MCP,
            ]
            sum_x = wrist.x
            sum_y = wrist.y
            count = 1
            for lm_id in mcp_points:
                lm = hand.landmark[lm_id]
                sum_x += lm.x
                sum_y += lm.y
                count += 1
            palm_center_norm = (sum_x / count, sum_y / count)  # 正規化座標 (0..1)

            # ピクセル座標に変換
            palm_center_px = (int(palm_center_norm[0] * W), int(palm_center_norm[1] * H))

            # 中心位置を手の平中心に合わせる（見た目の微調整は OFFSET_X_RATIO / OFFSET_Y_RATIO を使用）
            center_x = palm_center_px[0] + int((index_px[0] - wrist_px[0]) * OFFSET_X_RATIO)
            center_y = palm_center_px[1] + int((index_px[1] - wrist_px[1]) * OFFSET_Y_RATIO)

            # 回転角度（手首 -> 指先）
            angle = angle_from_points(wrist_px, index_px)
            # 画像の向き（銃のデザインに合わせて + / - 補正すること）
            # 例: 銃の元画像が右向きなら angle のままでOK。上下反転などはここで調整。

            # スケール: 手の幅（wrist<->index_mcp の距離）に基づく
            # あるいは手のbboxや画面高さを使って固定比率にしてもよい
            base_dist = math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)
            # base_dist は 0..1 の正規化値 (landmark の正規化座標)
            # これを元に銃画像のサイズを調整（調整係数は実験的に決める）
            scale = (base_dist * 6.0) * GUN_SCALE  # 係数6は経験則（調整してください）

            # スムージング
            if smoothed_pos is None:
                smoothed_pos = (center_x, center_y)
                smoothed_angle = angle
            else:
                smoothed_pos = (
                    int(smoothed_pos[0] * (1 - SMOOTH_ALPHA_POS) + center_x * SMOOTH_ALPHA_POS),
                    int(smoothed_pos[1] * (1 - SMOOTH_ALPHA_POS) + center_y * SMOOTH_ALPHA_POS),
                )
                # 角度は循環（-180..180）に注意して線形補間
                a_prev = smoothed_angle
                a_new = angle
                # 差を最短方向に正規化
                diff = (a_new - a_prev + 180) % 360 - 180
                smoothed_angle = a_prev + diff * SMOOTH_ALPHA_ANGLE

            # 銃を合成
            out = frame.copy()
            out = transform_and_overlay(out, gun_rgba, smoothed_pos, -smoothed_angle + 180, scale)
  # -angle は必要に応じて調整
            frame = out

            # デバッグ表示（手の位置 / 角度）
            cv2.circle(frame, smoothed_pos, 5, (0,255,0), -1)
            cv2.putText(frame, f"Angle: {smoothed_angle:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        else:
            # 手がないときはそのまま表示
            pass

        cv2.imshow("Gun Follow Hand", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
