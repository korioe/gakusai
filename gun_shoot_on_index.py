# gun_shoot_on_index.py
import cv2
import mediapipe as mp
import numpy as np
import math, random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

@dataclass
class Settings:
    """アプリケーションの設定を管理するデータクラス"""
    gun_path: str = r"C:\kaihatu\gungun.png"
    cam_id: int = 0
    max_hands: int = 1
    
    # スムージング係数
    smooth_alpha_pos: float = 0.35
    smooth_alpha_angle: float = 0.45
    
    # 銃の表示設定
    gun_scale: float = 0.9
    offset_x_ratio: float = 0.0
    offset_y_ratio: float = -0.15 # この値は使われなくなるが、互換性のため残す
    muzzle_ratio: Tuple[float, float] = (0.10, 0.50)  # 左向きの銃画像における銃口の相対位置 (0..1)
    grip_ratio: Tuple[float, float] = (0.80, 0.70)   # 左向きの銃画像におけるグリップの相対位置 (0..1)

    # 発射関連
    projectile_speed: float = 1200.0
    projectile_lifetime: float = 2.0
    fire_cooldown: float = 0.12

    # 指の判定
    finger_bend_angle_deg: float = 80.0       # 指を曲げたと判定する角度の閾値。大きいほど判定が緩やかになります (デフォルト: 60.0)

    # ターゲット関連
    target_spawn_interval: float = 1.5  # ターゲットが出現する間隔（秒）
    max_targets: int = 7                # 画面上の最大ターゲット数
    target_radius_min: int = 20         # ターゲットの最小半径
    target_radius_max: int = 45         # ターゲットの最大半径
    target_lifetime: float = 10.0       # ターゲットの生存時間（秒）
    game_duration_seconds: float = 60.0 # 1分間の時間制限

    # MediaPipe設定
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6

# --- MediaPipe Hands 初期化 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- PNG 読み込み ---
def load_rgba(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {path}")
    if img.shape[2] == 3:
        b,g,r = cv2.split(img)
        a = np.ones(b.shape, dtype=b.dtype) * 255
        img = cv2.merge((b,g,r,a))
    return img

# --- 回転・リサイズ・重ね合わせ（かつマズル座標を返す） ---
def transform_and_overlay_with_muzzle(bg, fg_rgba, center, angle_deg, scale, muzzle_local_ratio=(0.9,0.5), rotation_center_ratio=None):
    """
    bg: BGR background image
    fg_rgba: RGBA image (h,w,4)
    center: (x,y) center position on bg where fg image's rotation center should map
    angle_deg: rotation degrees
    scale: scale factor
    muzzle_local_ratio: (x_ratio,y_ratio) in fg image coord (0..1)
    rotation_center_ratio: (x_ratio, y_ratio) in fg image coord (0..1) for the rotation pivot. Defaults to image center.
    -> returns (bg_with_fg, muzzle_global_point_tuple_or_None, M_affine, paste_box)
    """
    h, w = fg_rgba.shape[:2]
    rot_center_x = (w / 2) if rotation_center_ratio is None else rotation_center_ratio[0] * w
    rot_center_y = (h / 2) if rotation_center_ratio is None else rotation_center_ratio[1] * h
    M = cv2.getRotationMatrix2D((rot_center_x, rot_center_y), angle_deg, scale)
    rotated = cv2.warpAffine(fg_rgba, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    cx, cy = center
    x1 = int(cx - rot_center_x)
    y1 = int(cy - rot_center_y)
    x2 = x1 + w
    y2 = y1 + h

    H, W = bg.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x2), min(H, y2)
    if sx1 >= sx2 or sy1 >= sy2:
        # 画面外でもマズル計算はしたい -> 変換してグローバル座標を返す
        # まずローカルマズル座標を affine 変換
        mlx = muzzle_local_ratio[0] * w
        mly = muzzle_local_ratio[1] * h
        # transformed local:
        tx = M[0,0]*mlx + M[0,1]*mly + M[0,2]
        ty = M[1,0]*mlx + M[1,1]*mly + M[1,2]
        # global:
        muzzle_global = (int(x1 + tx), int(y1 + ty))
        return bg, muzzle_global, M, (x1,y1,w,h)

    fx1 = sx1 - x1
    fy1 = sy1 - y1
    fx2 = fx1 + (sx2 - sx1)
    fy2 = fy1 + (sy2 - sy1)

    fg_crop = rotated[fy1:fy2, fx1:fx2]
    bg_crop = bg[sy1:sy2, sx1:sx2]

    fg_bgr = fg_crop[..., :3].astype(float)
    fg_a = fg_crop[..., 3:].astype(float) / 255.0
    bg_part = bg_crop.astype(float)

    out_part = fg_bgr * fg_a + bg_part * (1.0 - fg_a)
    bg[sy1:sy2, sx1:sx2] = out_part.astype(np.uint8)

    # マズル（ローカル->回転->グローバル）
    mlx = muzzle_local_ratio[0] * w
    mly = muzzle_local_ratio[1] * h
    tx = M[0,0]*mlx + M[0,1]*mly + M[0,2]
    ty = M[1,0]*mlx + M[1,1]*mly + M[1,2]
    muzzle_global = (int(x1 + tx), int(y1 + ty))

    return bg, muzzle_global, M, (x1,y1,w,h)

# --- 角度計算 ---
def angle_from_points(p_from, p_to):
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return math.degrees(math.atan2(dy, dx))

# --- 角度のスムージング ---
def smooth_angle(prev_angle: float, new_angle: float, alpha: float) -> float:
    """循環する角度を滑らかに補間する"""
    diff = (new_angle - prev_angle + 180) % 360 - 180
    return prev_angle + diff * alpha

# --- 指の屈曲角度（PIP での角度） ---
def finger_pip_angle_deg(
    hand_landmarks,
    pip_landmark: mp_hands.HandLandmark,
    mcp_landmark: mp_hands.HandLandmark,
    tip_landmark: mp_hands.HandLandmark
) -> float:
    """指定された3つのランドマークから指の関節の角度を計算する"""
    pip = hand_landmarks[pip_landmark]
    mcp = hand_landmarks[mcp_landmark]
    tip = hand_landmarks[tip_landmark]

    v1 = np.array([mcp.x - pip.x, mcp.y - pip.y])
    v2 = np.array([tip.x - pip.x, tip.y - pip.y])
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-8 or norm_v2 < 1e-8:
        return 180.0  # 計算不能な場合はまっすぐな角度を返す
    n1 = v1 / norm_v1
    n2 = v2 / norm_v2
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    ang = math.degrees(math.acos(dot))
    return ang

# --- プロジェクタ管理クラス ---
class Projectile:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.spawn_time = time.time()
    def age(self):
        return time.time() - self.spawn_time

# --- ターゲット管理クラス ---
@dataclass
class Target:
    pos: np.ndarray
    radius: float
    spawn_time: float

    def age(self):
        return time.time() - self.spawn_time


def update_and_draw_projectiles(frame: np.ndarray, projectiles: List[Projectile], dt: float, lifetime: float):
    """弾の更新と描画を行い、生存している弾のリストを返す"""
    H, W = frame.shape[:2]
    surviving_projectiles = []

    for p in projectiles:
        p.pos += p.vel * dt
        age = p.age()

        is_alive = (
            age <= lifetime and
            -50 < p.pos[0] < W + 50 and
            -50 < p.pos[1] < H + 50
        )

        if is_alive:
            surviving_projectiles.append(p)
            # 描画
            fade = max(0.0, 1.0 - age / lifetime)
            col = (0, 180, 255)
            
            cv2.circle(frame, tuple(p.pos.astype(int)), 6, col, -1)
            # 軌跡
            trail_len = int(6 + 8 * fade)
            tail = p.pos - p.vel * 0.03
            cv2.line(frame, tuple(p.pos.astype(int)), tuple(tail.astype(int)), col, trail_len)

    return surviving_projectiles

def update_and_draw_targets(frame: np.ndarray, targets: List[Target], lifetime: float):
    """ターゲットの更新と描画を行い、生存しているターゲットのリストを返す"""
    surviving_targets = []
    for t in targets:
        age = t.age()
        if age <= lifetime:
            surviving_targets.append(t)
            # 描画（年齢に応じて点滅するようなエフェクト）
            alpha = 0.6 + 0.4 * math.sin(age * 5)
            color = (int(150 * alpha), int(150 * alpha), 255)
            cv2.circle(frame, tuple(t.pos.astype(int)), int(t.radius), color, 3)
            cv2.circle(frame, tuple(t.pos.astype(int)), int(t.radius * 0.6), (0, 200, 255), -1)
            cv2.circle(frame, tuple(t.pos.astype(int)), int(t.radius * 0.3), (255, 255, 255), -1)

    return surviving_targets


# --- メイン ---
def main():
    cfg = Settings()

    # --- 先頭（設定の直後など）: 右向き画像読み込みを追加 ---
    left_gun_rgba = load_rgba(cfg.gun_path)          # 既存の左向き画像（例: gungun.png）
    right_gun_rgba = load_rgba(r"C:\kaihatu\right.png")  # あなたがアップロードした右向き画像

    # それぞれの画像のサイズ（必要なら個別に使える）
    left_h, left_w = left_gun_rgba.shape[:2]
    right_h, right_w = right_gun_rgba.shape[:2]

    # もし左右画像でマズル位置／グリップ位置が違うなら個別に設定（例）
    left_muzzle = cfg.muzzle_ratio        # 既存の左向き比率 (0..1)
    left_grip  = cfg.grip_ratio
    # 右向き画像用に反転した比率を作る（必要に応じて手動で微調整）
    right_muzzle = (1.0 - cfg.muzzle_ratio[0], cfg.muzzle_ratio[1])
    right_grip  = (1.0 - cfg.grip_ratio[0], cfg.grip_ratio[1])

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=cfg.max_hands,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )
    cap = cv2.VideoCapture(cfg.cam_id)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    smoothed_pos = None
    smoothed_angle = None

    projectiles: List[Projectile] = []
    last_fire_time = 0.0
    prev_finger_bent = False

    targets: List[Target] = []
    last_target_spawn_time = 0.0
    score = 0
    hit_effects = [] # [pos, radius, age]

    game_start_time = time.time() # ゲーム開始時刻を記録
    game_over = False

    prev_time = time.time()
    print("起動: qキーで終了")

    while True:
        now = time.time()
        dt = now - prev_time
        prev_time = now

        # ゲーム時間計算
        elapsed_time = now - game_start_time
        time_left = max(0, cfg.game_duration_seconds - elapsed_time)

        if time_left <= 0 and not game_over:
            game_over = True

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand = results.multi_hand_landmarks[0]
            # ランドマーク取得
            wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
            index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            wrist_px = (int(wrist.x * W), int(wrist.y * H))
            index_px = (int(index_tip.x * W), int(index_tip.y * H))

            # 手のひら中心（WRIST + MCP群 の平均）
            mcp_landmarks = [
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.RING_FINGER_MCP,
                mp_hands.HandLandmark.PINKY_MCP,
            ]
            sum_x = wrist.x
            sum_y = wrist.y
            count = 1
            for lm_id in mcp_landmarks:
                lm = hand.landmark[lm_id]
                sum_x += lm.x
                sum_y += lm.y
                count += 1
            palm_center_norm = (sum_x / count, sum_y / count)
            palm_center_px = (int(palm_center_norm[0] * W), int(palm_center_norm[1] * H))

            # 銃の描画位置と角度
            center_pos = palm_center_px
            angle = angle_from_points(wrist_px, index_px)

            # スケール（指付け根距離を使う）
            base_dist = math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)
            scale = (base_dist * 6.0) * cfg.gun_scale

            # スムージング
            if smoothed_pos is None:
                smoothed_pos = center_pos
                smoothed_angle = angle
            else:
                smoothed_pos = (
                    int(smoothed_pos[0] * (1 - cfg.smooth_alpha_pos) + center_pos[0] * cfg.smooth_alpha_pos),
                    int(smoothed_pos[1] * (1 - cfg.smooth_alpha_pos) + center_pos[1] * cfg.smooth_alpha_pos),
                )
                smoothed_angle = smooth_angle(smoothed_angle, angle, cfg.smooth_alpha_angle)

            # --- メインループ内：銃選択と描画の置き換え ---
            # 1) 手向き判定（smoothed_angle は既に計算・スムージング済み）
            # 角度は画面座標系なのでそのまま使う
            facing_right = math.cos(math.radians(smoothed_angle)) > 0.0

            # 2) 使う画像と比率を選択
            if facing_right:
                gun_rgba_to_draw = right_gun_rgba
                muzzle_ratio_to_use = right_muzzle
                rotation_center_ratio = right_grip   # 銃回転中心 / グリップ比率（既存処理と整合）
                draw_angle = -smoothed_angle # 右向き画像はそのまま角度を適用
            else:
                gun_rgba_to_draw = left_gun_rgba
                muzzle_ratio_to_use = left_muzzle
                rotation_center_ratio = left_grip
                draw_angle = -smoothed_angle + 180 # 左向き画像は180度回転

            # 3) 銃の描画（transform関数に選択した画像を渡す）
            out, muzzle_global, M, paste_box = transform_and_overlay_with_muzzle(
                frame.copy(),
                gun_rgba_to_draw,
                smoothed_pos,
                draw_angle,
                scale,
                muzzle_local_ratio=muzzle_ratio_to_use,
                rotation_center_ratio=rotation_center_ratio
            )
            frame = out

            # 指の屈曲角度（人差し指 PIP）
            pip_angle = finger_pip_angle_deg(
                hand.landmark,
                mp_hands.HandLandmark.INDEX_FINGER_PIP,
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            )
            finger_bent = pip_angle < cfg.finger_bend_angle_deg

            # トリガー: 未曲げ -> 曲げ の遷移を検出し、クールダウン内でなければ発射
            if not prev_finger_bent and finger_bent:
                if now - last_fire_time >= cfg.fire_cooldown:
                    last_fire_time = now
                    # 発射方向: 手の向きのベクトル（smoothed_angle に基づく）
                    rad = math.radians(smoothed_angle)
                    # 速度はスケールに依存して見た目を合わせる
                    speed = cfg.projectile_speed * (scale if scale > 0 else 1.0)
                    vx = math.cos(rad) * speed
                    vy = math.sin(rad) * speed
                    # muzzle_global が None でないことを確認
                    mg = muzzle_global or (int(smoothed_pos[0]), int(smoothed_pos[1]))
                    projectiles.append(Projectile(mg, (vx, vy)))

            prev_finger_bent = finger_bent

            # デバッグ表示
            cv2.circle(frame, smoothed_pos, 5, (0,255,0), -1)
            cv2.putText(frame, f"Angle: {smoothed_angle:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(frame, f"PIP angle: {pip_angle:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            # 小さめにマズル位置も描く（デバッグ）
            if muzzle_global is not None:
                cv2.circle(frame, muzzle_global, 6, (0,140,255), -1)
        else:
            # 手なし -> そのまま
            pass

        # --- ターゲットの生成 ---
        if now - last_target_spawn_time > cfg.target_spawn_interval and len(targets) < cfg.max_targets:
            last_target_spawn_time = now
            rand_x = random.randint(cfg.target_radius_max, W - cfg.target_radius_max)
            rand_y = random.randint(cfg.target_radius_max, H - cfg.target_radius_max)
            rand_radius = random.randint(cfg.target_radius_min, cfg.target_radius_max)
            targets.append(Target(pos=np.array([rand_x, rand_y]), radius=rand_radius, spawn_time=now))

        # --- 当たり判定 ---
        surviving_targets = []
        projectiles_to_remove = set()
        for t in targets:
            is_hit = False
            for i, p in enumerate(projectiles):
                if i in projectiles_to_remove:
                    continue
                dist = np.linalg.norm(p.pos - t.pos)
                if dist < t.radius:
                    is_hit = True
                    score += 10
                    projectiles_to_remove.add(i)
                    hit_effects.append([t.pos.copy(), t.radius, time.time()])
                    break # 1つのターゲットに複数の弾が当たらないように
            if not is_hit:
                surviving_targets.append(t)
        targets = surviving_targets
        projectiles = [p for i, p in enumerate(projectiles) if i not in projectiles_to_remove]

        # --- ターゲット更新・描画 ---
        targets = update_and_draw_targets(frame, targets, cfg.target_lifetime)

        # --- プロジェクタ更新・描画 ---
        projectiles = update_and_draw_projectiles(frame, projectiles, dt, cfg.projectile_lifetime)

        # --- ヒットエフェクト描画 ---
        new_hit_effects = []
        for pos, radius, spawn_time in hit_effects:
            effect_age = now - spawn_time
            if effect_age < 0.3:
                new_hit_effects.append([pos, radius, spawn_time])
                alpha = 1.0 - (effect_age / 0.3)
                color = (0, 255, int(255 * alpha))
                cv2.circle(frame, tuple(pos.astype(int)), int(radius * (1.0 + effect_age*2)), color, 2)
        hit_effects = new_hit_effects

        # --- スコア表示 ---
        cv2.putText(frame, f"SCORE: {score}", (W - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Gun Follow & Shoot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
