# bird_shooter_game.py
import cv2
import mediapipe as mp
import numpy as np
import math, random
import time
import pygame.mixer
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Settings:
    """アプリケーションの設定を管理するデータクラス"""
    gun_path: str = r"C:\kaihatu\gakusaigame\leftgungun.png"
    right_gun_path: str = r"C:\kaihatu\gakusaigame\dotright.png"
    bird_path: str = r"C:\kaihatu\gakusaigame\bird.png"
    shoot_sound_path: str = r"C:\kaihatu\shoot5.ogg"
    bgm_path: str = r"C:\kaihatu\heat_storm.ogg"
    explosion_sound_path: str = r"C:\kaihatu\explosion12.ogg"
    cam_id: int = 0
    max_hands: int = 1
    window_width: int = 1280
    window_height: int = 720

    # スムージング係数
    smooth_alpha_pos: float = 0.35
    smooth_alpha_angle: float = 0.45

    # 銃の表示設定
    gun_scale: float = 0.8
    muzzle_ratio: Tuple[float, float] = (0.10, 0.50)
    grip_ratio: Tuple[float, float] = (0.80, 0.70)

    # 発射関連
    projectile_speed: float = 1500.0
    projectile_lifetime: float = 2.0
    fire_cooldown: float = 0.3  # 弾を発射できる間隔（秒）。大きいほど連射速度が遅くなる
    bgm_volume: float = 0.4 # BGMの音量 (0.0から1.0)

    # 指の判定
    finger_bend_angle_deg: float = 80.0

    # 敵（鳥）関連
    big_bird_path: str = r"C:\kaihatu\gakusaigame\bigbird.png"
    big_bird_hp: int = 3
    big_bird_spawn_chance: float = 0.2  # 20%の確率でビッグバードが出現
    big_bird_score: int = 50
    bone_bird_path: str = r"C:\kaihatu\gakusaigame\bonebird.png"
    bone_bird_hp: int = 2
    bone_bird_spawn_chance: float = 0.15 # 15%の確率でボーンバードが出現
    bone_bird_score: int = 30
    bone_bird_amplitude: float = 50.0  # 上下運動の振幅
    bone_bird_frequency: float = 4.0   # 上下運動の速さ
    normal_bird_score: int = 10
    bird_spawn_interval: float = 1.0  # 敵が出現する間隔（秒）
    initial_max_birds: int = 3         # ゲーム開始時の最大敵数
    max_birds_limit: int = 10          # 敵の最大数（上限）
    bird_increase_interval: float = 15.0 # 敵が増える間隔（秒）
    bird_speed_min: float = 100.0      # 敵の最低速度
    bird_speed_max: float = 250.0      # 敵の最高速度
    bird_size_min: int = 80          # 敵の最小サイズ
    bird_size_max: int = 130         # 敵の最大サイズ

    game_duration_seconds: float = 60.0 # プレイ時間

    # MediaPipe設定
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6

# --- MediaPipe Hands 初期化 ---
mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

def load_rgba(path):
    """RGBA画像を読み込む"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {path}")
    if img.shape[2] == 3:
        b,g,r = cv2.split(img)
        a = np.ones(b.shape, dtype=b.dtype) * 255
        img = cv2.merge((b,g,r,a))
    return img

def overlay_image(bg: np.ndarray, fg_rgba: np.ndarray, pos: Tuple[int, int]):
    """背景に前景画像を重ねる（回転なし）"""
    x, y = pos
    h, w = fg_rgba.shape[:2]
    
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    H, W = bg.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x2), min(H, y2)

    if sx1 >= sx2 or sy1 >= sy2:
        return bg

    fx1 = sx1 - x1
    fy1 = sy1 - y1
    fx2 = fx1 + (sx2 - sx1)
    fy2 = fy1 + (sy2 - sy1)

    fg_crop = fg_rgba[fy1:fy2, fx1:fx2]
    bg_crop = bg[sy1:sy2, sx1:sx2]

    fg_bgr = fg_crop[..., :3]
    fg_a = fg_crop[..., 3:].astype(float) / 255.0

    out_part = bg_crop.astype(float) * (1.0 - fg_a) + fg_bgr.astype(float) * fg_a
    bg[sy1:sy2, sx1:sx2] = out_part.astype(np.uint8)
    return bg


def transform_and_overlay_with_muzzle(bg, fg_rgba, center, angle_deg, scale, muzzle_local_ratio=(0.9,0.5), rotation_center_ratio=None):
    h, w = fg_rgba.shape[:2]
    rot_center_x = (w / 2) if rotation_center_ratio is None else rotation_center_ratio[0] * w
    rot_center_y = (h / 2) if rotation_center_ratio is None else rotation_center_ratio[1] * h
    M = cv2.getRotationMatrix2D((rot_center_x, rot_center_y), angle_deg, scale)

    # 元の画像のサイズで回転。はみ出る部分はOpenCVによって切り取られる
    rotated = cv2.warpAffine(fg_rgba, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    cx, cy = center
    # 回転の中心が指定の `center` に来るように描画位置を計算
    x1 = int(cx - rot_center_x)
    y1 = int(cy - rot_center_y)
    
    bg = overlay_image(bg, rotated, (x1, y1))

    # 銃口位置の計算は、回転行列Mをそのまま使えばOK
    mlx = muzzle_local_ratio[0] * w
    mly = muzzle_local_ratio[1] * h
    tx = M[0,0]*mlx + M[0,1]*mly + M[0,2]
    ty = M[1,0]*mlx + M[1,1]*mly + M[1,2]
    muzzle_global = (int(x1 + tx), int(y1 + ty))

    return bg, muzzle_global

def angle_from_points(p_from, p_to):
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return math.degrees(math.atan2(dy, dx))

def smooth_angle(prev_angle: float, new_angle: float, alpha: float) -> float:
    diff = (new_angle - prev_angle + 180) % 360 - 180
    return prev_angle + diff * alpha

def finger_pip_angle_deg(hand_landmarks, pip_landmark, mcp_landmark, tip_landmark) -> float:
    pip = hand_landmarks[pip_landmark]
    mcp = hand_landmarks[mcp_landmark]
    tip = hand_landmarks[tip_landmark]
    v1 = np.array([mcp.x - pip.x, mcp.y - pip.y])
    v2 = np.array([tip.x - pip.x, tip.y - pip.y])
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-8 or norm_v2 < 1e-8: return 180.0
    dot = np.clip(np.dot(v1 / norm_v1, v2 / norm_v2), -1.0, 1.0)
    return math.degrees(math.acos(dot))

class Projectile:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.spawn_time = time.time()
    def age(self): return time.time() - self.spawn_time

@dataclass
class Bird:
    pos: np.ndarray
    vel: np.ndarray
    size: Tuple[int, int]
    image: np.ndarray
    hp: int
    max_hp: int
    score: int
    spawn_time: float
    bird_type: str = 'normal'
    hit_timer: int = 0  # ヒットエフェクト用タイマー
    initial_y: float = 0.0
    amplitude: float = 0.0
    frequency: float = 0.0

    def age(self): return time.time() - self.spawn_time

    def get_rect(self):
        return (self.pos[0], self.pos[1], self.size[0], self.size[1])

@dataclass
class Explosion:
    """爆発エフェクトを管理するクラス"""
    pos: np.ndarray
    spawn_time: float
    duration: float = 0.6
    num_particles: int = 20
    max_radius: float = 100.0
    particles: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]] = None

    def __post_init__(self):
        """パーティクルの初期化"""
        self.particles = []
        for _ in range(self.num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.4, 1.2) * self.max_radius / self.duration
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            color = random.choice([(0, 165, 255), (0, 215, 255), (255, 255, 255)])
            self.particles.append([self.pos.copy(), velocity, color])

    def age(self): return time.time() - self.spawn_time
    def is_alive(self): return self.age() < self.duration

    def update_and_draw(self, frame: np.ndarray, dt: float):
        if not self.is_alive(): return
        alpha = max(0, 1.0 - self.age() / self.duration)
        for p in self.particles:
            p[0] += p[1] * dt
            cv2.circle(frame, tuple(p[0].astype(int)), int(10 * alpha), p[2], -1)

def update_and_draw_projectiles(frame: np.ndarray, projectiles: List[Projectile], dt: float, lifetime: float):
    surviving = []
    H, W = frame.shape[:2]
    for p in projectiles:
        p.pos += p.vel * dt
        if p.age() <= lifetime and -50 < p.pos[0] < W + 50 and -50 < p.pos[1] < H + 50:
            surviving.append(p)
            fade = max(0.0, 1.0 - p.age() / lifetime)
            col = (0, 180, 255)
            cv2.circle(frame, tuple(p.pos.astype(int)), 10, col, -1)
            tail = p.pos - p.vel * 0.03
            cv2.line(frame, tuple(p.pos.astype(int)), tuple(tail.astype(int)), col, int(10 + 12 * fade))
    return surviving

def update_and_draw_birds(frame: np.ndarray, birds: List[Bird], dt: float):
    surviving = []
    H, W = frame.shape[:2]
    for bird in birds:
        bird.pos += bird.vel * dt
        if bird.pos[0] < W:
            # ボーンバードの上下運動
            if bird.bird_type == 'bone':
                age = bird.age()
                bird.pos[1] = bird.initial_y + bird.amplitude * math.sin(age * bird.frequency)

            if bird.hit_timer > 0:
                bird.hit_timer -= 1

            surviving.append(bird)
            
            # ヒットタイマーがアクティブなら画像を白くする
            if bird.hit_timer > 0:
                # 元の画像に白をオーバーレイして点滅させる
                overlay = np.full_like(bird.image, (255, 255, 255, 255), dtype=np.uint8)
                alpha = 0.7 * (bird.hit_timer / 5.0) # 5フレームでフェードアウト
                temp_image = cv2.addWeighted(overlay, alpha, bird.image, 1 - alpha, 0)
                overlay_image(frame, temp_image, tuple(bird.pos.astype(int)))
            else:
                overlay_image(frame, bird.image, tuple(bird.pos.astype(int)))

            # HPバー描画
            if bird.hp < bird.max_hp:
                hp_ratio = bird.hp / bird.max_hp
                bar_width = bird.size[0]
                bar_x = int(bird.pos[0])
                bar_y = int(bird.pos[1] - 8)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * hp_ratio), bar_y + 5), (0, 255, 0), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 5), (128, 128, 128), 1)
    return surviving

def main():
    cfg = Settings()
    
    # Pygameミキサーの初期化とサウンドの読み込み
    try:
        pygame.mixer.init()
        shoot_sound = pygame.mixer.Sound(cfg.shoot_sound_path)
        explosion_sound = pygame.mixer.Sound(cfg.explosion_sound_path)
        pygame.mixer.music.load(cfg.bgm_path)
        pygame.mixer.music.set_volume(cfg.bgm_volume)
        bgm_loaded = True
    except Exception as e:
        print(f"サウンドの読み込みに失敗しました: {e}")
        shoot_sound = None
        explosion_sound = None
        bgm_loaded = False

    left_gun_rgba = load_rgba(cfg.gun_path)
    right_gun_rgba = load_rgba(cfg.right_gun_path)
    bird_template_rgba = load_rgba(cfg.bird_path)
    big_bird_template_rgba = load_rgba(cfg.big_bird_path)
    bone_bird_template_rgba = load_rgba(cfg.bone_bird_path)

    left_muzzle = cfg.muzzle_ratio
    left_grip = cfg.grip_ratio
    right_muzzle = (1.0 - cfg.muzzle_ratio[0], cfg.muzzle_ratio[1])
    right_grip = (1.0 - cfg.grip_ratio[0], cfg.grip_ratio[1])

    hands = mp_hands.Hands(
        max_num_hands=cfg.max_hands,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )
    cap = cv2.VideoCapture(cfg.cam_id)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.window_height)

    projectiles: List[Projectile] = []
    birds: List[Bird] = []
    explosions: List[Explosion] = []
    score = 0
    smoothed_pos, smoothed_angle = None, None
    prev_finger_bent = False
    last_fire_time = 0.0
    last_bird_spawn_time = 0.0

    game_state = "waiting"
    game_start_time = 0
    countdown_start_time = 0
    game_over_reason = ""

    prev_time = time.time()
    print("起動: qキーで終了")

    while True:
        now = time.time()
        dt = now - prev_time
        prev_time = now

        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if game_state == "waiting":
            text = "Press SPACE to Start"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            text_x = (W - text_size[0]) // 2
            text_y = (H + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            if key == ord(' '):
                game_state = "countdown"
                countdown_start_time = now

        elif game_state == "countdown":
            countdown_elapsed = now - countdown_start_time
            countdown_sec = 3 - int(countdown_elapsed)
            if countdown_sec > 0:
                text = str(countdown_sec)
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
                text_x = (W - text_size[0]) // 2
                text_y = (H + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10, cv2.LINE_AA)
            else:
                game_state = "playing"
                game_start_time = now
                if bgm_loaded:
                    pygame.mixer.music.play(-1) # -1でループ再生
        
        elif game_state == "playing":
            elapsed_time = now - game_start_time
            time_left = max(0, cfg.game_duration_seconds - elapsed_time)

            if time_left <= 0:
                game_state = "game_over"
                game_over_reason = "TIME UP"
                if bgm_loaded:
                    pygame.mixer.music.stop()

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
                index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                wrist_px = (int(wrist.x * W), int(wrist.y * H))
                index_px = (int(index_tip.x * W), int(index_tip.y * H))
                
                mcp_landmarks = [hand.landmark[lm_id] for lm_id in [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]]
                palm_center_norm = (np.mean([lm.x for lm in mcp_landmarks] + [wrist.x]), np.mean([lm.y for lm in mcp_landmarks] + [wrist.y]))
                palm_center_px = (int(palm_center_norm[0] * W), int(palm_center_norm[1] * H))

                center_pos = palm_center_px
                angle = angle_from_points(wrist_px, index_px)
                base_dist = math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)
                scale = (base_dist * 6.0) * cfg.gun_scale

                if smoothed_pos is None:
                    smoothed_pos, smoothed_angle = center_pos, angle
                else:
                    smoothed_pos = (int(smoothed_pos[0] * (1 - cfg.smooth_alpha_pos) + center_pos[0] * cfg.smooth_alpha_pos), int(smoothed_pos[1] * (1 - cfg.smooth_alpha_pos) + center_pos[1] * cfg.smooth_alpha_pos))
                    smoothed_angle = smooth_angle(smoothed_angle, angle, cfg.smooth_alpha_angle)

                facing_right = math.cos(math.radians(smoothed_angle)) > 0.0
                gun_rgba, muzzle_ratio, rot_center_ratio, draw_angle = (right_gun_rgba, right_muzzle, right_grip, -smoothed_angle) if facing_right else (left_gun_rgba, left_muzzle, left_grip, -smoothed_angle + 180)
                
                frame_copy = frame.copy()
                frame, muzzle_global = transform_and_overlay_with_muzzle(frame_copy, gun_rgba, smoothed_pos, draw_angle, scale, muzzle_local_ratio=muzzle_ratio, rotation_center_ratio=rot_center_ratio)

                pip_angle = finger_pip_angle_deg(hand.landmark, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_TIP)
                finger_bent = pip_angle < cfg.finger_bend_angle_deg

                if not prev_finger_bent and finger_bent and (now - last_fire_time >= cfg.fire_cooldown):
                    last_fire_time = now
                    rad = math.radians(smoothed_angle)
                    speed = cfg.projectile_speed * (scale if scale > 0 else 1.0)
                    vel = (math.cos(rad) * speed, math.sin(rad) * speed)
                    projectiles.append(Projectile(muzzle_global or smoothed_pos, vel))
                    if shoot_sound:
                        shoot_sound.play()
                
                prev_finger_bent = finger_bent

            level = int(elapsed_time // cfg.bird_increase_interval) + 1
            current_max_birds = min(cfg.initial_max_birds + level - 1, cfg.max_birds_limit)

            if now - last_bird_spawn_time > cfg.bird_spawn_interval and len(birds) < current_max_birds:
                last_bird_spawn_time = now
                
                spawn_roll = random.random()
                bird_type = 'normal'
                if spawn_roll < cfg.bone_bird_spawn_chance:
                    bird_type = 'bone'
                elif spawn_roll < cfg.bone_bird_spawn_chance + cfg.big_bird_spawn_chance:
                    bird_type = 'big'
                
                start_y = random.randint(int(cfg.bone_bird_amplitude), int(H - cfg.bone_bird_amplitude - cfg.bird_size_max))
                
                if bird_type == 'big':
                    template_img = big_bird_template_rgba
                    hp = cfg.big_bird_hp
                    score_val = cfg.big_bird_score
                    size = random.randint(cfg.bird_size_min + 20, cfg.bird_size_max + 20)
                    amplitude, frequency = 0, 0
                elif bird_type == 'bone':
                    template_img = bone_bird_template_rgba
                    hp = cfg.bone_bird_hp
                    score_val = cfg.bone_bird_score
                    size = random.randint(cfg.bird_size_min, cfg.bird_size_max)
                    amplitude = cfg.bone_bird_amplitude
                    frequency = cfg.bone_bird_frequency
                else: # normal
                    template_img = bird_template_rgba
                    hp = 1
                    score_val = cfg.normal_bird_score
                    size = random.randint(cfg.bird_size_min, cfg.bird_size_max)
                    amplitude, frequency = 0, 0

                aspect_ratio = template_img.shape[1] / template_img.shape[0]
                new_width = int(size * aspect_ratio)
                new_height = size
                resized_bird_img = cv2.resize(template_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                start_x = -new_width - 10
                speed = random.uniform(cfg.bird_speed_min, cfg.bird_speed_max)
                birds.append(Bird(pos=np.array([start_x, start_y], dtype=float), 
                                  vel=np.array([speed, 0], dtype=float), 
                                  size=(new_width, new_height), 
                                  image=resized_bird_img, 
                                  hp=hp, max_hp=hp, score=score_val,
                                  spawn_time=now,
                                  bird_type=bird_type,
                                  initial_y=float(start_y),
                                  amplitude=amplitude,
                                  frequency=frequency))

            surviving_birds = []
            projectiles_to_remove = set()
            for bird in birds:
                is_hit = False
                bird_rect = bird.get_rect()
                bx, by, bw, bh = bird_rect
                for i, p in enumerate(projectiles):
                    if i in projectiles_to_remove: continue
                    px, py = p.pos
                    if bx < px < bx + bw and by < py < by + bh:
                        bird.hp -= 1
                        bird.hit_timer = 5 # ヒットエフェクト
                        projectiles_to_remove.add(i)
                        if bird.hp <= 0:
                            is_hit = True # HPが0になったら倒した判定
                            score += bird.score
                            bird_center = bird.pos + np.array([bw / 2, bh / 2])
                            explosions.append(Explosion(pos=bird_center, spawn_time=now))
                            if explosion_sound:
                                explosion_sound.play()
                            break # この鳥は倒したので、他の弾との判定は不要
                        # HPが残っている場合は、他の弾が当たらないようにループを抜けない
                if not is_hit:
                    surviving_birds.append(bird)
                    if bx >= W:
                        game_state = "game_over"
                        game_over_reason = "ENEMY REACHED"
                        if bgm_loaded:
                            pygame.mixer.music.stop()
            birds = surviving_birds
            projectiles = [p for i, p in enumerate(projectiles) if i not in projectiles_to_remove]

            birds = update_and_draw_birds(frame, birds, dt)
            projectiles = update_and_draw_projectiles(frame, projectiles, dt, cfg.projectile_lifetime)

            # 爆発エフェクトの更新と描画
            surviving_explosions = []
            for exp in explosions:
                if exp.is_alive():
                    exp.update_and_draw(frame, dt)
                    surviving_explosions.append(exp)
            explosions = surviving_explosions

            cv2.putText(frame, f"SCORE: {score}", (W - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"TIME: {int(time_left)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"LEVEL: {level}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        elif game_state == "game_over":
            text = f"GAME OVER: {game_over_reason}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            text_x = (W - text_size[0]) // 2
            text_y = H // 2 - 80
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            
            score_text = f"FINAL SCORE: {score}"
            score_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            score_x = (W - score_size[0]) // 2
            cv2.putText(frame, score_text, (score_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv2.LINE_AA)

            restart_text = "Press 'R' to Restart"
            restart_size, _ = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            restart_x = (W - restart_size[0]) // 2
            cv2.putText(frame, restart_text, (restart_x, text_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if key == ord('r'):
                # ゲームの状態をリセット
                projectiles.clear()
                birds.clear()
                explosions.clear()
                score = 0
                smoothed_pos, smoothed_angle = None, None
                game_state = "waiting"
                if bgm_loaded:
                    pygame.mixer.music.stop()

        cv2.imshow("Bird Shooter", frame)

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
