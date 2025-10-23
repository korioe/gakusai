import cv2
import mediapipe as mp
import numpy as np
import random
import math

# --- MediaPipe Hand Landmarkerの初期化 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1  # JS版に合わせて手は1つだけ検出
)
mp_drawing = mp.solutions.drawing_utils

# --- カメラの設定とゲーム変数の初期化 ---

# ユーザーが希望した解像度
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# ウェブカメラの起動
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラ
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# プレイヤーの初期設定
player = {'x': CAM_WIDTH // 2, 'y': CAM_HEIGHT // 2, 'radius': 20}
# 色 (BGR形式)
PLAYER_COLOR = (255, 255, 0) # Cyan (シアン)
ENEMY_COLOR = (0, 0, 255)    # Red (赤)
TEXT_COLOR = (255, 255, 255) # White (白)

# ゲームロジック用の変数
enemies = []
score = 0
gameOver = False
frameCount = 0

# --- ★ JS版の当たり判定関数をPythonに移植 ---
def check_collision(player_circle, enemy_rect):
    # プレイヤー(円)と敵(四角形)の最も近い点の座標を見つける
    testX = player_circle['x']
    testY = player_circle['y']

    if player_circle['x'] < enemy_rect['x']:
        testX = enemy_rect['x']
    elif player_circle['x'] > enemy_rect['x'] + enemy_rect['width']:
        testX = enemy_rect['x'] + enemy_rect['width']
    
    if player_circle['y'] < enemy_rect['y']:
        testY = enemy_rect['y']
    elif player_circle['y'] > enemy_rect['y'] + enemy_rect['height']:
        testY = enemy_rect['y'] + enemy_rect['height']

    # 最も近い点とプレイヤーの中心との距離を計算
    distX = player_circle['x'] - testX
    distY = player_circle['y'] - testY
    distance = math.sqrt((distX * distX) + (distY * distY))

    # 距離がプレイヤーの半径より小さければ衝突
    return distance <= player_circle['radius']

# --- ★ JS版の敵生成関数をPythonに移植 ---
def spawn_enemy():
    size = 30
    x = random.randint(0, CAM_WIDTH - size)
    y = -size # 画面の上から
    speed = 2 + random.random() * 3 # スピードをランダムに
    enemies.append({'x': int(x), 'y': int(y), 'width': size, 'height': size, 'speed': speed})

# --- ★ JS版のゲームオーバー描画をPythonに移植 ---
def draw_game_over(frame, final_score):
    # 画面全体を半透明の黒で覆う
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (CAM_WIDTH, CAM_HEIGHT), (0, 0, 0), -1)
    # 0.7の透明度で合成
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, 'GAME OVER', (CAM_WIDTH // 2 - 200, CAM_HEIGHT // 2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, TEXT_COLOR, 3)
    cv2.putText(frame, f'Score: {final_score}', (CAM_WIDTH // 2 - 100, CAM_HEIGHT // 2 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOR, 2)
    cv2.putText(frame, 'Press [R] to Restart', (CAM_WIDTH // 2 - 120, CAM_HEIGHT // 2 + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
    
# --- メインのゲームループ ---
while cap.isOpened():
    # キー入力を受け付ける (5ms待機)
    key = cv2.waitKey(5) & 0xFF

    # 'q' が押されたら終了
    if key == ord('q'):
        break
    
    # ゲームオーバー時に 'r' が押されたらリセット
    if gameOver and key == ord('r'):
        player = {'x': CAM_WIDTH // 2, 'y': CAM_HEIGHT // 2, 'radius': 20}
        enemies = []
        score = 0
        gameOver = False
        frameCount = 0

    # カメラからフレームを読み込む
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # ★ JS版の (1 - landmark.x) と style.css の transform: scaleX(-1) を再現
    # 左右反転（鏡像）にする
    image = cv2.flip(image, 1)

    # --- ゲームキャンバスの作成 ---
    # JS版の <canvas> と同じ役割の、真っ黒な画像を作成
    game_canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype="uint8")

    # --- ゲームロジックの実行 (ゲームオーバーでない場合) ---
    if not gameOver:
        frameCount += 1

        # 60フレームごと（約1秒ごと）に敵を生成
        if frameCount % 60 == 0:
            spawn_enemy()

        # 敵の描画と更新
        # (リストを逆順に処理すると、ループ中の削除が安全に行える)
        for enemy in reversed(enemies):
            # 敵を動かす
            enemy['y'] += enemy['speed']
            
            # 敵を描画 (四角形)
            cv2.rectangle(game_canvas, (enemy['x'], int(enemy['y'])), 
                          (enemy['x'] + enemy['width'], int(enemy['y']) + enemy['height']), 
                          ENEMY_COLOR, -1) # -1は塗りつぶし

            # 当たり判定
            if check_collision(player, enemy):
                gameOver = True

            # 画面外に出た敵を削除
            if enemy['y'] > CAM_HEIGHT:
                enemies.remove(enemy)
                score += 1 # スコアを加算

        # --- 手の検出とプレイヤーの操作 ---
        # MediaPipeのために画像をBGRからRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            # 検出した手（1つ目）のランドマークを取得
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 手首（landmark 0）の位置を取得
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            # 座標をキャンバスサイズに合わせる (0.0~1.0 -> 0~CAM_WIDTH)
            player['x'] = int(wrist.x * CAM_WIDTH)
            player['y'] = int(wrist.y * CAM_HEIGHT)
            
            # デバッグ用に手の骨格をカメラ映像に描画
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # プレイヤー（円）を描画
        cv2.circle(game_canvas, (player['x'], player['y']), player['radius'], 
                   PLAYER_COLOR, -1) # -1は塗りつぶし

        # スコアの描画
        cv2.putText(game_canvas, f'Score: {score}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOR, 2)
    
    # --- ゲームオーバー処理 ---
    else:
        draw_game_over(game_canvas, score)


    # --- 画面の合成 ---
    # JS版の <video id="webcam"> のように、右上にカメラ映像を小さく表示
    
    # カメラ映像をリサイズ
    debug_cam_width = 200
    debug_cam_height = int(debug_cam_width * (CAM_HEIGHT / CAM_WIDTH)) # アスペクト比を維持
    
    debug_cam = cv2.resize(image, (debug_cam_width, debug_cam_height))
    
    # 右上の座標に貼り付け
    margin = 10
    game_canvas[margin : margin + debug_cam_height, 
                CAM_WIDTH - debug_cam_width - margin : CAM_WIDTH - margin] = debug_cam

    # --- ウィンドウに最終的な画像を表示 ---
    cv2.imshow('Webcam Shooting Game (Press Q to Quit)', game_canvas)


# --- 終了処理 ---
cap.release()
cv2.destroyAllWindows()