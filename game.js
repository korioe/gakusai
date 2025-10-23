import { HandLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm/vision_bundle.js';

// --- DOM要素の取得 ---
document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');

    let handLandmarker;
    let lastVideoTime = -1;
    let player = { x: canvas.width / 2, y: canvas.height / 2, radius: 20 };

    // --- MediaPipe Hand Landmarkerの初期化 ---
    async function createHandLandmarker() {
        const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm');
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 1
        });
        console.log("HandLandmarker initialized");
        startWebcam();
    }

    // --- ウェブカメラの開始 ---
    function startWebcam() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, frameRate: { ideal: 30, max: 30 } } })
                .then(stream => {
                    video.srcObject = stream;
                    video.addEventListener('loadeddata', predictWebcam);
                })
                .catch(err => {
                    console.error("Error accessing webcam: ", err);
                    alert("ウェブカメラにアクセスできませんでした。許可を確認してください。");
                });
        } else {
            alert("お使いのブラウザはウェブカメラをサポートしていません。");
        }
    }

    // --- 手の検出とゲームループ ---
    function predictWebcam() {
        const nowInMs = Date.now();
        if (video.currentTime !== lastVideoTime) {
            lastVideoTime = video.currentTime;
            const results = handLandmarker.detectForVideo(video, nowInMs);

            // 描画処理
            drawGame(results);
        }
        // 次のフレームをリクエスト
        window.requestAnimationFrame(predictWebcam);
    }

    // --- ゲーム画面の描画 ---
    function drawGame(results) {
        // キャンバスをクリア
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0];
            // 手首（landmark 0）の位置をプレイヤーの座標に反映
            // 座標をキャンバスサイズに合わせ、左右反転させる
            const landmark = landmarks[0]; // 手首の座標を使用
            player.x = (1 - landmark.x) * canvas.width;
            player.y = landmark.y * canvas.height;

            // プレイヤー（円）を描画
            ctx.fillStyle = 'cyan';
            ctx.beginPath();
            ctx.arc(player.x, player.y, player.radius, 0, 2 * Math.PI);
            ctx.fill();

            // デバッグ用にランドマークを描画
            drawLandmarks(landmarks);
        }
    }
    
    // --- ランドマークを線で結んで描画する関数 ---
    function drawLandmarks(landmarks) {
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;

        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], // 親指
            [0, 5], [5, 6], [6, 7], [7, 8], // 人差し指
            [5, 9], [9, 10], [10, 11], [11, 12], // 中指
            [9, 13], [13, 14], [14, 15], [15, 16], // 薬指
            [0, 17], [17, 18], [18, 19], [19, 20], // 小指
            [13, 17]
        ];

        connections.forEach(pair => {
            const start = landmarks[pair[0]];
            const end = landmarks[pair[1]];
            ctx.beginPath();
            ctx.moveTo((1 - start.x) * canvas.width, start.y * canvas.height);
            ctx.lineTo((1 - end.x) * canvas.width, end.y * canvas.height);
            ctx.stroke();
        });

        landmarks.forEach(point => {
            ctx.fillStyle = 'red';
            ctx.beginPath();
            ctx.arc((1 - point.x) * canvas.width, point.y * canvas.height, 5, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    // --- 初期化の実行 ---
    createHandLandmarker();
});
