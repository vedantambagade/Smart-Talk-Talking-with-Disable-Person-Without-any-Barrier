from flask import Flask, render_template, jsonify
import threading
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import logging
from twilio.rest import Client
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===== Flask App =====
app = Flask(__name__)

# ===== Emotion Detection Setup =====
model = load_model("models/model_file_30epochs.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ===== Twilio Setup =====
account_sid = 'ACc1cf23b6c9da8c17336fc3f0ce5c1dc9'
auth_token = 'a3f75276a1786c52950b7ab655709993'
twilio_sms_number = '+16814164357'
your_verified_number = '+918999391122'
client = Client(account_sid, auth_token)

# ===== Flags =====
emotion_running = False
detection_enabled = False
camera_thread = None
message_sent = False
camera_on = True  # Camera always ON, black screen handled separately

# ===== Logging =====
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Emotion Detection =====
def run_emotion_detection():
    global emotion_running
    emotion_running = True
    cap = cv2.VideoCapture(0)
    while emotion_running:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            pred = model.predict(face_roi)
            emotion = emotion_labels[int(np.argmax(pred))]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    emotion_running = False

# ===== Person Detection =====
def run_person_detection():
    global detection_enabled, message_sent, camera_on
    cap = cv2.VideoCapture(0)
    face_cascade_fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    no_face_start_time = None
    countdown_duration = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if camera_on:
            if detection_enabled:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade_fd.detectMultiScale(gray, 1.1, 10, minSize=(100,100))
                if len(faces) > 0:
                    no_face_start_time = None
                    message_sent = False
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    if no_face_start_time is None:
                        no_face_start_time = time.time()
                    else:
                        elapsed = time.time() - no_face_start_time
                        if elapsed >= 2:
                            countdown_elapsed = elapsed - 2
                            remaining = countdown_duration - int(countdown_elapsed)
                            if remaining > 0:
                                cv2.putText(frame, f"No Face Detected. Countdown: {remaining}",
                                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            else:
                                cv2.putText(frame, "Countdown Over", (20,50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                if not message_sent:
                                    try:
                                        client.messages.create(
                                            body="⚠ Alert: No face detected for 30 seconds!",
                                            from_=twilio_sms_number,
                                            to=your_verified_number
                                        )
                                        logger.info("📩 Alert sent via SMS!")
                                        message_sent = True
                                    except Exception as e:
                                        logger.error(f"❌ Error sending SMS: {e}")
            else:
                # BLACK SCREEN MODE
                frame[:] = 0

            cv2.imshow("Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ===== Flask Routes =====
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/Register')
def Register():
    return render_template('Register.html')

@app.route('/predict_emotion', methods=['POST'])
def start_emotion():
    global emotion_running
    if not emotion_running:
        threading.Thread(target=run_emotion_detection, daemon=True).start()
    return jsonify({'status': 'Emotion detection started'})

@app.route('/detect_person', methods=['POST'])
def start_person():
    global detection_enabled, camera_thread
    detection_enabled = True
    if camera_thread is None or not camera_thread.is_alive():
        camera_thread = threading.Thread(target=run_person_detection, daemon=True)
        camera_thread.start()
    return jsonify({'status': 'Person detection ON (Flask)'})

@app.route('/stop_person', methods=['POST'])
def stop_person():
    global detection_enabled
    detection_enabled = False
    return jsonify({'status': 'Person detection OFF (Flask)'})

# ===== Telegram Bot =====
TELEGRAM_TOKEN = "8288819640:AAFIOEqAU_N3fQgtcrhhqDyDIU8TYuusfxI"

async def telegram_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global detection_enabled
    detection_enabled = True
    await update.message.reply_text("📷 Person Detection Started via Bot!")
    logger.info("✅ Person Detection started via Telegram")

async def telegram_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global detection_enabled
    detection_enabled = False
    await update.message.reply_text("🛑 Person Detection Stopped via Bot (Black Screen Mode)")
    logger.info("❌ Person Detection stopped via Telegram")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Welcome to Emotion Detection Bot!\n\n"
        "Available Commands:\n"
        "/on - Start Person Detection\n"
        "/off - Stop Person Detection\n"
        "/status - Check current status"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_text = "✅ ON" if detection_enabled else "❌ OFF"
    await update.message.reply_text(f"🔍 Person Detection Status: {status_text}")

def run_telegram_bot():
    """Run Telegram bot in a separate thread"""
    def start_bot():
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Build application
            application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            
            # Add handlers
            application.add_handler(CommandHandler("start", start))
            application.add_handler(CommandHandler("on", telegram_on))
            application.add_handler(CommandHandler("off", telegram_off))
            application.add_handler(CommandHandler("status", status))
            
            # Start the bot
            logger.info("🤖 Starting Telegram Bot...")
            application.run_polling()
            
        except Exception as e:
            logger.error(f"❌ Telegram Bot Error: {e}")

    # Start bot in a separate thread
    bot_thread = threading.Thread(target=start_bot, daemon=True)
    bot_thread.start()
    logger.info("✅ Telegram Bot thread started")

# ===== Initialize Telegram Bot =====
run_telegram_bot()

# ===== Run Flask =====
if __name__ == "__main__":
    logger.info("🚀 Starting Flask Application...")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)