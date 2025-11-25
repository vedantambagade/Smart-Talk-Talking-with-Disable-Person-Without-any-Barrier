import cv2
import time
import threading
import logging
import asyncio
from twilio.rest import Client
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ========== LOGGING ==========
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)  
logger = logging.getLogger(__name__)

# ========== TWILIO SETUP (SMS) ==========
account_sid = 'ACc1cf23b6c9da8c17336fc3f0ce5c1dc9'
auth_token = 'a3f75276a1786c52950b7ab655709993'
twilio_sms_number = '+16814164357'
your_verified_number = '+918999391122'
client = Client(account_sid, auth_token)

# ========== GLOBAL FLAGS ==========
running = False
message_sent = False

# ========== FACE DETECTION FUNCTION ==========
def face_detection_loop():
    global running, message_sent

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("❌ Error: Camera not detected.")
        running = False
        return

    no_face_start_time = None
    countdown_duration = 30

    while running:
        ret, frame = cap.read()
        if not ret:
            logger.error("❌ Error: Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100)
        )

        if len(faces) > 0:
            no_face_start_time = None
            message_sent = False
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
        else:
            if no_face_start_time is None:
                no_face_start_time = time.time()
            else:
                elapsed = time.time() - no_face_start_time
                if elapsed >= 2:
                    countdown_elapsed = elapsed - 2
                    remaining = countdown_duration - int(countdown_elapsed)
                    if remaining > 0:
                        cv2.putText(
                            frame,
                            f"No Face Detected. Countdown: {remaining}",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2
                        )
                    else:
                        cv2.putText(
                            frame,
                            "Countdown Over",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2
                        )
                        if not message_sent:
                            alert_message = "⚠️ Alert: No face detected for 30 seconds!"
                            try:
                                client.messages.create(
                                    body=alert_message,
                                    from_=twilio_sms_number,
                                    to=your_verified_number
                                )
                                logger.info("📩 Alert sent via SMS!")
                                message_sent = True
                            except Exception as e:
                                logger.error(f"❌ Error sending SMS: {e}")

        cv2.imshow("Face Detection with SMS Alert", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# ========== TELEGRAM COMMANDS ==========
async def start_camera(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global running
    if not running:
        running = True
        threading.Thread(target=face_detection_loop, daemon=True).start()
        await update.message.reply_text("📷 Camera Started")
        logger.info("✅ Camera started by Telegram command.")
    else:
        await update.message.reply_text("⚠️ Camera Already Running")

async def stop_camera(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global running
    if running:
        running = False
        await update.message.reply_text("🛑 Camera Stopped")
        logger.info("✅ Camera stopped by Telegram command.")
    else:
        await update.message.reply_text("⚠️ Camera Already Stopped")

# ========== TELEGRAM BOT ==========
TELEGRAM_TOKEN = "8288819640:AAFIOEqAU_N3fQgtcrhhqDyDIU8TYuusfxI"
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("on", start_camera))
app.add_handler(CommandHandler("off", stop_camera))

# ========== MAIN ==========
if __name__ == "__main__":
    logger.info("🚀 Bot is starting...")
    asyncio.run(app.run_polling())
