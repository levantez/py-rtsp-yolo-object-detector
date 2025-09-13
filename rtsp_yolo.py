import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR" # если нужно отключить спам в логи сообщения NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
import cv2
import time
import logging
import multiprocessing as mp
from ultralytics import YOLO
import numpy as np
from telegram import Bot
from io import BytesIO
import asyncio
import sys

# ==================== КОНФИГУРАЦИЯ ====================
RTSP_STREAMS = [
    "rtsp://login:pass@192.168.01.123:554/cam1",  # Замени на свои
    "rtsp://login:pass@192.168.01.123:554/cam2",
]

TELEGRAM_TOKEN = "<TOKEN>"          # Получи от @BotFather
TELEGRAM_CHAT_ID = "12345"          # Получи через @getmyid_bot

# Путь к модели YOLOv8n (если нет — скачается автоматически)
MODEL_PATH = "yolov8n.pt"

# Порог уверенности и частота обработки
CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 10  # Обрабатывать каждый 10-й кадр → ~2–3 FPS при 25–30 FPS потока
TARGET_SIZE = (640, 480)  # Оптимальный размер для YOLOv8n

# Классы YOLOv8n (COCO), нас интересуют только эти:
CLASS_NAMES = {
    0: "person",
    14: "dog",
    15: "cat"
    # 1: "bicycle",
    # 2: "car",
    # 3: "motorcycle",
    # 5: "bus",
    # 7: "truck"
}

import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== ОЧЕРЕДЬ ДЛЯ СООБЩЕНИЙ ====================
message_queue = mp.Queue()

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def draw_annotations(frame, detections):
    for det in detections:
        x_min, y_min, x_max, y_max = map(int, det['bbox'])
        label = f"{det['class']} ({det['confidence']:.2f})"
        color = (0, 255, 0)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def send_to_telegram(bot, chat_id, frame, camera_name, detections, queue):
    annotated_frame = draw_annotations(frame.copy(), detections)

    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    photo_bytes = buffer.tobytes()

    if len(photo_bytes) == 0:
        logger.error(f"Невозможно закодировать изображение для {camera_name}")
        return

    obj_counts = {}
    for det in detections:
        obj_counts[det['class']] = obj_counts.get(det['class'], 0) + 1

    summary = f"📸 Обнаружение на камере: {camera_name}\n"
    summary += "🔍 Объекты:\n"
    for cls, count in obj_counts.items():
        summary += f"  - {cls}: {count}\n"

    queue.put({
        'photo_bytes': photo_bytes,
        'caption': summary,
        'camera_name': camera_name,
        'photo_size_kb': len(photo_bytes) / 1024
    })
    logger.info(f"Отправлено в очередь: {camera_name} | Размер: {len(photo_bytes)/1024:.1f} KB")

# ==================== ОБРАБОТЧИК КАМЕРЫ ====================

def process_camera(camera_index, rtsp_url, model_path, message_queue):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть RTSP: {rtsp_url}")
        return

    frame_count = 0
    last_send_time = 0
    send_interval = 5

    logger.info(f"Процесс {camera_index} запущен для {rtsp_url}")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Потеря кадра в {rtsp_url}, переподключение...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        resized_frame = cv2.resize(frame, TARGET_SIZE)
        results = model(resized_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            if cls_id not in CLASS_NAMES:
                continue
            confidence = float(box.conf.item())
            x_min, y_min, x_max, y_max = map(float, box.xyxy[0])

            h, w = frame.shape[:2]
            x_min = int(x_min * w / TARGET_SIZE[0])
            y_min = int(y_min * h / TARGET_SIZE[1])
            x_max = int(x_max * w / TARGET_SIZE[0])
            y_max = int(y_max * h / TARGET_SIZE[1])

            detections.append({
                'class': CLASS_NAMES[cls_id],
                'confidence': confidence,
                'bbox': (x_min, y_min, x_max, y_max)
            })

        if detections and (time.time() - last_send_time > send_interval):
            camera_name = f"Камера {camera_index + 1}"
            send_to_telegram(None, None, frame, camera_name, detections, message_queue)  # ← ПЕРЕДАЁМ ОЧЕРЕДЬ!
            last_send_time = time.time()

        time.sleep(0.1)

    cap.release()

# ==================== АСИНХРОННЫЙ ОТПРАВИТЕЛЬ В TELEGRAM ====================

async def telegram_sender(bot_token, chat_id, queue):
    logger.info("Запуск асинхронного Telegram отправителя...")

    bot = Bot(
        token=bot_token
    )

    # Словарь времени последней отправки по камере
    last_sent_time = {}  # ключ: имя камеры, значение: timestamp (time.time())

    try:
        me = await bot.get_me()
        logger.info(f"Успешно подключились к боту: @{me.username} (ID: {me.id})")
    except Exception as e:
        logger.critical(f"НЕВОЗМОЖНО подключиться к Telegram API: {type(e).__name__}: {e}")
        return

    try:
        test_msg = await bot.send_message(chat_id=chat_id, text="🤖 Тестовое сообщение от системы мониторинга. Если ты это читаешь — чат ID корректен.")
        logger.info(f"Чат ID {chat_id} доступен. Сообщение отправлено (ID: {test_msg.message_id})")
    except Exception as e:
        logger.critical(f"Бот НЕ МОЖЕТ отправлять сообщения в чат {chat_id}: {type(e).__name__}: {e}")
        if "Forbidden" in str(e):
            logger.critical("Напиши боту в Telegram: /start")
        elif "Chat not found" in str(e):
            logger.critical("Проверь chat_id — он должен быть числом, например: -1001234567890")
        return

    logger.info("Telegram sender готов к работе. Жду сообщения из очереди...")

    while True:
        try:
            msg = queue.get(timeout=3)
            photo_bytes = msg['photo_bytes']
            caption = msg['caption']
            camera_name = msg['camera_name']
            size_kb = msg['photo_size_kb']

            logger.info(f"Отправка в Telegram: {camera_name} | Размер: {size_kb:.1f} KB")

            if not photo_bytes or len(photo_bytes) == 0:
                logger.error(f"Пустой байтовый массив для {camera_name}")
                continue

            # Не чаще 1 раза в 5 сек на камеру
            now = time.time()
            if camera_name in last_sent_time:
                elapsed = now - last_sent_time[camera_name]
                if elapsed < 5.0:  # 5 сек
                    logger.debug(f"⏱Пропущено: {camera_name} — ещё не прошло 3 сек. (прошло: {elapsed:.1f}c)")
                    continue  # Пропускаем отправку

            # Разрешено отправить — обновляем время
            last_sent_time[camera_name] = now

            # Отправляем фото
            try:
                message = await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_bytes,
                    caption=caption
                )
                logger.info(f"Успешно отправлено: {camera_name} | Message ID: {message.message_id}")

            except Exception as e:
                logger.error(f"ОШИБКА при отправке фото для {camera_name}: {type(e).__name__}: {e}")
                if "file is too big" in str(e).lower():
                    logger.error("Уменьши качество JPEG (IMWRITE_JPEG_QUALITY)")
                elif "Bad Request: wrong file identifier" in str(e):
                    logger.error("Повреждённые байты изображения. Проверь cv2.imencode()")
                elif "Forbidden" in str(e):
                    logger.error("Бот заблокирован пользователем или чат удалён.")
                elif "Timed out" in str(e):
                    logger.error("Проблемы с сетью. Проверь интернет.")

        except mp.queues.Empty:
            logger.debug("⏳ Очередь пуста — ждём следующего кадра...")
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Неожиданная ошибка в telegram_sender: {type(e).__name__}: {e}")
            await asyncio.sleep(1)

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    # Устанавливаем метод запуска до создания любых объектов multiprocessing
    if sys.platform == "darwin":  # macOS
        mp.set_start_method('spawn', force=True)
        logger.info("Установлен start_method='spawn' для macOS + Python 3.13")

    manager = mp.Manager()
    message_queue = manager.Queue()  # очередь

    logger.info("Запуск системы мониторинга RTSP с YOLOv8n и асинхронным Telegram...")

    # Запускаем процессы камер — теперь передаём им очередь
    processes = []
    for i, rtsp_url in enumerate(RTSP_STREAMS):
        p = mp.Process(
            target=process_camera,
            args=(i, rtsp_url, MODEL_PATH, message_queue), 
            name=f"Camera_{i+1}"
        )
        p.daemon = True
        p.start()
        processes.append(p)
        logger.info(f"Запущен процесс для камеры {i+1}")

    # асинхронная отправка в ТГ
    try:
        asyncio.run(telegram_sender(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message_queue))
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания. Остановка всех процессов...")
        for p in processes:
            p.terminate()
        logger.info("Все процессы остановлены.")