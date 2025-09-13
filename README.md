## About

Python script to watch RTSP cameras and detect objects (people, cars, animals, etc) and send images to telegram


### Install

```bash
git clone https://github.com/levantez/py-rtsp-yolo-object-detector.git
cd py-rtsp-yolo-object-detector
```

```bash
pip install opencv-python ultralytics numpy python-telegram-bot
```

### Change 
```python
RTSP_STREAMS = [
    "rtsp://login:pass@192.168.01.123:554/cam1",
    "rtsp://login:pass@192.168.01.123:554/cam2",
]

TELEGRAM_TOKEN = "<TOKEN>" # Получи от @BotFather
TELEGRAM_CHAT_ID = "12345" # Получи через @getmyid_bot
```

5 sec pause between send message to telegram
```python
if camera_name in last_sent_time:
                elapsed = now - last_sent_time[camera_name]
                if elapsed < 5.0:  # 5 sec
```


### Run

```bash
python3 rtsp_yolo.py
```

### Systemd daemon

/etc/systemd/system/rtsp-yolo.service
```
[Unit]
Description=Script watchin for RTSP and detect objects and send to Telegram
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/rtsp_dir
ExecStart=python3 rtsp_yolo.py
Restart=always
User=<YOUR_USER>
Group=<YOUR_USER>

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable rtsp-yolo.service
ssytemctl start rtsp-yolo.service
```