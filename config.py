"""Configuration and constants for the Hospital AI System."""

import os

# --- ENVIRONMENT VARIABLES ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer"
# --- PATHS ---
DB_PATH = "database"
REG_PATH = os.path.join(DB_PATH, "REGISTER_PERSONS")
INFO_PATH = os.path.join(DB_PATH, "INFORMATION")
YOLO_MODEL_PATH = 'runs/detect/train/weights/best_openvino_model/'

# Create necessary directories
os.makedirs(REG_PATH, exist_ok=True)
os.makedirs(INFO_PATH, exist_ok=True)

# --- AI MODEL PARAMETERS ---
YOLO_CONF_THRESHOLD = 0.6
FACE_DETECTION_CONFIDENCE = 0.5
HAND_DETECTION_CONFIDENCE = 0.4
HAND_TRACKING_CONFIDENCE = 0.4
MAX_NUM_HANDS = 2

# --- WASH TIMING (in seconds) ---
MIN_WASH_TIME = 20
MAX_WASH_TIME = 40

# --- AUTHENTICATION PARAMETERS ---
AUTH_COOLDOWN = 2.0  # seconds between auth attempts
PRESENCE_TIMEOUT = 6.0  # seconds before auto-logout if no face detected
TOUCH_TIMEOUT = 2.5  # seconds to allow single hand washing in bubble zone

# --- HAND GEOMETRY ---
WRIST_DISTANCE_THRESHOLD = 65
HAND_SIZE_MULTIPLIER = 2.5
MIN_BUBBLE_RADIUS = 250

# --- UI STYLING ---
STYLESHEET = """
    QMainWindow { background-color: #ffffff; }
    QWidget { color: #000000; font-family: 'Segoe UI', Arial, sans-serif; background-color: #ffffff; }
    QTabWidget::pane { border: 1px solid #cccccc; }
    QTabBar::tab { background: #f5f5f5; border: 1px solid #cccccc; padding: 10px 15px; margin-right: 2px; font-weight: bold; font-size: 12px; min-width: 180px; }
    QTabBar::tab:selected { background: #ffffff; border-bottom: 2px solid #000000; }
    QGroupBox { border: 1px solid #000000; margin-top: 20px; font-weight: bold; font-size: 12px; padding-top: 15px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
    QPushButton { background: #ffffff; border: 2px solid #000000; padding: 10px 20px; font-weight: bold; font-size: 12px; }
    QPushButton:hover { background: #eeeeee; }
    QPushButton:pressed { background: #000000; color: #ffffff; }
    QLineEdit, QSpinBox { border: 1px solid #cccccc; padding: 8px; background: #ffffff; font-size: 12px; }
    QProgressBar { border: 2px solid #000000; background: #ffffff; height: 30px; text-align: center; }
    QProgressBar::chunk { background-color: #1b4332; }
    QCheckBox, QRadioButton { font-size: 12px; font-weight: 500; }
"""

# --- BOT NOTIFICATION SETTINGS ---
BOT_API_URL = "https://tapi.bale.ai/1291761237:o-9xVmgV_Vw4iS-5XA9Yc4TWQ182YqNM5v8/sendMessage"
BOT_CHAT_ID = "6277616651"
BOT_TIMEOUT = 3  # seconds

# --- WINDOW SETTINGS ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

# --- CAMERA HARDWARE MAPPING ---
# Dahua RTSP Format: rtsp://username:password@IP_Address:554/cam/realmonitor?channel=1&subtype=1
SINK_CAMERAS = {
    "SINK_1": "rtsp://admin:1937@asdF@192.168.1.101:554/cam/realmonitor?channel=1&subtype=1",
    "SINK_2": "rtsp://admin:1937@asdF@192.168.1.102:554/cam/realmonitor?channel=1&subtype=1",
    "SINK_3": "rtsp://admin:1937@asdF@192.168.1.103:554/cam/realmonitor?channel=1&subtype=1",
    "SINK_4": "rtsp://admin:1937@asdF@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1",
    "SINK_5": "rtsp://admin:1937@asdF@192.168.1.105:554/cam/realmonitor?channel=1&subtype=1"
}