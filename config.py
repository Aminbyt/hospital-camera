"""Configuration and constants for the Hospital AI System."""

import os

# --- ENVIRONMENT & SECRETS LOADING ---
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads variables from a local .env file if present
except ImportError:
    pass

# --- PATHS ---
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
DB_PATH = os.path.join(desktop_path, "Hospital_Database")

REG_PATH = os.path.join(DB_PATH, "REGISTER_PERSONS")
INFO_PATH = os.path.join(DB_PATH, "INFORMATION")
YOLO_MODEL_PATH = 'runs/detect/train/weights/best_openvino_model/'

# Create necessary directories automatically
os.makedirs(REG_PATH, exist_ok=True)
os.makedirs(INFO_PATH, exist_ok=True)
os.makedirs(os.path.join(DB_PATH, "RECORDINGS"), exist_ok=True)
os.makedirs(os.path.join(DB_PATH, "LOGS"), exist_ok=True)

# --- PERFORMANCE & STABILITY SETTINGS ---
# Capture & UI Rates
CAMERA_FPS = 20
UI_FPS = 10

# AI Inference Rates (Target FPS)
FACE_FPS = 5
PPE_FPS = 5
HAND_FPS = 12
WHO_FPS = 8

# Queue Sizes
FRAME_QUEUE_SIZE = 1
RECORD_QUEUE_SIZE = 30

# ONNX Runtime Thread Limits
ORT_INTRA_THREADS = 2
ORT_INTER_THREADS = 1

# Watchdog & Health Monitoring
CAMERA_TIMEOUT_SEC = 5
WORKER_TIMEOUT_SEC = 10
HEALTH_LOG_INTERVAL_SEC = 30

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
AUTH_COOLDOWN = 2.0      # seconds between auth attempts
PRESENCE_TIMEOUT = 6.0   # seconds before auto-logout if no face detected
TOUCH_TIMEOUT = 2.5      # seconds to allow single hand washing in bubble zone

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

# --- SECURE BOT NOTIFICATION SETTINGS ---
BOT_SERVICE_URL = os.getenv("BOT_SERVICE_URL", "https://tapi.bale.ai")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
BOT_CHAT_ID = os.getenv("BOT_CHAT_ID", "")
BOT_TIMEOUT = int(os.getenv("BOT_TIMEOUT", "3"))

# Derived API Endpoints
if BOT_TOKEN:
    BOT_API_URL = f"{BOT_SERVICE_URL}/{BOT_TOKEN}/sendMessage"
    BOT_DOC_URL = f"{BOT_SERVICE_URL}/{BOT_TOKEN}/sendDocument"
else:
    BOT_API_URL = ""
    BOT_DOC_URL = ""

# --- WINDOW SETTINGS ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

# --- SECURE CAMERA HARDWARE MAPPING ---
def get_sink_cameras():
    """Dynamically builds camera endpoints without committing credentials."""
    sinks = {}
    
    sink_1_src = os.getenv("SINK_1_CAM", "0")
    sinks["SINK_1"] = int(sink_1_src) if sink_1_src.isdigit() else sink_1_src
    
    for i in range(2, 6):
        cam_env = os.getenv(f"SINK_{i}_CAM")
        if cam_env:
            sinks[f"SINK_{i}"] = int(cam_env) if cam_env.isdigit() else cam_env
            
    return sinks

SINK_CAMERAS = get_sink_cameras()