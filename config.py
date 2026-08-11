"""Configuration and constants for the Hospital AI System."""
import re
import os

# --- ENVIRONMENT & SECRETS LOADING ---
try:
    from dotenv import load_dotenv
    load_dotenv('hospital_camera.env')  # Loads variables from a local .env file if present
except ImportError:
    print("WARNING: python-dotenv is not installed. Run 'pip install python-dotenv'")

# --- PATHS ---
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
DB_PATH = os.path.join(desktop_path, "Hospital_Database")

REG_PATH = os.path.join(DB_PATH, "REGISTER_PERSONS")
INFO_PATH = os.path.join(DB_PATH, "INFORMATION")
YOLO_MODEL_PATH = 'runs/detect/train/weights/best_openvino_model/'

# Create necessary directories automatically
os.makedirs(REG_PATH, exist_ok=True)
os.makedirs(INFO_PATH, exist_ok=True)
os.makedirs(os.path.join(DB_PATH, "LOGS"), exist_ok=True)

# --- PERFORMANCE & STABILITY SETTINGS ---
# Capture & UI Rates
CAMERA_FPS = 20
UI_FPS = 30

# AI Inference Rates (Target FPS)
FACE_FPS = 5
PPE_FPS = 5
HAND_FPS = 12
WHO_FPS = 8

# Queue Sizes
FRAME_QUEUE_SIZE = 1

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

NETWORK_MAX_RETRIES = 3
NETWORK_BACKOFF_BASE = 2.0  

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
BOT_TOKEN = os.getenv("BOT_TOKEN", "PUT_YOUR_ACTUAL_TOKEN_HERE")
BOT_CHAT_ID = os.getenv("BOT_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")
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

import re

# --- CAMERA HARDWARE MAPPING ---
# --- CAMERA HARDWARE MAPPING ---
def get_sink_cameras():
    """Dynamically parses custom dictionary or standard env syntax from the .env file."""
    sinks = {}
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hospital_camera.env")
    
    if not os.path.exists(env_file):
        print(f"[WARNING] {env_file} not found! No cameras loaded.")
        return sinks
        
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # --- FIX: Match BOTH "SINK_1": 0 AND SINK_1=0 formats ---
            match = re.search(r'["\']?(SINK_\d+)["\']?\s*[:=]\s*(.*)', line)
            
            if match:
                sink_name = match.group(1)
                raw_val = match.group(2).strip()
                
                # Clean trailing commas
                if raw_val.endswith(','):
                    raw_val = raw_val[:-1]
                    
                # Clean quotes for strings
                if raw_val.startswith('"') and raw_val.endswith('"'):
                    raw_val = raw_val[1:-1]
                elif raw_val.startswith("'") and raw_val.endswith("'"):
                    raw_val = raw_val[1:-1]
                    
                # Convert to integer if it's a webcam ID (like 0)
                if raw_val.isdigit():
                    sinks[sink_name] = int(raw_val)
                else:
                    sinks[sink_name] = raw_val
                    
    return sinks

SINK_CAMERAS = get_sink_cameras()
