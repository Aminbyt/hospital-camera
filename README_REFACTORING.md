"""README - Modular Application Structure

# Hospital AI Smart Scrub Sink - Refactored Architecture

## Overview
This refactored version splits the monolithic 1028-line `UI_007.py` into well-organized, maintainable modules following the Single Responsibility Principle.

## Directory Structure

```
project/
├── main_app.py                 # Entry point - Main application window
├── config.py                   # All configuration constants & settings
├── ai_models.py                # AI model integration (YOLO, DeepFace, MediaPipe)
├── hand_wash_detector.py       # Hand washing detection & bubble zone logic
├── sink_calibration.py         # ROI drawing & sink line auto-detection
├── data_logger.py              # Excel export & bot notifications
├── ui_dashboard_tab.py         # Dashboard UI tab component
├── ui_registration_tab.py      # Registration UI tab component
├── ui_settings_tab.py          # Settings UI tab component
├── database/                   # Data storage
│   ├── REGISTER_PERSONS/       # Registered staff face images
│   └── INFORMATION/            # Session logs (Excel files)
└── runs/detect/train/weights/  # YOLO model weights
    └── best.pt
```

## Module Descriptions

### 1. **config.py** - Configuration Hub
**Purpose**: Centralized configuration management

**Key Components**:
- Environment variables
- File paths (database, model weights)
- AI model parameters (YOLO conf, hand detection confidence)
- Timing constants (wash time, auth cooldown)
- Hand gesture thresholds
- UI styling (CSS stylesheet)
- Bot API settings

**Benefits**:
- One place to modify all system settings
- No magic numbers scattered throughout code
- Easy A/B testing different parameters

**Usage**:
```python
import config
print(config.MIN_WASH_TIME)  # 35 seconds
print(config.MAX_WASH_TIME)  # 60 seconds
```

---

### 2. **ai_models.py** - AI Integration
**Purpose**: Manages all machine learning models

**Classes**:
- `FaceRecognitionThread(QThread)` - Async face recognition
- `AIModels` - YOLO, DeepFace, MediaPipe wrapper

**Key Methods**:
- `detect_ppe()` - YOLO detection for mask/hat
- `detect_face()` - MediaPipe face detection
- `detect_hands()` - MediaPipe hand detection
- `draw_hand_landmarks()` - Visualize hand pose
- `get_hand_bbox()` - Extract hand bounding box
- `bboxes_intersect()` - Check hand collision
- `calculate_hand_movement()` - Measure movement speed

**Benefits**:
- Isolated AI model management
- Easy to swap models (e.g., different YOLO version)
- Cleaner main loop logic
- Unit test friendly

---

### 3. **hand_wash_detector.py** - Washing Logic
**Purpose**: Detect and track hand washing

**Class**: `HandWashDetector`

**Key Methods**:
- `detect_washing()` - Core detection algorithm
- `update_wash_time()` - Accumulate wash duration
- `get_wash_status()` - Human-readable status
- `set_bubble_zone()` - Define scrubbing area
- `draw_bubble_zone()` - Visualize zone

**Features**:
- Two-hand interlocking detection
- Bubble zone tracking (single hand mode)
- Movement speed analysis
- Cumulative time tracking with pause/resume

**Benefits**:
- Isolated hand-washing logic
- Easy to debug detection algorithm
- Reusable across different UI frameworks

---

### 4. **sink_calibration.py** - Calibration System
**Purpose**: Auto-detect or manually define scrubbing zone

**Classes**:
- `ROIDrawer(QWidget)` - Interactive ROI drawing widget
- `SinkCalibration` - Static methods for calibration

**Key Features**:
- Auto-detection using edge + Hough transform
- Manual ROI drawing with mouse
- Normalized coordinates (0-1 range)
- Visualization helpers

**Methods**:
- `auto_detect_sink_line()` - Automatic detection
- `draw_sink_zone()` - Render zone boundary
- `draw_manual_roi()` - Render manual ROI
- `create_roi_dialog()` - Popup dialog for drawing

**Benefits**:
- Decoupled from main app
- Testable calibration logic
- Easy manual override

---

### 5. **data_logger.py** - Session Management
**Purpose**: User authentication, logging, and notifications

**Classes**:
- `DataLogger` - Excel export & bot notifications
- `UserSessionManager` - Session state tracking

**DataLogger Methods**:
- `log_session()` - Save to Excel
- `send_bot_notification()` - Send Telegram/Bale alerts
- `log_and_notify()` - Combined operation
- `clear_deepface_cache()` - Refresh face recognition

**UserSessionManager Methods**:
- `set_user()` - Login
- `clear_user()` - Logout
- `check_presence_timeout()` - Auto-logout
- `can_attempt_auth()` - Cooldown check
- `update_presence()` - Reset idle timer

**Benefits**:
- Separation of concerns
- Async-friendly for slow network
- Error resilience (timeouts handled)
- Reusable session management

---

### 6. **ui_dashboard_tab.py** - Main Dashboard
**Purpose**: Display protocol status and controls

**Class**: `DashboardTab(QWidget)`

**UI Components**:
- Video feed display
- User identity label
- Auto-scan status
- Authentication button
- PPE verification status
- Hand washing timer + progress bar
- Master status indicator

**Key Methods**:
- `update_video()` - Render frame
- `set_identity()` - Update user display
- `set_ppe_status()` - Update PPE indicators
- `set_wash_status()` - Update timer
- `set_master_status()` - Overall readiness
- `lock_dashboard()` - Disable when unauthenticated

**Benefits**:
- Pure presentation logic
- Signals for clean communication
- Easy UI tweaks without affecting logic

---

### 7. **ui_registration_tab.py** - Staff Registration
**Purpose**: Add new staff with face capture

**Class**: `RegistrationTab(QWidget)`

**Features**:
- Name & gender form
- Live camera preview
- 5-second countdown
- Automatic face capture
- Personal folder creation
- DeepFace cache clearing

**Key Methods**:
- `start_countdown()` - Begin registration
- `register_new_user()` - Save face & metadata
- `set_frame()` - Update preview
- `display_frame_with_countdown()` - Overlay timer

**Benefits**:
- Isolated registration workflow
- Reusable frame update logic
- Form validation

---

### 8. **ui_settings_tab.py** - Configuration UI
**Purpose**: System settings and camera controls

**Class**: `SettingsTab(QWidget)`

**Features**:
- Detection toggles (mask, hat, washing)
- Webcam vs. video file selection
- Manual ROI drawing
- Auto-calibration reset
- Camera index spinbox

**Signals**:
- `camera_source_changed` - Switch input
- `roi_requested` - Manual drawing mode
- `calibration_requested` - Reset calibration
- `toggles_changed` - Toggle changes

**Benefits**:
- Signal-based communication
- Decoupled from main logic
- Clean settings UI

---

### 9. **main_app.py** - Application Orchestrator
**Purpose**: Main window connecting all components

**Class**: `ScrubSinkKiosk(QMainWindow)`

**Responsibilities**:
- Create and manage tabs
- Coordinate all components
- Main frame processing loop
- Event handling
- Session lifecycle

**Key Methods**:
- `update_frame()` - Main processing loop
- `start_authentication()` - Begin face recognition
- `logout_user()` - End session
- `apply_camera_source()` - Switch input
- `enter_roi_mode()` - Manual calibration
- `trigger_calibration()` - Reset calibration

**Benefits**:
- Clean orchestration
- Easy to understand flow
- Main loop stays maintainable

---

## How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Execution
```bash
python main_app.py
```

### Original Monolithic Version
```bash
python UI_007.py  # Still available for comparison
```

---

## Debugging Tips

### Enable Debug Prints
All modules use `print()` with `[DEBUG]`, `[LOG]`, `[ERROR]` prefixes:
```
[DEBUG] Initializing AI Models...
[LOG] User logged in: John Smith
[ERROR] Could not save to Excel: Permission denied
```

### Test Individual Modules
```python
# Test AI models
from ai_models import AIModels
models = AIModels()
frame, has_mask, has_hat = models.detect_ppe(some_frame)

# Test hand washing
from hand_wash_detector import HandWashDetector
wash = HandWashDetector()
wash.detect_washing(hand_results, w, h, sink_y, models)

# Test calibration
from sink_calibration import SinkCalibration
y = SinkCalibration.auto_detect_sink_line(frame)
```

### Configuration Tweaking
Edit `config.py` for rapid iteration:
```python
MIN_WASH_TIME = 35  # Reduce for testing
YOLO_CONF_THRESHOLD = 0.6  # Lower for sensitivity
```

---

## Benefits of Modular Architecture

### 1. **Maintainability**
- Each module has single responsibility
- Bug fixes isolated to relevant file
- Clear naming conventions

### 2. **Testability**
- Unit test each module independently
- Mock external dependencies
- Easier to reproduce bugs

### 3. **Reusability**
- Use modules in other projects
- Combine with different UIs (web, mobile)
- Export hand washing algorithm separately

### 4. **Scalability**
- Add features without touching existing code
- Easy to add new detection models
- Database abstraction ready

### 5. **Team Collaboration**
- Different developers work on different modules
- Minimal merge conflicts
- Clear module APIs

### 6. **Performance**
- Profile individual modules
- Replace slow components easily
- Optional multithreading per module

---

## Future Enhancements

### Phase 2
- [ ] Multi-camera support
- [ ] Real-time analytics dashboard
- [ ] Advanced hand gesture recognition
- [ ] Integration with hospital EHR systems

### Phase 3
- [ ] Web API for remote monitoring
- [ ] Mobile app for staff check-in
- [ ] Machine learning model optimization
- [ ] Cloud-based data sync

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| YOLO model not found | Check `config.YOLO_MODEL_PATH` |
| DeepFace timeout | Reduce `config.BOT_TIMEOUT` or check internet |
| Camera not detected | Try different index in settings |
| Hand detection poor | Adjust `config.HAND_DETECTION_CONFIDENCE` |
| Wash time too fast | Increase `config.MIN_WASH_TIME` |

---

## License
[Add your license here]

## Contact
[Add contact information]
"""
