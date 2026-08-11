"""Sink State Module - Persistent state object to prevent memory reallocation."""

import time

class SinkState:
    """Holds the real-time state of a single sink without allocating new memory."""
    def __init__(self):
        # Camera & Performance
        self.connection_status = "STANDBY"
        self.last_frame_time = time.monotonic()
        self.fps = 0.0
        self.ai_latency = 0.0
        
        # Session & Authentication
        self.user = "EMPTY"
        self.is_authenticated = False
        self.auth_message = ""
        self.auth_color = "normal"
        
        # Toggles (from settings)
        self.check_mask = True
        self.check_hat = True
        self.check_wash = True
        
        # PPE Detection
        self.has_mask = False
        self.has_hat = False
        
        # Washing & WHO Tracking
        self.actively_washing = False
        self.wash_time = 0.0
        self.wash_status_text = "STANDBY"
        self.current_who_step = 0
        self.completed_who_steps = set()
        
        # Legacy/System
        self.is_recording = False  # Maintained for structure, though recording is removed
        self.master_ready = False