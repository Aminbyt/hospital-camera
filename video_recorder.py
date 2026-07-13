"""Video Recorder Module - Handles I/O saving in a background thread."""

import cv2
import os
import time
from queue import Queue
from PyQt5.QtCore import QThread
import config

class VideoRecorder(QThread):
    def __init__(self, sink_name):
        super().__init__()
        self.sink_name = sink_name
        self.frame_queue = Queue(maxsize=300) # Buffer to hold frames in RAM
        self.is_recording = False
        self.video_writer = None
        self.current_filepath = ""

    def start_recording(self, user_name, frame_w, frame_h, fps=20.0):
        """Initializes the video file and starts the background writing thread."""
        record_dir = os.path.join(config.DB_PATH, "RECORDINGS")
        os.makedirs(record_dir, exist_ok=True)
        
        user_str = user_name.replace(" ", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_filepath = os.path.join(record_dir, f"{user_str}_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.current_filepath, fourcc, fps, (frame_w, frame_h))
        
        self.is_recording = True
        self.start() # Starts the run() loop below
        print(f"[{self.sink_name}] 🔴 THREAD RECORDING: {self.current_filepath}")

    def add_frame(self, frame):
        """Instantly drops a frame into the queue without blocking the AI."""
        if self.is_recording and not self.frame_queue.full():
            self.frame_queue.put(frame)

    def stop_recording(self):
        """Flags the thread to stop and waits for the queue to empty."""
        self.is_recording = False
        # Do not call self.wait() here, let it finish naturally to avoid UI freezes

    def run(self):
        """The background loop that writes to the hard drive."""
        # Keep spinning as long as we are recording, OR if there are still frames left to save
        while self.is_recording or not self.frame_queue.empty():
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if self.video_writer:
                    self.video_writer.write(frame)
            else:
                self.msleep(10) # Rest the CPU if the queue is empty
        
        # Cleanup when finished
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"[{self.sink_name}] ⏹️ THREAD SAVED VIDEO: {self.current_filepath}")