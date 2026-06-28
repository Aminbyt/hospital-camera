"""Home Summary Tab - Displays a modern UI card grid of all active sinks."""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QFrame, QLabel, QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import config

class HomeSummaryTab(QWidget):
    """Grid overview of all 5 hospital sinks."""
   
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sink_cards = {}
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
       
        title = QLabel("HOSPITAL WARD OVERVIEW")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1b4332; letter-spacing: 2px;")
        layout.addWidget(title)
        layout.addSpacing(30)

        self.grid = QGridLayout()
        self.grid.setSpacing(20) # Add nice spacing between the cards
        layout.addLayout(self.grid)

        # Create 5 modern cards dynamically
        for i in range(1, 6):
            sink_name = f"SINK {i}"
            card, labels = self.create_sink_card(sink_name)
            self.sink_cards[sink_name] = labels
           
            # Math to arrange them in a clean 2x3 grid layout
            row = (i - 1) // 3
            col = (i - 1) % 3
            self.grid.addWidget(card, row, col)
           
        layout.addStretch()

    def create_sink_card(self, title_text):
        """Builds a single sleek summary card for the grid."""
        card = QFrame()
        # Modern Card Styling: Light background, subtle border, rounded corners
        card.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 10px;
            }
        """)
       
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Header (Sink Name)
        lbl_title = QLabel(title_text)
        lbl_title.setFont(QFont("Arial", 16, QFont.Bold))
        lbl_title.setStyleSheet("color: #212529; border: none; background: transparent;")
        layout.addWidget(lbl_title)
       
        # Divider Line
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #dee2e6; border: none;")
        layout.addWidget(line)
        layout.addSpacing(5)
        

        # User Info
        lbl_user = QLabel("USER: EMPTY")
        lbl_user.setFont(QFont("Arial", 11, QFont.Bold))
        lbl_user.setStyleSheet("color: #6c757d; border: none; background: transparent;")
       
        # PPE Info
        lbl_ppe = QLabel("MASK: - | HAT: -")
        lbl_ppe.setFont(QFont("Arial", 10))
        lbl_ppe.setStyleSheet("color: #495057; border: none; background: transparent;")
       
        # Status Info
        lbl_status = QLabel("STATUS: STANDBY")
        lbl_status.setFont(QFont("Arial", 10, QFont.Bold))
        lbl_status.setStyleSheet("color: #495057; border: none; background: transparent;")

        # --- MODERN THIN PROGRESS BAR ---
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(5)
       
        # Time counter (sits above the bar now)
        lbl_time = QLabel("0 SECONDS")
        lbl_time.setFont(QFont("Arial", 9, QFont.Bold))
        lbl_time.setStyleSheet("color: #1b4332; border: none; background: transparent;")
        lbl_time.setAlignment(Qt.AlignRight)

        # The actual bar
        progress = QProgressBar()
        progress.setMaximum(config.MAX_WASH_TIME)
        progress.setValue(0)
        progress.setTextVisible(False) # Hide the default chunky text
        progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background: #e9ecef;
                height: 8px;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #2d6a4f;
                border-radius: 4px;
            }
        """)

        progress_layout.addWidget(lbl_time)
        progress_layout.addWidget(progress)

        # Add everything to the card
        layout.addWidget(lbl_user)
        layout.addWidget(lbl_ppe)
        layout.addWidget(lbl_status)
        layout.addSpacing(10)
        layout.addLayout(progress_layout)

        # Return a dictionary of the labels so the background thread can update them!
        labels = {
            'user': lbl_user,
            'ppe': lbl_ppe,
            'status': lbl_status,
            'progress': progress,
            'time': lbl_time
        }
        return card, labels

    def update_sink_data(self, sink_name, data):
        """Triggered automatically by the background threads to update the UI."""
       
        # --- THE FIX: Convert "SINK_1" to "SINK 1" so the dictionary finds it! ---
        clean_name = sink_name.replace("_", " ")

        if clean_name not in self.sink_cards:
            return
           
        labels = self.sink_cards[clean_name]
       
        # Update User Status & Color
        user = data.get('user', 'EMPTY')
        labels['user'].setText(f"USER: {user}")
        if user != 'EMPTY':
            labels['user'].setStyleSheet("color: #1b4332; border: none; background: transparent;")
        else:
            labels['user'].setStyleSheet("color: #dc3545; border: none; background: transparent;")

        # Update PPE Status
        mask = "✅" if data.get('mask', False) else "❌"
        hat = "✅" if data.get('hat', False) else "❌"
        labels['ppe'].setText(f"MASK: {mask}   |   HAT: {hat}")

        # Update Progress Bar and the new Timer Text
        wash_time = int(data.get('wash_time', 0))
        labels['progress'].setValue(wash_time)
        labels['time'].setText(f"{wash_time} SECONDS")
       
        # Update Status Text
        labels['status'].setText(f"STATUS: {data.get('wash_status', 'STANDBY')}")



