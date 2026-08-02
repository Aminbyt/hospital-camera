import os
import glob
import math
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict

# --- CONFIGURATION ---
FRAMES_DIR = "./AI handwash/dataset-pskus/PSKUS_dataset/frames/trainval"
ONNX_SAVE_PATH = "who_cnn_lstm_model.onnx"
MAX_SAMPLES_PER_CLASS = 15000
SEQ_LEN = 30       # INCREASED: Look at 30 continuous frames (1 full second of motion)
INPUT_DIM = 128    # INCREASED: 126 (coords) + 2 (hand-to-hand distances)
HIDDEN_DIM = 64    # Lightweight hidden layer for fast CPU inference
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 25

# --- MEDIAPIPE INITIALIZATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def extract_features_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks: return None

    sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
    features = []
    
    # 1. Extract standard normalized coordinates for up to 2 hands
    for hand in sorted_hands[:2]:
        wrist = hand.landmark[0]
        middle_base = hand.landmark[9]
        hand_size = math.hypot(wrist.x - middle_base.x, wrist.y - middle_base.y)
        if hand_size == 0: hand_size = 1.0

        for lm in hand.landmark:
            features.extend([
                (lm.x - wrist.x) / hand_size,
                (lm.y - wrist.y) / hand_size,
                (lm.z - wrist.z) / hand_size
            ])
            
    # Pad coordinate features if only 1 hand is visible to reach 126
    while len(features) < 126:
        features.append(0.0)
        
    # 2. Extract Relative Distance Features (Crucial for WHO steps)
    if len(sorted_hands) == 2:
        h1_wrist = sorted_hands[0].landmark[0]
        h2_wrist = sorted_hands[1].landmark[0]
        wrist_dist = math.hypot(h1_wrist.x - h2_wrist.x, h1_wrist.y - h2_wrist.y)
        
        h1_index = sorted_hands[0].landmark[8]
        h2_index = sorted_hands[1].landmark[8]
        index_dist = math.hypot(h1_index.x - h2_index.x, h1_index.y - h2_index.y)
        
        features.extend([wrist_dist, index_dist])
    else:
        # If only one hand, apply maximum distance penalty (1.0 is max normalized space)
        features.extend([1.0, 1.0])
        
    return features

# --- PYTORCH CNN + LSTM ARCHITECTURE ---
class HandwashCNN_LSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super(HandwashCNN_LSTM, self).__init__()
        
        # 1D CNN explicitly designed to find local velocity/acceleration between frames
        self.conv1d = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Incoming x shape: (batch_size, seq_len, input_dim)
        
        # 1. Prepare for Conv1D: PyTorch Conv1d expects (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1) 
        
        # 2. Apply Conv1D + Activation
        x = torch.relu(self.conv1d(x))
        
        # 3. Prepare for LSTM: Revert shape to (batch_size, sequence_length, channels)
        x = x.permute(0, 2, 1)
        
        # 4. Standard LSTM processing
        out, (hn, cn) = self.lstm(x)
        
        # 5. Take the output of the very last timestep
        last_out = out[:, -1, :]
        return self.fc(last_out)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    print("[INFO] Starting Sequential Feature Extraction...")
    class_data = defaultdict(lambda: defaultdict(list))

    for class_id in range(7):
        folder_path = os.path.join(FRAMES_DIR, str(class_id))
        if not os.path.exists(folder_path): continue
        images = glob.glob(os.path.join(folder_path, "*.jpg"))[:MAX_SAMPLES_PER_CLASS]
        print(f" -> Processing Class {class_id}: {len(images)} frames...")

        for img_path in images:
            basename = os.path.basename(img_path).replace(".jpg", "")
            parts = basename.split("_")
            try:
                frame_num = int(parts[1])
                snippet_id = "_".join(parts[2:])
            except:
                continue
            features = extract_features_from_image(img_path)
            if features:
                class_data[class_id][snippet_id].append((frame_num, features))
    hands.close()

    # --- BUILD TEMPORAL SLIDING WINDOWS ---
    print(f"\n[INFO] Building {SEQ_LEN}-Frame Time Sequences...")
    X_seq, y_seq = [], []

    for class_id, snippets in class_data.items():
        for snippet_id, frames in snippets.items():
            frames.sort(key=lambda x: x[0])
            for i in range(len(frames) - SEQ_LEN + 1):
                window = [f[1] for f in frames[i : i + SEQ_LEN]]
                X_seq.append(window)
                y_seq.append(class_id)

    X = np.array(X_seq)
    y = np.array(y_seq)
    print(f"[INFO] Final Sequence Dataset Shape: {X.shape} (Samples x {SEQ_LEN} frames x {INPUT_DIM} features)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training CNN-LSTM on device: {device}...")

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate the new CNN-LSTM architecture
    model = HandwashCNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(device)
                    preds = torch.argmax(model(batch_X), dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(batch_y.numpy())
            acc = accuracy_score(all_targets, all_preds)
            print(f"  Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc*100:.2f}%")

    # --- EXPORT TO ONNX ---
    print(f"\n[INFO] Exporting CNN-LSTM to ONNX format ({ONNX_SAVE_PATH})...")
    model.eval().to("cpu")
    
    # Notice the dummy input uses the new SEQ_LEN (30) and INPUT_DIM (128)
    dummy_input = torch.randn(1, SEQ_LEN, INPUT_DIM)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_SAVE_PATH,
        input_names=["input_sequence"],
        output_names=["class_logits"],
        dynamic_axes={"input_sequence": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
        opset_version=12
    )
    print(f"[SUCCESS] Saved ONNX model to {ONNX_SAVE_PATH}. Ready for main_app.py!")

if __name__ == "__main__":
    main()