import csv
import glob
import os
import cv2
import mediapipe as mp
import pandas as pd
import math

# --- CONFIGURATION ---
DATASET_DIR = r"C:\Users\0150027771\Desktop\hospital_camera\DataSet1\DataSet1"  # Or your root dataset folder path
VIDEOS_DIR = os.path.join(DATASET_DIR, "videos")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
OUTPUT_CSV = "who_handwashing_dataset.csv"

# Map researcher movement codes (Column 2) to WHO Steps (0-6)
VALID_CODES = {
    0: 0,  # Idle / Non-WHO movement
    1: 1,  # Step 1: Palm to palm
    2: 2,  # Step 2: Right over left dorsum
    3: 3,  # Step 3: Palm to palm interlaced
    4: 4,  # Step 4: Backs of fingers
    5: 5,  # Step 5: Thumb rotation
    6: 6,  # Step 6: Fingertips
    7: 0,  # Wrist washing / Other -> mapped to 0 (Idle)
}


def find_annotation_file(base_name, video_name):
  """Fuzzy searches all subfolders in ANNOTATIONS_DIR for a matching CSV or TXT file."""
  clean_target = (
      base_name.lower()
      .replace("_camera101", "")
      .replace("_camera102", "")
      .strip()
  )

  for root, dirs, files in os.walk(ANNOTATIONS_DIR):
    for file in files:
      file_lower = file.lower()
      if not (file_lower.endswith(".csv") or file_lower.endswith(".txt")):
        continue

      file_base = os.path.splitext(file)[0].lower()
      clean_base = (
          file_base.replace("_camera101", "").replace("_camera102", "").strip()
      )

      if (
          clean_target in clean_base
          or clean_base in clean_target
          or file_lower == f"{video_name.lower()}.csv"
          or file_lower == f"{video_name.lower()}.txt"
      ):
        return os.path.join(root, file)
  return None


def get_annotation_for_time(df_annotations, current_msec):
  """Locates the WHO movement code for a given timestamp in MILLISECONDS with a 50ms tolerance."""
  try:
    # Column 0 is frame_time in MILLISECONDS (33.333, 66.667, 100.000...)
    idx = (df_annotations.iloc[:, 0] - current_msec).abs().idxmin()
    time_diff = abs(df_annotations.iloc[idx, 0] - current_msec)

    # 50 millisecond tolerance covers minor frame rate jitter
    if time_diff < 50.0:
      is_washing = int(df_annotations.iloc[idx, 1])
      code = int(df_annotations.iloc[idx, 2])

      if is_washing == 0:
        return 0  # Not actively washing
      return VALID_CODES.get(code, 0)
  except Exception:
    pass
  return None


def main():
  print(f"🚀 Starting Extraction on '{DATASET_DIR}'...")

  # 1. MUST DELETE OLD CSV BEFORE THE LOOP STARTS!
  if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)
    print(f"🧹 Removed old '{OUTPUT_CSV}' to ensure clean extraction.")

  # Write headers once
  headers = [
      f"h{i}_lm{j}_{axis}"
      for i in (1, 2)
      for j in range(21)
      for axis in ("x", "y", "z")
  ]
  headers.append("hands_dist") 
  headers.append("label")
  with open(OUTPUT_CSV, mode="w", newline="") as f:
    csv.writer(f).writerow(headers)


  video_files = [
      f
      for f in os.listdir(VIDEOS_DIR)
      if f.lower().endswith((".mp4", ".avi", ".mov"))
  ]
  print(f"📁 Found {len(video_files)} videos in '{VIDEOS_DIR}'.")

  # Initialize MediaPipe Hands
  mp_hands = mp.solutions.hands
  hands = mp_hands.Hands(
      static_image_mode=False,
      max_num_hands=2,
      min_detection_confidence=0.5,
  )

  stats = {"success": 0, "no_annotation": 0, "no_hands": 0, "total_rows": 0}

  # 2. PROCESS VIDEOS
  for idx, video_name in enumerate(video_files, 1):
    video_path = os.path.join(VIDEOS_DIR, video_name)
    base_name = os.path.splitext(video_name)[0]

    csv_path = find_annotation_file(base_name, video_name)
    if not csv_path:
      print(
          f"[{idx}/{len(video_files)}] ⚠️ SKIPPED: {video_name} (No annotation"
          " file found)"
      )
      stats["no_annotation"] += 1
      continue

    try:
      # Use header=0 so pandas correctly reads column names instead of treating row 0 as data!
      df_annotations = pd.read_csv(csv_path, header=0)
    except Exception as e:
      print(
          f"[{idx}/{len(video_files)}] ❌ ERROR reading CSV for {video_name}:"
          f" {e}"
      )
      continue

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0
    video_rows_saved = 0

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      frame_count += 1
      # Sample every 2nd frame (~15 frames per second) to speed up extraction
      if frame_count % 2 != 0:
        continue

      # CONVERT TO MILLISECONDS TO MATCH RESEARCHER CSV!
      current_msec = (frame_count / fps) * 1000.0
      who_label = get_annotation_for_time(df_annotations, current_msec)

      if who_label is None:
        continue

      # Process AI Hand Landmarks
      img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = hands.process(img_rgb)

      if results.multi_hand_landmarks:
        # --- 1. SORT BY PHYSICAL HANDEDNESS (LEFT VS RIGHT), NOT X-COORD ---
        # This prevents features from swapping slots when hands cross over!
        hand_dict = {'Left': None, 'Right': None}
        for idx, hand_info in enumerate(results.multi_handedness):
            label = hand_info.classification[0].label  # 'Left' or 'Right'
            hand_dict[label] = results.multi_hand_landmarks[idx]
           
        ordered_hands = [hand_dict['Left'], hand_dict['Right']]

        row = []
        base_hand_size = 0.01

        for hand in ordered_hands:
          if hand is not None:
            wrist_x = hand.landmark[0].x
            wrist_y = hand.landmark[0].y
            wrist_z = hand.landmark[0].z
           
            mcp_x = hand.landmark[9].x
            mcp_y = hand.landmark[9].y
            hand_size = max(math.hypot(mcp_x - wrist_x, mcp_y - wrist_y), 0.01)
           
            # Save the first valid hand size to normalize inter-hand distance later
            if base_hand_size == 0.01:
                base_hand_size = hand_size

            for lm in hand.landmark:
              row.extend([
                  (lm.x - wrist_x) / hand_size,
                  (lm.y - wrist_y) / hand_size,
                  (lm.z - wrist_z) / hand_size
              ])
          else:
            # If Left or Right hand is missing, fill its 63 slots with 0.0
            row.extend([0.0] * 63)

        # --- 2. NORMALIZED INTER-HAND DISTANCE ---
        if ordered_hands[0] is not None and ordered_hands[1] is not None:
          h1 = ordered_hands[0].landmark[0]
          h2 = ordered_hands[1].landmark[0]
          # Divide by base_hand_size so it is scale-invariant!
          dist = math.hypot(h1.x - h2.x, h1.y - h2.y) / base_hand_size
        else:
          dist = 0.0
        row.append(dist)

        row.append(who_label)


        # 3. APPEND TO CSV INSIDE THE LOOP (mode="a")
        with open(OUTPUT_CSV, mode="a", newline="") as f:
          csv.writer(f).writerow(row)

        video_rows_saved += 1

    cap.release()

    if video_rows_saved > 0:
      print(
          f"[{idx}/{len(video_files)}] ✅ {video_name} -> Extracted"
          f" {video_rows_saved} rows."
      )
      stats["success"] += 1
      stats["total_rows"] += video_rows_saved
    else:
      print(
          f"[{idx}/{len(video_files)}] ⚠️ {video_name} -> 0 rows extracted"
          " (No hands detected / timestamp mismatch)."
      )
      stats["no_hands"] += 1

  hands.close()

  # --- DIAGNOSTIC SUMMARY ---
  print("\n" + "=" * 40)
  print("🏁 EXTRACTION SUMMARY REPORT")
  print("=" * 40)
  print(f"Total Videos Processed:  {len(video_files)}")
  print(f"✅ Successfully Extracted: {stats['success']} videos")
  print(f"⚠️ Missing Annotations:    {stats['no_annotation']} videos")
  print(f"⚠️ 0 Samples Extracted:    {stats['no_hands']} videos")
  print(
      f"📊 Total Dataset Rows:   {stats['total_rows']} rows saved to"
      f" '{OUTPUT_CSV}'"
  )
  print("=" * 40)


if __name__ == "__main__":
  main()