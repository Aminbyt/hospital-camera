import cv2
import os
import glob
import shutil

VIDEO_PATH = r"C:\Users\0150027771\Desktop\Hospital Camera_V2\recordings\VID_20260502_131134_874.mp4.mov"
LABELS_FOLDER = r"C:\Users\bayat\OneDrive\Desktop\project\dataset"    
OUTPUT_FOLDER = r"C:\Users\bayat\OneDrive\Desktop\project\dataset_raw"      

def extract_images():
    os.makedirs(f"{OUTPUT_FOLDER}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/labels", exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video '{VIDEO_PATH}'")
        return

    txt_files = glob.glob(os.path.join(LABELS_FOLDER, "*.txt"))
    print(f" Found {len(txt_files)} label files. Extracting frames...")

    count = 0
    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        
        try:
            frame_str = filename.split('_')[-1].replace('.txt', '')
            frame_num = int(frame_str)
        except ValueError:
            print(f" Skipping weird file: {filename}")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()

        if success:
            img_filename = f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(f"{OUTPUT_FOLDER}/images/{img_filename}", frame)
            
            shutil.copy(txt_file, f"{OUTPUT_FOLDER}/labels/{img_filename.replace('.jpg', '.txt')}")
            count += 1
        else:
            print(f"⚠️ Could not read frame {frame_num}")

    cap.release()
    print(f"✅ Done! Extracted {count} images to '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    extract_images()