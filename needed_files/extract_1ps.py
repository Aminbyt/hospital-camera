import cv2
import os

INPUT_FOLDER = "recordings"        
OUTPUT_BASE_FOLDER = "images_for_cvat"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm','.dav')

def process_all_videos():
    if not os.path.exists(OUTPUT_BASE_FOLDER):
        os.makedirs(OUTPUT_BASE_FOLDER)
        print(f"Created base folder: {OUTPUT_BASE_FOLDER}")

    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Could not find the folder '{INPUT_FOLDER}'.")
        print(f"Please create a folder named '{INPUT_FOLDER}' and put your videos inside.")
        return

    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(VIDEO_EXTENSIONS)]

    if not video_files:
        print(f"⚠️ No videos found in '{INPUT_FOLDER}'.")
        return

    print(f"🚀 Found {len(video_files)} videos. Starting extraction...")

    for video_name in video_files:
        video_path = os.path.join(INPUT_FOLDER, video_name)
       
        video_output_folder = os.path.join(OUTPUT_BASE_FOLDER, os.path.splitext(video_name)[0])
        os.makedirs(video_output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
       
        frame_count = 0
        saved_count = 0

        print(f"--- Processing: {video_name} ({fps} FPS) ---")

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % fps == 0:
                filename = f"{video_output_folder}/{os.path.splitext(video_name)[0]}_{saved_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                saved_count += 1
           
            frame_count += 1

        cap.release()
        print(f"✅ Saved {saved_count} images to {video_output_folder}")

    print("\n" + "="*40)
    print(f"✨ ALL DONE! Processed {len(video_files)} videos.")
    print(f"📂 Check the '{OUTPUT_BASE_FOLDER}' folder.")

if __name__ == "__main__":
    process_all_videos()
