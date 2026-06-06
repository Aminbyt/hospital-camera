import os
from collections import Counter

labels_folder_path = r"C:\Users\0150027771\Desktop\CCTV\dataset_final\val\labels"

class_mapping = {
    0: 'Hardhat',
    1: 'NO-Hardhat',
    2: 'Vest',
    3: 'NO-Vest'
}


def count_yolo_labels(folder_path):
    class_counts = Counter()
    total_files = 0
    total_boxes = 0
   
    if not os.path.exists(folder_path):
        print(f"Error: Could not find the folder '{folder_path}'. Please check the path.")
        return
       
    print(f"Scanning folder: {folder_path}...\n")
   
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and filename != "classes.txt":
            total_files += 1
            file_path = os.path.join(folder_path, filename)
           
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5: 
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_boxes += 1
                       
    print("-" * 40)
    print("📊 DATASET BALANCE REPORT 📊")
    print("-" * 40)
    print(f"Total images (txt files) scanned: {total_files}")
    print(f"Total bounding boxes found:       {total_boxes}\n")
   
    for class_id, name in class_mapping.items():
        count = class_counts.get(class_id, 0)
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"[{class_id}] {name:<12}: {count} boxes ({percentage:.1f}%)")
       
    print("-" * 40)

count_yolo_labels(labels_folder_path)
