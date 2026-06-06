import os
import shutil
import random
import yaml

INPUT_FOLDER = "raw_dataset"      # Make sure this matches your CVAT folder name!
OUTPUT_FOLDER = "dataset_final"
CLASSES = ['mask', 'hat']       # Make sure this matches the order in data.yaml!
SPLIT_RATIO = {'train': 0.8, 'val': 0.2}

def split_data():
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_FOLDER}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_FOLDER}/{split}/labels", exist_ok=True)

    images_found = []
    labels_found = {}

    print("🔍 Searching for files...")
    # 1. Search EVERY subfolder for images and text files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for f in files:
            if f.endswith(valid_extensions):
                images_found.append(os.path.join(root, f))
            elif f.endswith('.txt') and f != 'classes.txt':
                base_name = os.path.splitext(f)[0]
                labels_found[base_name] = os.path.join(root, f)

    print(f"Found {len(images_found)} images and {len(labels_found)} labels total.")

    # 2. Match them together AND save the background images!
    valid_pairs = []
    for img_path in images_found:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if base_name in labels_found:
            # It has a label file
            valid_pairs.append((img_path, labels_found[base_name]))
        else:
            # It is a BACKGROUND image (no label file)
            valid_pairs.append((img_path, None)) 

    if len(valid_pairs) == 0:
        print("❌ Error: Could not find any images!")
        return

    print(f"✅ Successfully matched {len(valid_pairs)} total images (including background images)!")

    # 3. Shuffle and split
    random.shuffle(valid_pairs)
    n_train = int(len(valid_pairs) * SPLIT_RATIO['train'])
    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:]

    # 4. Move files and generate empty txt files for background images
    print("Copying files into train and val folders... (this may take a minute)")
    for split, pairs in [('train', train_pairs), ('val', val_pairs)]:
        for img_path, txt_path in pairs:
            # Copy the image
            img_basename = os.path.basename(img_path)
            shutil.copy(img_path, f"{OUTPUT_FOLDER}/{split}/images/{img_basename}")
            
            # Handle the text file
            txt_basename = os.path.splitext(img_basename)[0] + '.txt'
            if txt_path:
                # Copy the existing label file
                shutil.copy(txt_path, f"{OUTPUT_FOLDER}/{split}/labels/{txt_basename}")
            else:
                # CREATE AN EMPTY TEXT FILE FOR YOLO!
                open(f"{OUTPUT_FOLDER}/{split}/labels/{txt_basename}", 'w').close()

    # 5. Create data.yaml
    data_yaml = {
        'path': os.path.abspath(OUTPUT_FOLDER).replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    with open(f"{OUTPUT_FOLDER}/data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"🎉 All done! Dataset ready at '{OUTPUT_FOLDER}'")
    print(f"📊 Stats: Train={len(train_pairs)}, Val={len(val_pairs)}")

if __name__ == "__main__":
    split_data()