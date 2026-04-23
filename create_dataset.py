import os
import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- CONFIG ---
# Make sure this path is exactly where your folders are
DATA_DIR = r"D:\college\sahil major\PROJECT\hindi_sign_language_images"
OUTPUT_FILE = "data.pickle"

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
# slightly lowered confidence to ensure it picks up hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) 

data = []
labels = []

print("📂 Scanning dataset directories...")

if not os.path.exists(DATA_DIR):
    print(f"❌ ERROR: Dataset path not found: {DATA_DIR}")
    exit()

classes = sorted(os.listdir(DATA_DIR))

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    print(f"Processing Class: {class_name}")
    
    files = os.listdir(class_dir)
    if len(files) == 0:
        print(f"   ⚠️ Warning: Folder '{class_name}' is empty!")
    
    image_count = 0
    success_count = 0
    
    for img_name in files:
        image_count += 1
        img_path = os.path.join(class_dir, img_name)
        
        # --- FIX: UNICODE SAFE LOADING ---
        # OpenCV standard imread fails with Hindi paths, so we use numpy
        try:
            stream = open(img_path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            stream.close()
        except Exception as e:
            print(f"   ❌ Error loading {img_name}: {e}")
            continue

        if img is None:
            print(f"   ❌ Failed to load image: {img_name}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(class_name)
                success_count += 1
        else:
            # Optional: Uncomment to see which images fail detection
            # print(f"   ⚠️ No hand detected in: {img_name}")
            pass

    print(f"   --> Found hands in {success_count}/{image_count} images")

# Save
if len(data) > 0:
    f = open(OUTPUT_FILE, 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()
    print(f"\n✅ SUCCESS! Saved {len(data)} samples to {OUTPUT_FILE}")
else:
    print("\n❌ FAILED: Still 0 samples. Check your images manually.")