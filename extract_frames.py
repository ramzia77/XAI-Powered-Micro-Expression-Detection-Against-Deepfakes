import os
import cv2
import pandas as pd

# Paths
labels_csv = 'labels.csv'
real_dir = 'real_videos'
fake_dir = 'fake_videos'
output_dir = 'frames_dataset'
frames_per_video = 5  # you can change this

# Create output dirs
os.makedirs(os.path.join(output_dir, 'REAL'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'FAKE'), exist_ok=True)

# Read CSV
df = pd.read_csv(labels_csv)

# Extract frames
for idx, row in df.iterrows():
    filename, label = row['filename'], row['label']
    video_path = os.path.join(real_dir if label == 'REAL' else fake_dir, filename)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open {filename}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = list(range(0, total_frames, max(total_frames // frames_per_video, 1)))[:frames_per_video]

    for i, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            out_filename = f"{os.path.splitext(filename)[0]}_frame{i}.jpg"
            out_path = os.path.join(output_dir, label, out_filename)
            cv2.imwrite(out_path, frame)
    
    cap.release()

print("✅ Frame extraction complete!")
