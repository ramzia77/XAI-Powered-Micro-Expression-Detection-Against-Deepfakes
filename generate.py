import os
import csv

# Set the paths
real_dir = 'real_videos'
fake_dir = 'fake_videos'
output_csv = 'labels.csv'

# Collect filenames and labels
data = []

# Label real videos
for filename in os.listdir(real_dir):
    if filename.endswith('.mp4'):
        data.append([filename, 'REAL'])

# Label fake videos
for filename in os.listdir(fake_dir):
    if filename.endswith('.mp4'):
        data.append([filename, 'FAKE'])

# Write to CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])  # Header
    writer.writerows(data)

print(f"✅ labels.csv generated with {len(data)} entries.")
