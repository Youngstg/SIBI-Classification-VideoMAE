import os
import cv2

# Calculate average video duration in a folder
folder = "Dataset/dataset_bersih/Ibu"
durations = []

# Extract duration from each video file
for f in os.listdir(folder):
    path = os.path.join(folder, f)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Total frames
    if fps > 0:
        # Duration = total_frames / fps
        durations.append(frames / fps)

# Print average duration
print("Rata rata durasi:", sum(durations) / len(durations))