import os
import cv2
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir, frames_per_second=1):
    video_name = os.path.basename(video_path).split('.')[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    
    fps = max(1, round(cap.get(cv2.CAP_PROP_FPS)))
    frame_interval = max(1, fps // frames_per_second)

    frame_count, saved_count = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % frame_interval == 0:
            file_path = os.path.join(video_output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(file_path, frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    return saved_count

def process_folder(category, raw_base, processed_base):
    input_dir = os.path.join(raw_base, category)
    output_dir = os.path.join(processed_base, category)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir): 
        print(f"Skipping {category}: Directory not found.")
        return
        
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    for video_file in tqdm(video_files, desc=f"Extracting {category} videos"):
        video_path = os.path.join(input_dir, video_file)
        extract_frames_from_video(video_path, output_dir)

def main():
    raw_video_dir = "data/raw/videos"
    output_frames_dir = "data/processed/video_frames"
    
    print("Starting frame extraction pipeline...")
    process_folder("real", raw_video_dir, output_frames_dir)
    process_folder("fake", raw_video_dir, output_frames_dir)
    print("Frame extraction complete!")

if __name__ == "__main__":
    main()