import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

def process_faces(category, input_base, output_base, mtcnn):
    input_dir = os.path.join(input_base, category)
    output_dir = os.path.join(output_base, category)
    
    if not os.path.exists(input_dir): return
    
    video_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    for video_folder in tqdm(video_folders, desc=f"Cropping {category} faces"):
        vid_input_path = os.path.join(input_dir, video_folder)
        vid_output_path = os.path.join(output_dir, video_folder)
        os.makedirs(vid_output_path, exist_ok=True)
        
        frames = [f for f in os.listdir(vid_input_path) if f.endswith('.jpg')]
        for frame_file in frames:
            img_path = os.path.join(vid_input_path, frame_file)
            save_path = os.path.join(vid_output_path, frame_file)
            
            try:
                img = Image.open(img_path)
                # Detect and save the cropped face
                mtcnn(img, save_path=save_path)
            except Exception:
                # MTCNN throws an error if no face is detected; we safely skip these frames
                pass

def main():
    # UPDATED: Removed 'mps' to bypass the Apple Silicon bug.
    # It will use 'cuda' on GCP later, but 'cpu' locally on your Mac.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running MTCNN face detection on: {device}")
    
    mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)
    
    input_dir = "data/processed/video_frames"
    output_dir = "data/processed/cropped_faces"
    
    process_faces("real", input_dir, output_dir, mtcnn)
    process_faces("fake", input_dir, output_dir, mtcnn)
    print("Face cropping complete!")

if __name__ == "__main__":
    main()