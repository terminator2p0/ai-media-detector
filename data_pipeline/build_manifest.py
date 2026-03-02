import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    base_dir = "data/processed/cropped_faces"
    data = []
    
    # 0 = Real, 1 = AI Generated
    categories = [("real", 0), ("fake", 1)]
    
    print("Scanning cropped faces to build dataset manifest...")
    for category, label in categories:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir): continue
        
        for video_folder in os.listdir(cat_dir):
            vid_path = os.path.join(cat_dir, video_folder)
            if not os.path.isdir(vid_path): continue
                
            for img_file in os.listdir(vid_path):
                if img_file.endswith('.jpg'):
                    # Save the relative path so the dataloader can find it easily
                    rel_path = os.path.join(category, video_folder, img_file)
                    data.append({"image_path": rel_path, "label": label})
                    
    df = pd.DataFrame(data)
    if len(df) == 0:
        print("Error: No cropped faces found. Did the MTCNN script run successfully?")
        return
        
    print(f"Found {len(df)} total valid face frames.")
    
    # Split into 80% Train, 10% Validation, 10% Test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # Save the CSV files
    train_df.to_csv("data/train_manifest.csv", index=False)
    val_df.to_csv("data/val_manifest.csv", index=False)
    test_df.to_csv("data/test_manifest.csv", index=False)
    
    print("Manifests successfully generated in data/ folder!")
    print(f"Train set: {len(train_df)} | Validation set: {len(val_df)} | Test set: {len(test_df)}")

if __name__ == "__main__":
    main()