import cv2
from pathlib import Path

def extract_frames(video_path: Path, output_dir: Path, frame_step: int = 2):
    print(f"[INFO] Using video:   {video_path}")
    print(f"[INFO] Saving frames: {output_dir}")
    print(f"[INFO] Saving every {frame_step}th frame.")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_MSMF) # Flag to force Windows Media Foundation instead of GStreamer cause it caused issues 
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video FPS: {fps:.2f}")
    print(f"[INFO] Total frames: {total_frames}")
    print(f"[INFO] Expected output: ~{total_frames // frame_step} frames")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only save every Nth frame
        if frame_idx % frame_step == 0:
            frame_path = output_dir / f"frame_{saved_count:04d}.png"  # ✓ Use saved_count for sequential naming
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
            
            # Progress indicator
            if saved_count % 10 == 0:
                print(f"[PROGRESS] Saved {saved_count} frames...")
        
        frame_idx += 1
    
    cap.release()
    print(f"[DONE] Saved {saved_count} frames out of {frame_idx} total.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    video_path = BASE_DIR / "data" / "raw" / "2in2outTest.MOV"  # ✓ Updated filename
    output_dir = BASE_DIR / "data" / "frames"
    
    extract_frames(video_path, output_dir, frame_step=2)