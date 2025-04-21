import cv2
import os
from natsort import natsorted

def create_video_from_images(image_folder, output_video_path, fps=5):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = natsorted(images)  # Sort naturally by filename

    if not images:
        print("❌ No images found in folder.")
        return

    # Read first image to get size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"❌ Failed to read the first image: {first_image_path}")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'mp4v'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            out.write(frame)
        else:
            print(f"⚠️ Skipped unreadable image: {img_name}")

    out.release()
    print(f"✅ Video created at: {output_video_path}")

if __name__ == "__main__":
    predicted_folder = "../utils/predicted_image"
    output_video = "../utils/predicted_video.mp4"
    create_video_from_images(predicted_folder, output_video, fps=5)
