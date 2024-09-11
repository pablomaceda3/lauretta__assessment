import os
import torch
import cv2 as cv
import time
import numpy as np

from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip


def download_youtube_video(youtube_url: str, save_path: str = "main_video.mp4") -> str:
    """Download a YouTube video and save it to the ./main_video.mp4 file."""
    ydl_opts = {'format': 'best', 'outtmpl': save_path}

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    return save_path

def trim_video(input_path: str, start_time: int, end_time: int, output_path: str) -> None:
    """Trim the video to the specified time range using moviepy"""
    clip = VideoFileClip(input_path).subclip(start_time, end_time)
    clip.write_videofile(output_path, codec='libx264')


def detect_people_in_video(video_path: str) -> None:
    """Detect and count each person in the frame from a given video."""

    # Load the YouOnlyLookOnceV5 model, a state-of-the-art 
    # model used for object detection, with custom weights 
    # used for detecting people at a high, aerial level view.
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./crowdhuman_yolov5m.pt')

    # Open the video file
    capture = cv.VideoCapture(video_path)

    total_people = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        results = model(frame)

        # Extract all detected classes from the results
        detected_classes = results.xyxy[0]

        # Extract all detected people. For results,
        # the data is in the format [xmin, xmax, ymin, ymax, confidence_score, class],
        # with class == 0 depicting a person.
        num_people = sum([1 for cls in detected_classes if int(cls[-1]) == 0])

        total_people += num_people

        cv.putText(frame, f'Total people count: {total_people}', (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('Frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    time.sleep(5)
    cv.destroyAllWindows()
    return total_people


def main(video_path: str, start_time: int, end_time: int) -> None:
    trimmed_path = "trimmed_video.mp4"

    trim_video(video_path, start_time, end_time, trimmed_path)

    total_people = detect_people_in_video(trimmed_path)

    # Cleanup
    if os.path.exists(trimmed_path):
        os.remove(trimmed_path)

    return total_people

def validate(estimates: list[int]) -> None:
    all_counts = np.array(estimates)

    mean = np.mean(all_counts)
    var = np.var(all_counts)

    print(f"MEAN COUNT: {mean}")
    print(f"VAR: {var}")

    coeff_var = (np.std(all_counts) / mean) * 100
    print(f"COEFFICIENT OF VARIATION: {coeff_var}%")

    corr_12 = np.corrcoef(all_counts[0], all_counts[1])[0, 1]
    corr_13 = np.corrcoef(all_counts[0], all_counts[2])[0, 1]
    corr_23 = np.corrcoef(all_counts[1], all_counts[2])[0, 1]

    print(f"Correlation between Window 1 and 2: {corr_12}")
    print(f"Correlation between Window 1 and 3: {corr_13}")
    print(f"Correlation between Window 2 and 3: {corr_23}")

def cleanup(video_path):
    if os.path.exists(video_path):
        os.remove(video_path)

if __name__ == "__main__":
    youtube_url = "https://youtu.be/y2zyucfCyjM?si=YaQc2CJFg5X9xEhr"
    video_path = download_youtube_video(youtube_url)

    # 18-second window intervals
    start_times_and_end_times = [[14, 32], [32, 50], [50, 68]]
    estimates = []

    for start_time, end_time in start_times_and_end_times:
        total_people = main(video_path, start_time=start_time, end_time=end_time)
        estimates.append(total_people)

    validate(estimates)
    cleanup(video_path)
