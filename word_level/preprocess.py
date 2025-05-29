import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd 
from tqdm import tqdm
import json
from collections import defaultdict

TARGET_GLOSSES = {"computer", "drink"}


DATA_PATH = "./v1/raw_videos" # Replace with your actual data path
OUTPUT_PATH = "./v1/processed_data" # Folder to save processed data
SEQUENCE_LENGTH = 60  # Number of frames per sequence (video clip)
NUM_LANDMARKS_POSE = 33 # MediaPipe Pose uses 33 landmarks

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def normalize_landmarks(landmarks_mp, image_shape):
    """
    Normalizes pose landmarks.

    Returns a flat numpy array of (x, y) coordinates for all landmarks
    """
    if not landmarks_mp:
        return None

    landmarks = landmarks_mp.landmark

    # center landmarks around nose (translation invariance)
    try:
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    except IndexError: # Should not happen with 33 landmarks if pose is detected
        print("Warning: Nose landmark not found.")
        return None 

    relative_landmarks = []
    for lm in landmarks:
        relative_landmarks.extend([lm.x - nose_x, lm.y - nose_y])

    # normalize by shoulder width (scale invariance) 
    try:
        left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    except IndexError:
        print("Warning: Shoulder landmarks not found.")
        return None # Cannot normalize scale without shoulder landmarks

    shoulder_width = np.sqrt(
        (right_shoulder_x - left_shoulder_x)**2 + \
        (right_shoulder_y - left_shoulder_y)**2
    )

    # use a small epsilon to prevent division by zero if shoulders are at the same point (person facing sideways or bad detection)
    if shoulder_width < 1e-6: 
        return None

    normalized_landmarks = np.array(relative_landmarks) / shoulder_width

    return normalized_landmarks.flatten() # Flatten to a 1D array (33 * 2 = 66 features)


def extract_features_from_video(video_path, pose_estimator, seq_length):
    """
    Extracts features from a provided video.
    
    Returns a np array corresponding to # of frames being analyzed. Each entry contains hand landmark data (from Mediapipe)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    frames_features = []
    frame_count = 0

    while cap.isOpened() and frame_count < seq_length * 2: 
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # To improve performance

        # Process the image and detect pose
        results = pose_estimator.process(image_rgb)
        image_rgb.flags.writeable = True 

        if results.pose_landmarks:
            normalized_data = normalize_landmarks(results.pose_landmarks, frame.shape)
            if normalized_data is not None:
                frames_features.append(normalized_data)
            else:
                # If normalization fails (e.g. key points not visible), append zeros for this frame
                frames_features.append(np.zeros(NUM_LANDMARKS_POSE * 2))
        else:
            # If no pose detected, append zeros for this frame
            frames_features.append(np.zeros(NUM_LANDMARKS_POSE * 2))

        frame_count += 1

    cap.release()

    # Pad or truncate the sequence for consistent length
    num_features = NUM_LANDMARKS_POSE * 2
    if not frames_features: 
        print(f"Warning: No features extracted from {video_path}. Returning zeros.")
        return np.zeros((seq_length, num_features))

    if len(frames_features) < seq_length:
        # Pad with zeros at the end
        padding = np.zeros((seq_length - len(frames_features), num_features))
        processed_sequence = np.vstack((np.array(frames_features), padding))
    else:
        # Truncate (take the first seq_length frames)
        processed_sequence = np.array(frames_features[:seq_length])

    return processed_sequence


def main_preprocess():
    """
    Main function to iterate through WLASL data, preprocess videos, and save features.
    """
    # Load the JSON file
    with open('./WLASL_v0.3.json', 'r') as f:
        data = json.load(f)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    all_sequences = []
    all_labels = []
    gloss_to_idx = {}
    current_label = 0
    gloss_to_files = defaultdict(list)

    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        for entry in data:
            gloss = entry['gloss']
            if TARGET_GLOSSES and gloss not in TARGET_GLOSSES:
                continue

            if gloss not in gloss_to_idx:
                gloss_to_idx[gloss] = current_label
                current_label += 1

            label_idx = gloss_to_idx[gloss]
            instances = entry.get('instances', [])

            print(f"\nProcessing gloss: {gloss} (Label: {label_idx})")

            for inst in tqdm(instances, desc=f"Videos for {gloss}"):
                video_id = inst.get('video_id')
                if not video_id:
                    continue

                video_file = f"{video_id}.mp4"
                video_path = os.path.join(DATA_PATH, video_file)

                if not os.path.isfile(video_path):
                    print(f"Missing file: {video_path}")
                    continue

                sequence_features = extract_features_from_video(video_path, pose, SEQUENCE_LENGTH)
                if sequence_features is not None:
                    all_sequences.append(sequence_features)
                    all_labels.append(label_idx)
                    gloss_to_files[gloss].append(video_file)
                else:
                    print(f"Warning: Could not process video {video_path}")

    if not all_sequences:
        print("No sequences were processed. Check your paths and data.")
        return

    # Save the results
    X = np.array(all_sequences)
    y = np.array(all_labels)

    print(f"\n--- Preprocessing Complete ---")
    print(f"Shape of X (sequences): {X.shape}")
    print(f"Shape of y (labels): {y.shape}")

    np.save(os.path.join(OUTPUT_PATH, "sequences.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "labels.npy"), y)

    # Save the label-to-gloss mapping
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    with open(os.path.join(OUTPUT_PATH, "words_list.txt"), "w") as f:
        for i in range(len(idx_to_gloss)):
            f.write(f"{idx_to_gloss[i]}\n")

    with open(os.path.join(OUTPUT_PATH, "processed_files.json"), "w") as f:
        json.dump(gloss_to_files, f, indent=2)

    print(f"Processed data saved to: {OUTPUT_PATH}")
    print("Files: sequences.npy, labels.npy, words_list.txt")



if __name__ == '__main__':
    main_preprocess()