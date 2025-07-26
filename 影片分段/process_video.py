
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from collections import defaultdict
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def detect_video(weights_path, video_path, conf_thres=0.25, target_objects=None):
    """
    Detects events in a video based on object tracking and pose estimation.
    """
    if not torch.cuda.is_available():
        print('CUDA not available. Running on CPU.')
        device = 'cpu'
    else:
        print('CUDA is available.')
        device = 'cuda'
    
    model = YOLO(weights_path)
    model.to(device)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    object_history = defaultdict(lambda: [])
    movement_threshold = 5
    history_length = 50
    
    events = [[] for _ in range(7)]
    
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1

        results = model(frame, conf=conf_thres, iou=0.1, max_det=7, device=device, verbose=False)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        
        left_wrist = None
        right_wrist = None
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            if landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.5:
                left_wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * width),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * height))
                
            if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.5:
                right_wrist = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * width),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * height))

        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()
            x1,y1,x2,y2 = map(int,box)
            cls = int(result.cls[0].item())
            
            object_name = model.names[cls]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if object_name == 'tube':
                if left_wrist or right_wrist:
                    left_distance = float('inf')
                    right_distance = float('inf')
                    
                    if left_wrist:
                        left_distance = np.sqrt((center_x - left_wrist[0])**2 + (center_y - left_wrist[1])**2)
                    if right_wrist:
                        right_distance = np.sqrt((center_x - right_wrist[0])**2 + (center_y - right_wrist[1])**2)
                    
                    if min(left_distance, right_distance) < 60:
                        if len(events[6])==0 or frame_count-events[6][-1]>150:
                            events[6].append(frame_count)
            
            if object_name == 'front_tube':
                if len(events[5]) == 0 or frame_count - events[5][-1] > 150:
                    events[5].append(frame_count)
            
            if object_name == 'hand_package':
                if left_wrist or right_wrist:
                    left_distance = float('inf')
                    right_distance = float('inf')
                    
                    if left_wrist:
                        left_distance = np.sqrt((center_x - left_wrist[0])**2 + (center_y - left_wrist[1])**2)
                    if right_wrist:
                        right_distance = np.sqrt((center_x - right_wrist[0])**2 + (center_y - right_wrist[1])**2)
                    
                    if min(left_distance, right_distance) < 80 and (len(events[4]) == 0 or frame_count - events[4][-1] > 150):
                        events[4].append(frame_count)

            if object_name in ['machine1', 'machine_2']:
                if left_wrist or right_wrist:
                    left_distance = float('inf')
                    right_distance = float('inf')
                    
                    if left_wrist:
                        left_distance = np.sqrt((center_x - left_wrist[0])**2 + (center_y - left_wrist[1])**2)
                    if right_wrist:
                        right_distance = np.sqrt((center_x - right_wrist[0])**2 + (center_y - right_wrist[1])**2)
                    
                    if min(left_distance, right_distance) < 40:
                        if object_name == 'machine1' and (len(events[2]) == 0 or frame_count - events[2][-1] > 300):
                            events[2].append(frame_count)
                        elif object_name == 'machine_2' and (len(events[3]) == 0 or frame_count - events[3][-1] > 300):
                            events[3].append(frame_count)

            if object_name == 'package':
                if left_wrist or right_wrist:
                    left_distance = float('inf')
                    right_distance = float('inf')
                    
                    if left_wrist:
                        left_distance = np.sqrt((center_x - left_wrist[0])**2 + (center_y - left_wrist[1])**2)
                    if right_wrist:
                        right_distance = np.sqrt((center_x - right_wrist[0])**2 + (center_y - right_wrist[1])**2)
                    
                    if min(left_distance, right_distance) < 60:
                        if len(events[1])==0 or frame_count-events[1][-1]>90:
                            events[1].append(frame_count)
    
    pose.close()
    cap.release()
    
    return events

def save_video_segments(video_path, events, output_dir, event_names):
    """
    Saves video segments based on event timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(output_dir, exist_ok=True)

    for i, event_frames in enumerate(events):
        if not event_frames:
            continue
        
        event_name = event_names[i]
        event_output_dir = os.path.join(output_dir, event_name)

        if os.path.exists(event_output_dir):
            for filename in os.listdir(event_output_dir):
                file_path = os.path.join(event_output_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        os.makedirs(event_output_dir, exist_ok=True)
        
        for start_frame in event_frames:
            output_path = os.path.join(event_output_dir, f"frame_{start_frame}.mp4")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            frames_count = 0
            while frames_count < 120:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_count += 1
            
            out.release()

    cap.release()

def extract_features(video_path, yolo_model, pose_model, event_name=None):
    """
    Extracts features from a video file.
    """
    keypoints_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 30:
        return None
    
    frame_indices = np.linspace(0, total_frames - 1, 30, dtype=int)
    
    feature_dim = 1 + 2 + len(keypoints_indices) * 2
    features = np.zeros((30, feature_dim))

    class_id_map = {
        'open_package': 5,
        'oxy_machine': 4,
        'suck_machine': 3,
        'suck_nose_neck_mouth': 6,
        'take_out_tube': 1,
        'wear_glove': 2,
    }
    target_class_id = class_id_map.get(event_name)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        results = yolo_model(frame, verbose=False)
        
        detected_object = 0
        object_center = [0, 0]
        
        if target_class_id is not None:
            for detection in results[0].boxes.data:
                class_id = int(detection[5])
                conf = detection[4]
                
                if class_id == target_class_id and conf > 0.25:
                    detected_object = 1
                    x1, y1, x2, y2 = detection[0:4].tolist()
                    object_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                    break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_model.process(frame_rgb)
        
        keypoints = np.zeros(len(keypoints_indices) * 2)
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            frame_height, frame_width, _ = frame.shape
            
            for j, idx in enumerate(keypoints_indices):
                if idx < len(landmarks):
                    keypoints[j*2] = landmarks[idx].x * frame_width
                    keypoints[j*2+1] = landmarks[idx].y * frame_height
        
        frame_features = np.concatenate([
            [detected_object],
            object_center,
            keypoints
        ])
        
        features[i] = frame_features
    
    cap.release()
    return features

def predict_video(video_path, yolo_model, pose_model, lstm_model, scaler, event_name=None):
    """
    Predicts whether the action in the video is correct.
    """
    features = extract_features(video_path, yolo_model, pose_model, event_name=event_name)
    
    if features is None:
        return None, None
    
    if scaler:
        features_reshaped = features.reshape(-1, features.shape[-1])
        features_scaled = scaler.transform(features_reshaped)
        features = features_scaled.reshape(features.shape)

    
    features = np.expand_dims(features, axis=0)
    
    prediction = lstm_model.predict(features)[0][0]
    
    is_correct = prediction > 0.4
    confidence = prediction if is_correct else 1 - prediction
    
    return is_correct, confidence

if __name__ == '__main__':
    weights_path = "yolov11_v4.pt"
    video_path = "C:\\Users\\jerry\\program\\graduateproject\\test\\video.mp4"
    output_dir = "process_video_output"
    target_objects = ['T-tube']
    
    event_names = [
        "T-tube_moving",
        "open_package",
        "suck_machine",
        "oxy_machine",
        "wear_glove",
        "take_out_tube",
        "suck_nose_neck_mouth"
    ]
    
    # events = detect_video(
    #     weights_path=weights_path,
    #     video_path=video_path,
    #     conf_thres=0.25,
    #     target_objects=target_objects
    # )

    # save_video_segments(video_path, events, output_dir, event_names)
    
    # Load models and scalers
    yolo_model = YOLO(weights_path)
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5)

    lstm_models = {}
    scalers = {}
    lstm_data_dir = "lstm_data"
    for event_name in event_names:
        model_path = os.path.join(lstm_data_dir, f"{event_name}_model.h5")
        scaler_path = os.path.join(lstm_data_dir, f"{event_name}_scaler_params.npy")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            lstm_models[event_name] = load_model(model_path)
            scaler_params = np.load(scaler_path, allow_pickle=True)
            scaler = StandardScaler()
            scaler.mean_ = scaler_params[0]
            scaler.scale_ = scaler_params[1]
            scalers[event_name] = scaler

    # Process videos and store results
    all_predictions = []
    for event_name in event_names:
        event_dir = os.path.join(output_dir, event_name)
        if not os.path.exists(event_dir) or event_name not in lstm_models:
            continue

        for video_file in os.listdir(event_dir):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(event_dir, video_file)
                is_correct, confidence = predict_video(
                    video_path,
                    yolo_model,
                    pose_model,
                    lstm_models[event_name],
                    scalers[event_name],
                    event_name=event_name
                )
                if is_correct is not None:
                    all_predictions.append({
                        "event": event_name,
                        "video": video_file,
                        "is_correct": is_correct,
                        "confidence": confidence
                    })

    pose_model.close()

    # --- Final Report ---
    print("\n--- Video Processing Report ---")

    # Detailed predictions
    print("\n--- Detailed Predictions ---")
    for pred in all_predictions:
        status = "Correct" if pred['is_correct'] else "Incorrect"
        print(f"Event: {pred['event']:<25} | Video: {pred['video']:<20} | Prediction: {status:<10} | Confidence: {pred['confidence']:.4f}")

    # Accuracy per event
    print("\n--- Accuracy per Event ---")
    for event_name in event_names:
        event_predictions = [p for p in all_predictions if p['event'] == event_name]
        if not event_predictions:
            continue
        
        correct_count = sum(1 for p in event_predictions if p['is_correct'])
        total_count = len(event_predictions)
        accuracy = (correct_count / total_count) * 100
        
        print(f"Event: {event_name:<25} | Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

    # Best performance video per event
    print("\n--- Best Performance per Event ---")
    for event_name in event_names:
        correct_predictions = [p for p in all_predictions if p['event'] == event_name and p['is_correct']]
        if not correct_predictions:
            print(f"Event: {event_name:<25} | No correctly predicted videos found.")
            continue

        best_video = max(correct_predictions, key=lambda x: x['confidence'])
        print(f"Event: {best_video['event']:<25} | Best Video: {best_video['video']:<20} | Confidence: {best_video['confidence']:.4f}")


    print("\n--- End of Report ---")
