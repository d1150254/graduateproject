import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
import mediapipe as mp
import re

def extract_features_from_frame(video_path, yolo_model, pose):
    """
    從4秒鐘30fps的影片中提取30個frame的特徵資料
    
    Args:
        video_path (str): 影片檔案路徑
        yolo_model: 預先載入的YOLO模型
        pose: 預先載入的MediaPipe姿勢估計模型
    
    Returns:
        np.array: 特徵資料的numpy陣列,形狀為 (30, feature_dim)
                 feature_dim 包含機器偵測布林值(1)、機器2中心點座標(2)、關鍵骨架點座標(24)
    """
    # 需要提取的骨架關鍵點索引
    keypoints_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
    # 打開影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片：{video_path}")
    
    # 獲取影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"影片總幀數: {total_frames}, 幀率: {fps}")
    
    # 確保我們有足夠的幀
    if total_frames < 30:
        raise ValueError(f"影片幀數不足30: {total_frames}")
    
    # 計算要抽取的幀的索引 (均勻分佈在整個影片中)
    frame_indices = np.linspace(0, total_frames-1, 30, dtype=int)
    print(f"將抽取這些幀: {frame_indices}")
    
    # 初始化特徵資料陣列
    # 每個特徵包含：機器2偵測布林值(1) + 機器2中心點座標(2) + 12個關鍵點的xy座標(24)
    feature_dim = 1 + 2 + len(keypoints_indices) * 2
    features = np.zeros((30, feature_dim))
    
    # 對每個選定的幀進行處理
    for i, frame_idx in enumerate(frame_indices):
        # 設置讀取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"無法讀取幀 {frame_idx}")
            # 如果讀取失敗，使用零向量填充
            features[i] = np.zeros(feature_dim)
            continue
        
        # 1. 使用YOLO偵測物體
        results = yolo_model(frame, verbose=False)
        
        # 預設值：未偵測到機器2
        machine_2_detected = 0
        machine_2_center = [0, 0]
        
        # 檢查是否偵測到machine_2
        for detection in results[0].boxes.data:
            class_id = int(detection[5])
            conf = detection[4]
            
            # 假設要得的class_id是1，請根據您的模型調整
            if class_id == 3 and conf > 0.25:  # 偵測到要的class且信心度大於0.25
                machine_2_detected = 1
                # 獲取邊界框座標 [x1, y1, x2, y2]
                x1, y1, x2, y2 = detection[0:4].tolist()
                # 計算中心點
                machine_2_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                break
        
        # 2. 使用MediaPipe偵測人體姿勢
        # 轉換為RGB格式（MediaPipe需要）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        
        # 初始化關鍵點座標
        keypoints = np.zeros(len(keypoints_indices) * 2)
        
        # 如果偵測到姿勢
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            frame_height, frame_width, _ = frame.shape
            
            # 提取所需的關鍵點座標
            for j, idx in enumerate(keypoints_indices):
                if idx < len(landmarks):
                    keypoints[j*2] = landmarks[idx].x * frame_width      # x座標
                    keypoints[j*2+1] = landmarks[idx].y * frame_height   # y座標
        
        # 將所有特徵合併
        frame_features = np.concatenate([
            [machine_2_detected],          # 機器2偵測布林值
            machine_2_center,              # 機器2中心點座標
            keypoints                      # 關鍵骨架點座標
        ])
        
        # 儲存該幀的特徵
        features[i] = frame_features
    
    # 關閉影片
    cap.release()
    
    return features

def extract_frame_number(filename):
    """
    從檔案名稱中提取frame數字
    例如：'frame_1234.mp4' -> 1234
    
    Args:
        filename (str): 檔案名稱
        
    Returns:
        int or None: 提取到的frame數字,如果提取失敗則返回None
    """
    match = re.search(r'frame_(\d+)\.mp4', filename)
    if match:
        return int(match.group(1))
    return None

def process_videos_in_folder(folder_path, positive_frames, output_path=None):
    """
    處理資料夾中的所有影片並提取特徵，同時生成標籤
    
    Args:
        folder_path (str): 含有影片的資料夾路徑
        positive_frames (list): 正樣本的幀數列表，格式為 [1234, 5678, ...]
        output_path (str, optional): 輸出特徵檔案的路徑前綴,默認為None
    
    Returns:
        tuple: (X_features, y_labels, filenames) 元組，包含特徵、標籤和對應的檔案名
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"資料夾不存在: {folder_path}")
    
    # 載入YOLO模型 - 移到這裡只載入一次
    print("載入YOLO模型...")
    yolo_model = YOLO('yolov11_v4.pt')
    
    # 初始化MediaPipe姿勢估計 - 也只初始化一次
    print("初始化MediaPipe姿勢估計模型...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, f)) and 
                  any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"在 {folder_path} 中未找到影片檔案")
        return np.array([]), np.array([]), []
    
    # 初始化列表用於存儲特徵和標籤
    X_features = []
    y_labels = []
    processed_files = []
    
    for video_file in video_files:
        print(f"處理影片: {video_file}")
        video_path = os.path.join(folder_path, video_file)
        
        try:
            # 提取特徵 - 傳入已載入的模型
            features = extract_features_from_frame(video_path, yolo_model, pose)
            
            # 提取frame數字並判斷標籤
            frame_number = extract_frame_number(video_file)
            
            if frame_number is not None:
                # 如果frame_number在positive_frames列表中，標記為1，否則為0
                label = 1 if frame_number in positive_frames else 0
                
                X_features.append(features)
                y_labels.append(label)
                processed_files.append(video_file)
                
                print(f"成功提取 {video_file} 的特徵，形狀: {features.shape}, 標籤: {label}")
            else:
                print(f"無法從 {video_file} 提取frame數字,跳過此檔案")
                
        except Exception as e:
            print(f"處理 {video_file} 時發生錯誤: {str(e)}")
    
    # 關閉 MediaPipe 姿勢估計模型
    pose.close()
    
    # 將列表轉換為numpy陣列
    X_features = np.array(X_features) if X_features else np.array([])
    y_labels = np.array(y_labels) if y_labels else np.array([])
    
    # 如果指定了輸出路徑，儲存特徵和標籤
    if output_path:
        np.save(f"{output_path}_features.npy", X_features)
        np.save(f"{output_path}_labels.npy", y_labels)
        # 儲存檔案名稱-標籤映射，以便後續查看
        file_label_map = {f: l for f, l in zip(processed_files, y_labels)}
        np.save(f"{output_path}_file_labels.npy", file_label_map)
        
        print(f"特徵已儲存至: {output_path}_features.npy")
        print(f"標籤已儲存至: {output_path}_labels.npy")
        print(f"檔案-標籤映射已儲存至: {output_path}_file_labels.npy")
    
    return X_features, y_labels, processed_files

# 使用範例
if __name__ == "__main__":
    # 設定影片資料夾路徑
    video_folder = "train_suck_machine"  # 包含影片的資料夾路徑
    output_file = "lstm_data"
    
    # 指定要標記為1的幀數列表
    positive_frames =  [1298,1192,2824,3180,3498,3197,2682,1213,2066,2398,4142,4200] # 這些幀數的影片會被標記為1，其他為0
    
    # 處理資料夾中的所有影片
    X_features, y_labels, processed_files = process_videos_in_folder(
        video_folder, positive_frames, output_file)
    
    # 顯示提取到的特徵和標籤數量
    print(f"共處理了 {len(processed_files)} 個影片")
    print(f"特徵形狀: {X_features.shape if len(X_features) > 0 else '空'}")
    print(f"標籤形狀: {y_labels.shape if len(y_labels) > 0 else '空'}")
    print(f"正樣本數量: {np.sum(y_labels) if len(y_labels) > 0 else 0}")
    print(f"負樣本數量: {len(y_labels) - np.sum(y_labels) if len(y_labels) > 0 else 0}")
    
    # 顯示每個檔案的標籤
    for filename, label in zip(processed_files, y_labels):
        print(f"{filename}: {'正樣本' if label == 1 else '負樣本'}")