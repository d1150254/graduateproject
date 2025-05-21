import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from ultralytics import YOLO
import mediapipe as mp

def extract_features(video_path, yolo_model, pose_model):
    """從影片中提取特徵"""
    # 需要提取的骨架關鍵點索引
    keypoints_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
    # 打開影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片：{video_path}")
        return None
    
    # 獲取影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 30:
        print(f"影片幀數不足30: {total_frames}")
        return None
    
    # 計算要抽取的幀的索引 (均勻分佈在整個影片中)
    frame_indices = np.linspace(0, total_frames-1, 30, dtype=int)
    
    # 初始化特徵資料陣列
    feature_dim = 1 + 2 + len(keypoints_indices) * 2
    features = np.zeros((30, feature_dim))
    
    # 對每個選定的幀進行處理
    for i, frame_idx in enumerate(frame_indices):
        # 設置讀取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"無法讀取幀 {frame_idx}")
            features[i] = np.zeros(feature_dim)
            continue
        
        # 1. 使用YOLO偵測物體
        results = yolo_model(frame, verbose=False)
        
        # 預設值：未偵測到機器2
        machine_2_detected = 0
        machine_2_center = [0, 0]
        
        # 檢查是否偵測到machine_2 (class_id=4)
        for detection in results[0].boxes.data:
            class_id = int(detection[5])
            conf = detection[4]
            
            if class_id == 3 and conf > 0.25:
                machine_2_detected = 1
                x1, y1, x2, y2 = detection[0:4].tolist()
                machine_2_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                break
        
        # 2. 使用MediaPipe偵測人體姿勢
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_model.process(frame_rgb)
        
        # 初始化關鍵點座標
        keypoints = np.zeros(len(keypoints_indices) * 2)
        
        # 如果偵測到姿勢
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            frame_height, frame_width, _ = frame.shape
            
            # 提取所需的關鍵點座標
            for j, idx in enumerate(keypoints_indices):
                if idx < len(landmarks):
                    keypoints[j*2] = landmarks[idx].x * frame_width
                    keypoints[j*2+1] = landmarks[idx].y * frame_height
        
        # 將所有特徵合併
        frame_features = np.concatenate([
            [machine_2_detected],
            machine_2_center,
            keypoints
        ])
        
        features[i] = frame_features
    
    cap.release()
    return features

def predict_video(video_path, model_path, scaler_path=None):
    """預測影片是否為正確動作"""
    # 載入模型
    print("載入LSTM模型...")
    model = load_model(model_path)
    
    # 載入縮放參數（如果有）
    scaler_params = None
    if scaler_path and os.path.exists(scaler_path):
        print("載入特徵縮放參數...")
        try:
            scaler_params = np.load(scaler_path, allow_pickle=True)
        except:
            print("無法載入特徵縮放參數，將使用原始特徵")
    
    # 載入YOLO模型
    print("載入YOLO模型...")
    yolo_model = YOLO('yolov11_v4.pt')
    
    # 初始化MediaPipe姿勢估計
    print("初始化MediaPipe姿勢估計...")
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5)
    
    try:
        # 提取特徵
        print(f"從影片提取特徵: {video_path}")
        features = extract_features(video_path, yolo_model, pose_model)
        
        if features is None:
            print("特徵提取失敗")
            return
        
        # 應用特徵縮放（如果有縮放參數）
        if scaler_params is not None:
            X_mean, X_std = scaler_params
            features = (features - X_mean) / X_std
        
        # 擴展維度以符合模型輸入格式
        features = np.expand_dims(features, axis=0)
        
        # 進行預測
        print("進行預測...")
        prediction = model.predict(features)[0][0]
        

        print(f"預測結果: {prediction:.4f}")
        # 輸出結果
        is_correct = prediction > 0.3
        confidence = prediction if is_correct else 1 - prediction

        print(f"信心度: {confidence:.4f}")
        
        print("/n" + "="*50)
        if is_correct:
            print(f"✅ 這是正確的動作 (信心度: {confidence:.4f})")
        else:
            print(f"❌ 這不是正確的動作 (信心度: {confidence:.4f})")
        print("="*50 + "/n")
        
    finally:
        # 釋放資源
        pose_model.close()

if __name__ == "__main__":
    # 載入設定
    model_path = ""  # 訓練好的模型路徑
    scaler_path = ""  # 特徵縮放參數路徑（可選）
    
    # 接受使用者輸入的影片路徑
    video_path = ""
    
    if not os.path.exists(video_path):
        print(f"找不到影片: {video_path}")
    else:
        predict_video(video_path, model_path, scaler_path=scaler_path)


# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os
# import time
# from ultralytics import YOLO
# import mediapipe as mp
# import glob
# from pathlib import Path

# def extract_features(video_path, yolo_model, pose_model):
#     """從影片中提取特徵"""
#     # 需要提取的骨架關鍵點索引
#     keypoints_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
#     # 打開影片
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"無法開啟影片：{video_path}")
#         return None
    
#     # 獲取影片資訊
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     if total_frames < 30:
#         print(f"影片幀數不足30: {total_frames}")
#         return None
    
#     # 計算要抽取的幀的索引 (均勻分佈在整個影片中)
#     frame_indices = np.linspace(0, total_frames-1, 30, dtype=int)
    
#     # 初始化特徵資料陣列
#     feature_dim = 1 + 2 + len(keypoints_indices) * 2
#     features = np.zeros((30, feature_dim))
    
#     # 對每個選定的幀進行處理
#     for i, frame_idx in enumerate(frame_indices):
#         # 設置讀取位置
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, frame = cap.read()
        
#         if not ret:
#             print(f"無法讀取幀 {frame_idx}")
#             features[i] = np.zeros(feature_dim)
#             continue
        
#         # 1. 使用YOLO偵測物體
#         results = yolo_model(frame, verbose=False)
        
#         # 預設值：未偵測到機器2
#         machine_2_detected = 0
#         machine_2_center = [0, 0]
        
#         # 檢查是否偵測到machine_2 (class_id=4)
#         for detection in results[0].boxes.data:
#             class_id = int(detection[5])
#             conf = detection[4]
            
#             if class_id == 4 and conf > 0.25:
#                 machine_2_detected = 1
#                 x1, y1, x2, y2 = detection[0:4].tolist()
#                 machine_2_center = [(x1 + x2) / 2, (y1 + y2) / 2]
#                 break
        
#         # 2. 使用MediaPipe偵測人體姿勢
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pose_results = pose_model.process(frame_rgb)
        
#         # 初始化關鍵點座標
#         keypoints = np.zeros(len(keypoints_indices) * 2)
        
#         # 如果偵測到姿勢
#         if pose_results.pose_landmarks:
#             landmarks = pose_results.pose_landmarks.landmark
#             frame_height, frame_width, _ = frame.shape
            
#             # 提取所需的關鍵點座標
#             for j, idx in enumerate(keypoints_indices):
#                 if idx < len(landmarks):
#                     keypoints[j*2] = landmarks[idx].x * frame_width
#                     keypoints[j*2+1] = landmarks[idx].y * frame_height
        
#         # 將所有特徵合併
#         frame_features = np.concatenate([
#             [machine_2_detected],
#             machine_2_center,
#             keypoints
#         ])
        
#         features[i] = frame_features
    
#     cap.release()
#     return features

# def predict_single_video(video_path, model, yolo_model, pose_model, scaler_params=None, threshold=0.3):
#     """預測單個影片是否為正確動作"""
#     try:
#         # 提取特徵
#         features = extract_features(video_path, yolo_model, pose_model)
        
#         if features is None:
#             return None, None
        
#         # 應用特徵縮放（如果有縮放參數）
#         if scaler_params is not None:
#             X_mean, X_std = scaler_params
#             features = (features - X_mean) / X_std
        
#         # 擴展維度以符合模型輸入格式
#         features = np.expand_dims(features, axis=0)
        
#         # 進行預測
#         prediction = model.predict(features, verbose=0)[0][0]
        
#         # 輸出結果
#         is_correct = prediction > threshold
#         confidence = prediction if is_correct else 1 - prediction
        
#         return is_correct, prediction
    
#     except Exception as e:
#         print(f"預測影片時發生錯誤: {str(e)}")
#         return None, None

# def test_folder(folder_path, model_path, scaler_path=None, threshold=0.3):
#     """測試資料夾中的所有影片"""
#     print(f"開始測試資料夾: {folder_path}")
#     print(f"使用閾值: {threshold}")
    
#     # 檢查資料夾是否存在
#     if not os.path.exists(folder_path):
#         print(f"找不到資料夾: {folder_path}")
#         return
    
#     # 載入模型
#     print("載入LSTM模型...")
#     model = load_model(model_path)
    
#     # 載入縮放參數（如果有）
#     scaler_params = None
#     if scaler_path and os.path.exists(scaler_path):
#         print("載入特徵縮放參數...")
#         try:
#             scaler_params = np.load(scaler_path, allow_pickle=True)
#         except:
#             print("無法載入特徵縮放參數，將使用原始特徵")
    
#     # 載入YOLO模型
#     print("載入YOLO模型...")
#     yolo_model = YOLO('yolov11_v4.pt')
    
#     # 初始化MediaPipe姿勢估計
#     print("初始化MediaPipe姿勢估計...")
#     mp_pose = mp.solutions.pose
#     pose_model = mp_pose.Pose(
#         static_image_mode=False,
#         model_complexity=1,
#         enable_segmentation=False,
#         min_detection_confidence=0.5)
    
#     # 取得資料夾中的所有影片
#     video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
#     video_files = []
    
#     for ext in video_extensions:
#         video_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    
#     print(f"在資料夾中找到 {len(video_files)} 個影片檔")
    
#     # 如果沒有找到影片
#     if not video_files:
#         print("未找到任何影片，請檢查資料夾路徑是否正確")
#         pose_model.close()
#         return
    
#     # 結果列表
#     results = []
#     positive_count = 0
#     negative_count = 0
    
#     # 新增：正確和錯誤動作的幀數陣列
#     correct_frames = []
#     incorrect_frames = []
    
#     # 從文件名提取幀數的函數
#     def extract_frame_number(filename):
#         import re
#         match = re.search(r'frame_(\d+)\.mp4', filename)
#         if match:
#             return int(match.group(1))
#         return None
    
#     # 測試開始時間
#     start_time = time.time()
    
#     # 逐一測試影片
#     for i, video_path in enumerate(video_files):
#         video_name = Path(video_path).name
#         print(f"\n[{i+1}/{len(video_files)}] 測試影片: {video_name}")
        
#         # 提取幀數
#         frame_number = extract_frame_number(video_name)
        
#         is_correct, prediction = predict_single_video(
#             video_path, model, yolo_model, pose_model, scaler_params, threshold)
        
#         if is_correct is not None and frame_number is not None:
#             result_str = "✅ 正確" if is_correct else "❌ 不正確"
#             confidence = prediction if is_correct else 1 - prediction
            
#             if is_correct:
#                 positive_count += 1
#                 correct_frames.append(frame_number)  # 添加到正確動作陣列
#             else:
#                 negative_count += 1
#                 incorrect_frames.append(frame_number)  # 添加到錯誤動作陣列
            
#             print(f"預測結果: {result_str} (原始值: {prediction:.4f}, 信心度: {confidence:.4f})")
            
#             # 儲存結果
#             results.append({
#                 'video_path': video_path,
#                 'video_name': video_name,
#                 'frame_number': frame_number,
#                 'is_correct': is_correct,
#                 'prediction': prediction,
#                 'confidence': confidence
#             })
#         else:
#             print(f"無法預測影片: {video_name}")
    
#     # 釋放資源
#     pose_model.close()
    
#     # 測試結束時間
#     end_time = time.time()
#     total_time = end_time - start_time
    
#     # 輸出總結報告
#     print("\n" + "="*50)
#     print("測試完成")
#     print(f"共測試 {len(video_files)} 個影片，耗時 {total_time:.2f} 秒")
#     print(f"預測為正確動作的影片: {positive_count} 個")
#     print(f"預測為不正確動作的影片: {negative_count} 個")
    
#     # 排序幀數陣列
#     correct_frames.sort()
#     incorrect_frames.sort()
    
#     print("\n正確動作的影片幀數列表:")
#     print(correct_frames)
    
#     print("\n錯誤動作的影片幀數列表:")
#     print(incorrect_frames)
    
#     # 如果有至少一個成功預測
#     if results:
#         # 找出預測信心度最高的正確和不正確影片
#         correct_videos = [r for r in results if r['is_correct']]
#         incorrect_videos = [r for r in results if not r['is_correct']]
        
#         if correct_videos:
#             top_correct = max(correct_videos, key=lambda x: x['confidence'])
#             print(f"\n信心度最高的正確動作影片: {top_correct['video_name']} (信心度: {top_correct['confidence']:.4f})")
        
#         if incorrect_videos:
#             top_incorrect = max(incorrect_videos, key=lambda x: x['confidence'])
#             print(f"信心度最高的不正確動作影片: {top_incorrect['video_name']} (信心度: {top_incorrect['confidence']:.4f})")
    
#     print("="*50)
    
#     # 返回正確和錯誤動作的幀數陣列和結果
#     return correct_frames, incorrect_frames, results

# if __name__ == "__main__":
#     # 設定參數
#     model_path = "lstm_model.h5"  # 訓練好的模型路徑
#     scaler_path = "scaler_params.npy"  # 特徵縮放參數路徑
#     folder_path = "C:/Users/jerry/program/graduateproject/test/影片分段/train_oxy_machine"  # 或直接指定路徑
#     threshold = 0.3  # 分類閾值
    
#     # 執行測試
#     correct_frames, incorrect_frames, results = test_folder(folder_path, model_path, scaler_path, threshold)
    
#     # 輸出可以直接複製到程式碼中的陣列格式
#     print("\n可直接複製的正確動作幀數陣列:")
#     print(f"correct_frames = {correct_frames}")
    
#     print("\n可直接複製的錯誤動作幀數陣列:")
#     print(f"incorrect_frames = {incorrect_frames}")