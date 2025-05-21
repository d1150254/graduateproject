from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
from collections import defaultdict  # 用於追蹤物件歷史位置
import mediapipe as mp  # 新增 mediapipe 導入

def detect_video(weights_path, video_path, output_path, conf_thres=0.25, target_objects=None, visual=True): # 新增 visual 參數，預設為 True
    # 0=T-tube moving 1=package hold in hand
    # 2=machine1 touched 3=machine_2 touched
    # 4=hand_package touched 5=front_tube detected
    # 6=tube touched
    test=[[] for _ in range(7)]  # 擴充 test 列表以記錄 front_tube 偵測
    if not torch.cuda.is_available():
        raise Exception('CUDA not available. Please check your CUDA installation.')
    else:
        print('CUDA is available.')
    # 載入自訓練的模型
    model = YOLO(weights_path)
    model.to('cuda')
    
    # 初始化 MediaPipe 姿勢偵測
    mp_pose = mp.solutions.pose
    # mp_drawing = mp.solutions.drawing_utils  # 註解繪圖相關
    # mp_drawing_styles = mp.solutions.drawing_styles  # 註解繪圖相關
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    
    # 取得影片資訊
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 註解影片輸出相關程式碼
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化物件位置歷史紀錄
    object_history = defaultdict(lambda: [])
    # 移動閾值 - 物體中心點移動超過此距離才視為移動
    movement_threshold = 5
    # 追蹤歷史數量
    history_length = 50
    move_times = 0
    # 目標物體移動狀態記錄
    target_movement_status = {}
    # 記錄package是否被手持拿
    package_holding_status = False
    # 手部關鍵點持有package的距離閾值（像素）
    hand_holding_threshold = 60
    # 手部關鍵點接觸machine的距離閾值（像素）
    hand_touch_threshold = 40
    # 手部關鍵點接觸hand_package的距離閾值（像素）
    hand_package_touch_threshold = 80
    # 手部關鍵點接觸tube的距離閾值（像素）
    tube_touch_threshold = 60
    # package位置記錄
    package_position = None
    # machine位置記錄
    machine1_position = None
    machine2_position = None
    # hand_package位置記錄
    hand_package_position = None
    # front_tube位置記錄
    front_tube_position = None
    # machine接觸狀態
    machine1_touched = False
    machine2_touched = False
    # hand_package接觸狀態
    hand_package_touched = False
    # front_tube狀態
    front_tube_detected = False
    
    print(f"將特別檢測以下物件的移動: {', '.join(target_objects) if target_objects else '所有物件'}")
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1

        # if frame_count % 100 == 0:
            # print(f"Processing frame {frame_count}...")
        
        # 執行物件偵測
        results = model(frame, conf=conf_thres, iou=0.1, max_det=7, device='cuda', verbose=False)
        
        # 註解繪製偵測結果的程式碼
        # annotated_frame = results[0].plot()
        
        # 執行 MediaPipe 骨架偵測
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        
        # 儲存手部關鍵點的位置
        left_wrist = None
        right_wrist = None
        
        # 如果偵測到骨架，則獲取手腕位置（不繪製骨架）
        if pose_results.pose_landmarks:
            # 註解骨架繪製相關程式碼
            # mp_drawing.draw_landmarks(
            #     annotated_frame,
            #     pose_results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
            # 取得左右手腕的座標
            landmarks = pose_results.pose_landmarks.landmark
            if landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.5:
                left_wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * width),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * height))
                # 註解手腕標記繪製
                # cv2.circle(annotated_frame, left_wrist, 8, (255, 255, 0), -1)  # 黃色
                
            if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.5:
                right_wrist = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * width),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * height))
                # 註解手腕標記繪製
                # cv2.circle(annotated_frame, right_wrist, 8, (0, 255, 255), -1)  # 青色

        detected_objects = {}  # 存儲當前幀檢測到的物件
        package_holding_status = False  # 重設狀態
        machine1_touched = False  # 重設 machine1 接觸狀態
        machine2_touched = False  # 重設 machine_2 接觸狀態
        hand_package_touched = False  # 重設 hand_package 接觸狀態
        front_tube_detected = False  # 重設 front_tube 偵測狀態

        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()
            x1,y1,x2,y2 = map(int,box)
            cls = int(result.cls[0].item())
            conf = result.conf[0].item()
            
            object_name = model.names[cls]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # 生成物件的唯一標識
            object_id = f"{object_name}_{cls}"

            if object_name == 'tube':
                tube_position = (center_x, center_y)
                
                # 檢查tube是否靠近任一隻手
                if left_wrist or right_wrist:
                    # 計算雙手與tube的距離
                    left_distance = float('inf')
                    right_distance = float('inf')
                    
                    if left_wrist:
                        left_distance = np.sqrt((center_x - left_wrist[0])**2 + (center_y - left_wrist[1])**2)
                    if right_wrist:
                        right_distance = np.sqrt((center_x - right_wrist[0])**2 + (center_y - right_wrist[1])**2)
                    
                    # 判斷是否有手靠近tube
                    min_distance = min(left_distance, right_distance)
                    tube_touched = min_distance < tube_touch_threshold
                    
                    # 在畫面上顯示tube被接觸狀態
                    # touch_text = f"Touched: {'Yes' if tube_touched else 'No'}"
                    # touch_color = (0, 255, 0) if tube_touched else (0, 0, 255)  # 綠色表示接觸，紅色表示未接觸
                    
                    # cv2.putText(annotated_frame, touch_text, 
                    #           (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    #           touch_color, 2)
                    
                    # 如果被手接觸，畫出連接線
                    if tube_touched:
                    #     # 判斷是哪隻手離tube更近，並畫出連接線
                    #     if left_distance < right_distance and left_wrist:
                    #         cv2.line(annotated_frame, left_wrist, (center_x, center_y), (255, 255, 0), 2)
                    #         cv2.putText(annotated_frame, f"L: {left_distance:.1f}px", 
                    #                  (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    #     elif right_wrist:
                    #         cv2.line(annotated_frame, right_wrist, (center_x, center_y), (0, 255, 255), 2)
                    #         cv2.putText(annotated_frame, f"R: {right_distance:.1f}px", 
                    #                  (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # 記錄接觸tube的幀位置
                        if len(test[6])==0 or frame_count-test[6][-1]>150:
                            test[6].append(frame_count)
                            # print(f"Tube touched at frame {frame_count}")
            
            # 檢查是否為front_tube，紀錄偵測
            if object_name == 'front_tube':
                front_tube_position = (center_x, center_y)
                front_tube_detected = True
                
                # 記錄偵測到front_tube的幀位置
                if len(test[5]) == 0 or frame_count - test[5][-1] > 150:
                    test[5].append(frame_count)
                    # print(f"Front tube detected at frame {frame_count}")
                
                # 註解標籤和框的繪製程式碼
                # cv2.putText(annotated_frame, "front_tube detected", 
                #           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                #           (0, 255, 0), 2)
                # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # 檢查是否為hand_package，判斷手是否接觸
            if object_name == 'hand_package':
                hand_package_position = (center_x, center_y)
                
                # 檢查是否有手接觸
                if left_wrist or right_wrist:
                    # 計算雙手與hand_package的距離
                    left_distance = float('inf')
                    right_distance = float('inf')
                    
                    if left_wrist:
                        left_distance = np.sqrt((center_x - left_wrist[0])**2 + (center_y - left_wrist[1])**2)
                    if right_wrist:
                        right_distance = np.sqrt((center_x - right_wrist[0])**2 + (center_y - right_wrist[1])**2)
                    
                    min_distance = min(left_distance, right_distance)
                    hand_package_touched = min_distance < hand_package_touch_threshold
                    
                    # 記錄接觸事件
                    if hand_package_touched and (len(test[4]) == 0 or frame_count - test[4][-1] > 150):
                        test[4].append(frame_count)
                    
                    # 註解接觸狀態顯示和連接線的繪製程式碼
                    # touch_text = f"Touched: {'Yes' if hand_package_touched else 'No'}"
                    # touch_color = (0, 255, 0) if hand_package_touched else (0, 0, 255)
                    # cv2.putText(annotated_frame, touch_text,
                    #           (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #           touch_color, 2)
                    
                    # if hand_package_touched:
                    #     if left_distance < right_distance and left_wrist:
                    #         cv2.line(annotated_frame, left_wrist, (center_x, center_y), (255, 255, 0), 2)
                    #         cv2.putText(annotated_frame, f"L: {left_distance:.1f}px",
                    #                  (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    #     elif right_wrist:
                    #         cv2.line(annotated_frame, right_wrist, (center_x, center_y), (0, 255, 255), 2)
                    #         cv2.putText(annotated_frame, f"R: {right_distance:.1f}px",
                    #                  (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 檢查是否為machine1或machine_2，判斷手是否接觸
            if object_name in ['machine1', 'machine_2']:
                # 儲存位置
                if object_name == 'machine1':
                    machine1_position = (center_x, center_y)
                else:
                    machine2_position = (center_x, center_y)
                
                # 檢查是否有手接觸
                if left_wrist or right_wrist:
                    # 計算雙手與機器的距離
                    left_distance = float('inf')
                    right_distance = float('inf')
                    
                    if left_wrist:
                        left_distance = np.sqrt((center_x - left_wrist[0])**2 + (center_y - left_wrist[1])**2)
                    if right_wrist:
                        right_distance = np.sqrt((center_x - right_wrist[0])**2 + (center_y - right_wrist[1])**2)
                    
                    min_distance = min(left_distance, right_distance)
                    is_touched = min_distance < hand_touch_threshold
                    
                    # 更新接觸狀態
                    if object_name == 'machine1':
                        machine1_touched = is_touched
                        if is_touched and (len(test[2]) == 0 or frame_count - test[2][-1] > 300):
                            test[2].append(frame_count)
                    else:  # machine_2
                        machine2_touched = is_touched
                        if is_touched and (len(test[3]) == 0 or frame_count - test[3][-1] > 300):
                            test[3].append(frame_count)
                    
                    # 註解接觸狀態顯示和連接線的繪製程式碼
                    # touch_text = f"Touched: {'Yes' if is_touched else 'No'}"
                    # touch_color = (0, 255, 0) if is_touched else (0, 0, 255)
                    # cv2.putText(annotated_frame, touch_text,
                    #           (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #           touch_color, 2)
                    
                    # if is_touched:
                    #     if left_distance < right_distance and left_wrist:
                    #         cv2.line(annotated_frame, left_wrist, (center_x, center_y), (255, 255, 0), 2)
                    #         cv2.putText(annotated_frame, f"L: {left_distance:.1f}px",
                    #                  (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    #     elif right_wrist:
                    #         cv2.line(annotated_frame, right_wrist, (center_x, center_y), (0, 255, 255), 2)
                    #         cv2.putText(annotated_frame, f"R: {right_distance:.1f}px",
                    #                  (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 檢查是否為package，用於判斷是否被手持拿
            if object_name == 'package':
                package_position = (center_x, center_y)
                
                # 檢查package是否靠近任一隻手
                if left_wrist or right_wrist:
                    # 計算雙手與package的距離
                    left_distance = float('inf')
                    right_distance = float('inf')
                    
                    if left_wrist:
                        left_distance = np.sqrt((center_x - left_wrist[0])**2 + (center_y - left_wrist[1])**2)
                    if right_wrist:
                        right_distance = np.sqrt((center_x - right_wrist[0])**2 + (center_y - right_wrist[1])**2)
                    
                    # 判斷是否有手靠近package
                    package_holding_status = min(left_distance, right_distance) < hand_holding_threshold
                    
                    # 註解持有狀態顯示和連接線的繪製程式碼
                    # holding_text = f"Held by hand: {'Yes' if package_holding_status else 'No'}"
                    # holding_color = (0, 255, 0) if package_holding_status else (0, 0, 255)
                    # cv2.putText(annotated_frame, holding_text, 
                    #           (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    #           holding_color, 2)
                    
                    if package_holding_status:
                    #     if left_distance < right_distance and left_wrist:
                    #         cv2.line(annotated_frame, left_wrist, (center_x, center_y), (255, 255, 0), 2)
                    #         cv2.putText(annotated_frame, f"L: {left_distance:.1f}px", 
                    #                  (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    #     elif right_wrist:
                    #         cv2.line(annotated_frame, right_wrist, (center_x, center_y), (0, 255, 255), 2)
                    #         cv2.putText(annotated_frame, f"R: {right_distance:.1f}px", 
                    #                  (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # 記錄被手持拿的package幀位置
                        if len(test[1])==0 or frame_count-test[1][-1]>90:
                            test[1].append(frame_count)
            
            # 檢查是否為目標物件
            is_target = len(target_objects) == 0 or object_name in target_objects
            
            # 註解基本資訊顯示的程式碼
            if not is_target:
            #     cv2.putText(annotated_frame, f"{object_name} {conf:.2f}", 
            #                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                continue
                
            # 記錄當前位置
            detected_objects[object_id] = (center_x, center_y)
            
            # 更新物件歷史位置
            if object_id in object_history:
                object_history[object_id].append((center_x, center_y))
                # 限制歷史記錄長度
                if len(object_history[object_id]) > history_length:
                    object_history[object_id].pop(0)
            else:
                object_history[object_id] = [(center_x, center_y)]
            
            # 判斷物件是否移動
            is_moving = False
            movement_distance = 0
            if len(object_history[object_id]) >= 20:
                # 計算最新兩個位置的距離
                prev_x, prev_y = object_history[object_id][0]
                current_x, current_y = object_history[object_id][-1]
                movement_distance = ((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2) ** 0.5
                is_moving = movement_distance > movement_threshold and movement_distance < 50
            
            # 更新目標物體移動狀態
            target_movement_status[object_name] = is_moving
            if is_moving:
                # print(f"{object_name} is moving, distance: {movement_distance:.1f}")
                if len(test[0])==0 or frame_count-test[0][-1]>300:
                    move_times += 1
                    test[0].append(frame_count)
            
            # 註解物件標籤、座標和移動狀態的顯示程式碼
            # label = f"{object_name} {conf:.2f}"
            # coord_text = f"({center_x}, {center_y})"
            # movement_text = f"moving: {'yes' if is_moving else 'no'}, dis: {movement_distance:.1f}"
            # cv2.putText(annotated_frame, coord_text, 
            #            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(annotated_frame, movement_text, 
            #            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
            #            (0, 255, 0) if is_moving else (0, 0, 255), 2)
            # color = (0, 255, 0) if is_moving else (0, 0, 255)
            # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            # cv2.circle(annotated_frame, (center_x, center_y), 5, color, -1)
            
            # 註解軌跡繪製的程式碼
            # if len(object_history[object_id]) >= 2:
            #     for i in range(1, len(object_history[object_id])):
            #         pt1 = object_history[object_id][i-1]
            #         pt2 = object_history[object_id][i]
            #         cv2.line(annotated_frame, pt1, pt2, (255, 0, 255), 2)
        
        # 清理已不存在的物件歷史記錄
        object_ids_to_remove = []
        for obj_id in object_history:
            if obj_id not in detected_objects:
                object_ids_to_remove.append(obj_id)
        
        for obj_id in object_ids_to_remove:
            del object_history[obj_id]
        
        # 註解狀態摘要顯示的程式碼
        # y_pos = 30
        # cv2.putText(annotated_frame, "Status:", 
        #            (width - 300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # 
        # for obj_name, is_moving in target_movement_status.items():
        #     y_pos += 30
        #     status_text = f"{obj_name}: {'moving' if is_moving else 'stop'}"
        #     status_color = (0, 255, 0) if is_moving else (0, 0, 255)
        #     cv2.putText(annotated_frame, status_text, 
        #                (width - 300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        # 
        # if package_position:
        #     y_pos += 30
        #     holding_status = "Package: " + ("held by hand" if package_holding_status else "not held")
        #     holding_color = (0, 255, 0) if package_holding_status else (0, 0, 255)
        #     cv2.putText(annotated_frame, holding_status, 
        #                (width - 300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, holding_color, 2)
        # 
        # if machine1_position:
        #     y_pos += 30
        #     machine1_status = "Machine1: " + ("touched" if machine1_touched else "not touched")
        #     machine1_color = (0, 255, 0) if machine1_touched else (0, 0, 255)
        #     cv2.putText(annotated_frame, machine1_status, 
        #                (width - 300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, machine1_color, 2)
        # 
        # if machine2_position:
        #     y_pos += 30
        #     machine2_status = "Machine_2: " + ("touched" if machine2_touched else "not touched")
        #     machine2_color = (0, 255, 0) if machine2_touched else (0, 0, 255)
        #     cv2.putText(annotated_frame, machine2_status, 
        #                (width - 300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, machine2_color, 2)
        # 
        # if hand_package_position:
        #     y_pos += 30
        #     hand_package_status = "Hand_package: " + ("touched" if hand_package_touched else "not touched")
        #     hand_package_color = (0, 255, 0) if hand_package_touched else (0, 0, 255)
        #     cv2.putText(annotated_frame, hand_package_status, 
        #                (width - 300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_package_color, 2)
        # 
        # if front_tube_position:
        #     y_pos += 30
        #     front_tube_status = "Front_tube: detected"
        #     cv2.putText(annotated_frame, front_tube_status, 
        #                (width - 300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 註解 FPS 顯示的程式碼
        # elapsed_time = time.time() - start_time
        # fps_current = frame_count / elapsed_time
        # cv2.putText(annotated_frame, f'FPS: {fps_current:.1f}', 
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 註解影片寫入的程式碼
        # out.write(annotated_frame)
        
        # 註解預覽顯示的程式碼
        # if visual:
        #     annotated_frame_display = cv2.resize(annotated_frame, (800, 600))
        #     cv2.imshow('Object Detection', annotated_frame_display)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
    
    # 釋放資源
    pose.close()  # 關閉 MediaPipe 姿勢偵測器
    cap.release()
    # 註解輸出影片釋放
    # out.release()
    # 註解視窗釋放
    # if visual:
    #     cv2.destroyAllWindows()
    
    return test  # 返回記錄的事件幀號

if __name__ == '__main__':
    # 設定路徑
    weights_path = "" # yolo模型路徑
    video_path = ""   # 影片路徑
    output_path = "result.mp4"  # 因為不輸出影片，此參數實際上不會被使用
    # 指定要監測移動的目標物件
    target_objects = ['T-tube']
    
    # 執行偵測，設定 visual=False 以禁用預覽窗口
    events = detect_video(
        weights_path=weights_path,
        video_path=video_path,
        output_path=output_path,
        conf_thres=0.25,
        target_objects=target_objects,
        visual=False  # 設置為 False，完全禁用畫面顯示
    )
    
    print("事件記錄摘要:")
    print(f"T-tube 移動幀數: {events[0]}")
    print(f"Package 持拿幀數: {events[1]}")
    print(f"suck machine 接觸幀數: {events[2]}")
    print(f"oxy machine 接觸幀數: {events[3]}")
    print(f"Hand_package 接觸幀數: {events[4]}")
    print(f"Front_tube 偵測幀數: {events[5]}")
    print(f"Tube 接觸幀數: {events[6]}")
    print(events)