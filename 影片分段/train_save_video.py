import cv2
import numpy as np
import time
import os
import glob

target = [4200]

# 設定輸入影片路徑
video_path = ""
# 打開影片檔
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("無法開啟影片檔案")
    exit()

# 獲取影片的幀寬、幀高和幀率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"原始影片幀率: {fps}")

# 建立輸出資料夾
output_dir = "train_suck_machine"
os.makedirs(output_dir, exist_ok=True)

# 清空輸出資料夾中已有的影片檔
# existing_videos = glob.glob(os.path.join(output_dir, "*.mp4"))
# if existing_videos:
#     print(f"正在清空資料夾 {output_dir} 中的 {len(existing_videos)} 個影片檔")
#     for video_file in existing_videos:
#         os.remove(video_file)

# 對每個起始幀進行處理
for start_frame in target:
    # 設置讀取位置到起始幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 為這個片段創建視頻寫入器
    output_path = os.path.join(output_dir, f"frame_{start_frame}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"處理從幀 {start_frame} 開始的影片片段")
    
    # 讀取並寫入120幀
    frames_count = 0
    while frames_count < 120:
        ret, frame = cap.read()
        if not ret:
            print(f"無法讀取完整的120幀,已讀取 {frames_count} 幀")
            break
        
        # 寫入影格到輸出影片
        out.write(frame)
        frames_count += 1
    
    # 釋放這個區段的視頻寫入器
    out.release()
    print(f"已完成從幀 {start_frame} 開始的影片片段，儲存於 {output_path}")

# 釋放資源
cap.release()
print("所有影片片段處理完成")