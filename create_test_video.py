import cv2
import os
import glob

def create_frankenstein_video():
    # 1. Tìm 1 video NonFight và 1 video Fight bất kỳ
    nonfight_vid = glob.glob("data_videos_mp4/NonFight/*.mp4")[0]
    fight_vid = glob.glob("data_videos_mp4/Fight/*.mp4")[0]
    
    print(f"Đang lấy 10 frame đầu từ NonFight: {os.path.basename(nonfight_vid)}")
    print(f"Đang lấy 5 frame cuối từ Fight: {os.path.basename(fight_vid)}")
    
    # 2. Đọc frame
    cap1 = cv2.VideoCapture(nonfight_vid)
    frames_nonfight = []
    for _ in range(10):
        ret, frame = cap1.read()
        if not ret: break
        frames_nonfight.append(frame)
    cap1.release()
    
    cap2 = cv2.VideoCapture(fight_vid)
    all_frames_fight = []
    while True:
        ret, frame = cap2.read()
        if not ret: break
        all_frames_fight.append(frame)
    cap2.release()
    
    frames_fight = all_frames_fight[-5:] # Lấy 5 frame cuối
    
    # 3. Ghép lại và ghi thành video mới
    all_frames = frames_nonfight + frames_fight
    h, w = all_frames[0].shape[:2]
    
    out = cv2.VideoWriter("test_frankenstein.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
    for frame in all_frames:
        out.write(frame)
    out.release()
    
    print("\n✅ Đã tạo xong video 'test_frankenstein.mp4' ở thư mục gốc.")
    print("Hãy sửa VIDEO_TEST trong demo.py thành 'test_frankenstein.mp4' để kiểm tra!")

if __name__ == "__main__":
    create_frankenstein_video()