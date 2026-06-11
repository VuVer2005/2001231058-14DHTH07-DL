import cv2
import os


def extractFramesFromFolders(videoRoot, outputRoot, seqLength=15, imgSize=(64, 64)):
    """
    Hàm quét qua các thư mục video, cắt thành ảnh và lưu vào thư mục đích.
    """
    classes = ['Fight', 'NonFight']

    for cls in classes:
        videoDir = os.path.join(videoRoot, cls)
        outputDir = os.path.join(outputRoot, cls)
        os.makedirs(outputDir, exist_ok=True)

        # Lấy danh sách tất cả các video trong thư mục
        videoFiles = [f for f in os.listdir(videoDir) if f.endswith(('.mp4', '.avi', '.mkv'))]

        print(f"--- Đang xử lý nhóm: {cls} (Tìm thấy {len(videoFiles)} video) ---")

        for vIdx, vFile in enumerate(videoFiles):
            videoPath = os.path.join(videoDir, vFile)
            cap = cv2.VideoCapture(videoPath)

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize ảnh về kích thước nhỏ cho nhẹ máy
                frameResized = cv2.resize(frame, imgSize)
                frames.append(frameResized)

            cap.release()

            # Tính toán để lấy ra đúng 'seqLength' trải đều toàn bộ video
            totalFrames = len(frames)
            if totalFrames < seqLength:
                print(f"Video {vFile} quá ngắn, bỏ qua.")
                continue

            # Lấy các chỉ số frame cách đều nhau
            step = totalFrames // seqLength
            selectedIndices = [i * step for i in range(seqLength)]

            # Tạo một thư mục riêng cho video này bên thư mục ảnh
            videoName = os.path.splitext(vFile)[0]
            videoOutputPath = os.path.join(outputDir, f"video_{vIdx}_{videoName}")
            os.makedirs(videoOutputPath, exist_ok=True)

            # Lưu các frame đã chọn thành file ảnh .jpg
            for count, idx in enumerate(selectedIndices):
                frameToSave = frames[idx]
                # Tên file xuất ra vẫn giữ nguyên cấu trúc cũ cho hệ thống dễ đọc
                imgName = f"frame_{count:03d}.jpg"
                cv2.imwrite(os.path.join(videoOutputPath, imgName), frameToSave)

            print(f" Thành công: Cắt xong {seqLength} frames từ {vFile}")


if __name__ == "__main__":
    VIDEO_ROOT = "data_videos"
    OUTPUT_ROOT = "data_frames"

    # Chạy hàm
    extractFramesFromFolders(VIDEO_ROOT, OUTPUT_ROOT, seqLength=15, imgSize=(64, 64))
    print("\n Hoàn thành tất cả! Kiểm tra thư mục 'data_frames' xem có ảnh chưa nhé.")