import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import kiến trúc của 2 model (nhớ để đúng tên file bạn đã lưu)
from model_lstm import CnnLstmModel
from model_bilstm import CnnBiLstmModel

def predict_video(video_path, model_type='lstm', weights_path='model_lstm_final.pth'):
    # 1. Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Khởi tạo "bộ não" trống (kiến trúc)
    if model_type == 'lstm':
        model = CnnLstmModel(numClasses=2).to(device)
    else:
        model = CnnBiLstmModel(numClasses=2).to(device)
        
    # 3. Nạp "kinh nghiệm" (file .pth) vào bộ não
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Chuyển sang chế độ dự đoán (tắt dropout)

    # 4. Đọc video và cắt lấy 15 frames (Giống hệt lúc cắt frame train)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Resize về 64x64 và chuyển từ BGR (OpenCV) sang RGB (PIL/PyTorch)
        frame_resized = cv2.resize(frame, (64, 64))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    total_frames = len(frames)
    if total_frames < 15:
        return "Video quá ngắn, không thể phân tích!"

    # Lấy đều 15 frames
    step = total_frames // 15
    selected_indices = [i * step for i in range(15)]
    selected_frames = [frames[i] for i in selected_indices]

    # 5. Tiền xử lý ảnh (Transform) - Phải giống hệt lúc train
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_frames = []
    for f in selected_frames:
        img_pil = Image.fromarray(f)
        tensor_frames.append(transform(img_pil))
    
    # Gom thành 1 tensor có shape (1, 15, 3, 64, 64) - 1 là batch size
    input_tensor = torch.stack(tensor_frames).unsqueeze(0).to(device)

    # 6. Dự đoán
    with torch.no_grad():
        outputs = model(input_tensor)
        # Dùng Softmax để chuyển đổi số liệu thô thành phần trăm %
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item() * 100

    # 7. Trả về kết quả
    class_names = ['Fight (Đánh nhau)', 'NonFight (Bình thường)']
    return f"{class_names[predicted_class]} - Độ tin cậy: {confidence:.2f}%"


if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN TẠI ĐÂY ---
    VIDEO_TEST = "data_videos/Fight/fi426_xvid.avi" 
    
    print("--- ĐANG TEST PROJECT 1 (LSTM) ---")
    # Tải file lstm bạn đã train ở trên
    kq1 = predict_video(VIDEO_TEST, model_type='lstm', weights_path='model_lstm_final.pth')
    print("Kết quả:", kq1)

    print("\n--- ĐANG TEST PROJECT 2 (BI-LSTM) ---")
    # Tải file bilstm bạn đã train ở trên
    kq2 = predict_video(VIDEO_TEST, model_type='bilstm', weights_path='model_bilstm_final.pth')
    print("Kết quả:", kq2)