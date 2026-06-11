import os
import glob
from demo import predict_video

def parse_result(res_str):
    if "LỖI" in res_str or "quá ngắn" in res_str:
        return None, 0.0
    # Parse string: "Fight (Đánh nhau) - Độ tin cậy: 99.02%"
    cls = "Fight" if "Fight" in res_str.split("-")[0] else "NonFight"
    conf = float(res_str.split(":")[-1].strip().replace("%", ""))
    return cls, conf

def scan_dataset():
    # Quét tất cả video mp4 trong thư mục
    video_paths = glob.glob("data_videos_mp4/**/*.mp4", recursive=True)
    print(f"🔍 Đang quét {len(video_paths)} video để tìm sự chênh lệch...\n")
    
    results = []
    
    for i, v_path in enumerate(video_paths):
        # Chạy cả 2 model
        res_lstm = predict_video(v_path, 'lstm', 'model_lstm_final.pth')
        res_bilstm = predict_video(v_path, 'bilstm', 'model_bilstm_final.pth')
        
        cls_lstm, conf_lstm = parse_result(res_lstm)
        cls_bilstm, conf_bilstm = parse_result(res_bilstm)
        
        if cls_lstm and cls_bilstm:
            # Tính độ chênh lệch tuyệt đối
            diff = abs(conf_lstm - conf_bilstm)
            
            # Lưu lại nếu: 2 model đoán khác nhau HOẶC chênh lệch confidence > 10%
            if cls_lstm != cls_bilstm or diff > 10.0:
                results.append({
                    'path': v_path,
                    'lstm': f"{cls_lstm} ({conf_lstm:.2f}%)",
                    'bilstm': f"{cls_bilstm} ({conf_bilstm:.2f}%)",
                    'diff': diff
                })
        
        if (i + 1) % 50 == 0:
            print(f"  Đã quét được {i + 1}/{len(video_paths)} video...")

    # Sắp xếp theo độ chênh lệch giảm dần
    results.sort(key=lambda x: x['diff'], reverse=True)
    
    print("\n" + "="*60)
    print("🏆 TOP CÁC VIDEO CÓ SỰ CHÊNH LỆCH LỚN NHẤT:")
    print("="*60)
    for idx, r in enumerate(results[:10]): # In ra top 10
        print(f"\n{idx + 1}. File: {os.path.basename(r['path'])}")
        print(f"   -> LSTM:   {r['lstm']}")
        print(f"   -> Bi-LSTM: {r['bilstm']}")
        print(f"   -> Chênh lệch: {r['diff']:.2f}%")

if __name__ == "__main__":
    scan_dataset()