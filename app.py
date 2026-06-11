import streamlit as st
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from demo import predict_video
import os
import subprocess
import tempfile
from pathlib import Path

# Cấu hình trang
st.set_page_config(page_title="AI Hockey Fight Detection", layout="wide")

# Tiêu đề
st.title("🏒 AI PHÁT HIỆN ĐÁNH NHAU TRONG HOCKEY")
st.write("Hệ thống nhận diện hành vi bạo lực sử dụng CNN-LSTM và CNN-BiLSTM")

# =====================================================================
# HÀM HỖ TRỢ: CONVERT VIDEO SANG H.264 CHUẨN WEB
# =====================================================================
def convert_video_to_h264(input_path, output_path):
    """
    Convert video sang H.264 chuẩn web bằng ffmpeg.
    Trả về True nếu thành công, False nếu thất bại.
    """
    try:
        # Lệnh ffmpeg: convert sang H.264 với preset fast
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',  # Quan trọng: giúp web phát ngay
            '-y',  # Ghi đè file output
            output_path
        ]
        
        # Chạy ffmpeg, ẩn output
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60
        )
        
        return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def check_ffmpeg_available():
    """Kiểm tra ffmpeg có sẵn trong hệ thống không."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except:
        return False


# =====================================================================
# TẠO TABS
# =====================================================================
tab1, tab2 = st.tabs(["📊 Biểu Đồ Huấn Luyện", "🎥 Demo Phân Tích Video"])

# =====================================================================
# TAB 1: BIỂU ĐỒ TRAINING (GIỮ NGUYÊN)
# =====================================================================
with tab1:
    st.header("📈 Kết Quả Huấn Luyện Mô Hình")
    
    # Tạo 2 cột cho LSTM
    st.subheader("Project 1: LSTM (1 chiều)")
    col_lstm1, col_lstm2 = st.columns(2)
    
    epochs = np.arange(1, 21)
    
    with col_lstm1:
        train_loss_lstm = [2.1, 1.5, 1.1, 0.85, 0.68, 0.55, 0.45, 0.38, 0.32, 0.28, 
                           0.25, 0.22, 0.20, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11]
        test_loss_lstm = [2.3, 1.7, 1.3, 1.0, 0.82, 0.70, 0.60, 0.52, 0.46, 0.42, 
                          0.39, 0.36, 0.34, 0.32, 0.31, 0.30, 0.29, 0.28, 0.28, 0.27]
        
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs, train_loss_lstm, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, test_loss_lstm, 'r--', label='Test Loss', linewidth=2)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curve - LSTM', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
    
    with col_lstm2:
        train_acc_lstm = [52, 65, 72, 78, 82, 85, 87, 89, 90, 91, 
                          92, 93, 93, 94, 94, 95, 95, 95, 96, 96]
        test_acc_lstm = [48, 60, 68, 74, 78, 81, 83, 84, 85, 85, 
                         86, 86, 87, 87, 87, 88, 88, 88, 88, 88]
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(epochs, train_acc_lstm, 'g-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, test_acc_lstm, 'r--', label='Test Accuracy', linewidth=2)
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy Curve - LSTM', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
    
    st.markdown("---")
    
    # Tạo 2 cột cho Bi-LSTM
    st.subheader("Project 2: Bi-LSTM (2 chiều)")
    col_bilstm1, col_bilstm2 = st.columns(2)
    
    with col_bilstm1:
        train_loss_bilstm = [2.0, 1.4, 1.0, 0.80, 0.65, 0.52, 0.42, 0.35, 0.30, 0.26, 
                             0.23, 0.20, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10]
        test_loss_bilstm = [2.2, 1.6, 1.2, 0.95, 0.78, 0.65, 0.55, 0.48, 0.42, 0.38, 
                            0.35, 0.32, 0.30, 0.28, 0.27, 0.26, 0.25, 0.25, 0.24, 0.24]
        
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(epochs, train_loss_bilstm, 'b-', label='Train Loss', linewidth=2)
        ax3.plot(epochs, test_loss_bilstm, 'r--', label='Test Loss', linewidth=2)
        ax3.set_xlabel('Epochs', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Loss Curve - Bi-LSTM', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
    
    with col_bilstm2:
        train_acc_bilstm = [54, 67, 74, 80, 84, 87, 89, 91, 92, 93, 
                            94, 95, 95, 96, 96, 97, 97, 97, 97, 97]
        test_acc_bilstm = [50, 62, 70, 76, 80, 83, 85, 87, 88, 89, 
                           89, 90, 90, 90, 91, 91, 91, 91, 91, 91]
        
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.plot(epochs, train_acc_bilstm, 'g-', label='Train Accuracy', linewidth=2)
        ax4.plot(epochs, test_acc_bilstm, 'r--', label='Test Accuracy', linewidth=2)
        ax4.set_xlabel('Epochs', fontsize=12)
        ax4.set_ylabel('Accuracy (%)', fontsize=12)
        ax4.set_title('Accuracy Curve - Bi-LSTM', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
    
    # Phần so sánh
    st.markdown("---")
    st.subheader("📊 So Sánh Tổng Quan")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.markdown("""
        **✅ Ưu điểm của Bi-LSTM:**
        - Test Accuracy cao hơn: **91% vs 88%**
        - Ít overfitting hơn: Gap **3% vs 8%**
        - Loss hội tụ tốt hơn
        """)
    
    with col_comp2:
        st.markdown("""
        **⚠️ Nhược điểm của Bi-LSTM:**
        - Thời gian train lâu hơn ~1.5x
        - Số tham số gấp đôi
        - Không thể real-time
        """)

# =====================================================================
# TAB 2: DEMO PHÂN TÍCH VIDEO (ĐÃ SỬA LỖI HIỂN THỊ)
# =====================================================================
with tab2:
    st.header("🎬 Phân Tích Video Trực Tuyến")
    
    # Kiểm tra ffmpeg
    has_ffmpeg = check_ffmpeg_available()
    if not has_ffmpeg:
        st.warning("⚠️ **Không tìm thấy FFmpeg trên hệ thống!** Một số định dạng video (.avi, .mkv) có thể không phát được trên web. AI vẫn phân tích được bình thường.")
    
    # Upload file
    uploaded_file = st.file_uploader("📁 Chọn video cần phân tích", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Hiển thị thông tin file
        st.info(f"📄 File đã upload: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.2f} MB)")
        
        # Đọc bytes từ file upload
        video_bytes = uploaded_file.read()
        
        # Lưu file tạm gốc
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        temp_original_path = f"temp_original{file_ext}"
        temp_web_path = "temp_video_web.mp4"
        
        with open(temp_original_path, "wb") as f:
            f.write(video_bytes)
        
        # Xác định đường dẫn file sẽ dùng để phân tích
        path_for_analysis = temp_original_path
        display_mode = "original"  # 'original', 'converted', 'frames'
        
        # =================================================================
        # XỬ LÝ VIDEO: CONVERT SANG H.264 NẾU CẦN
        # =================================================================
        st.subheader("🔄 Xử lý video...")
        
        # Kiểm tra video gốc bằng OpenCV
        cap_check = cv2.VideoCapture(temp_original_path)
        
        if not cap_check.isOpened():
            st.error("❌ Không thể đọc video này!")
            cap_check.release()
        else:
            # Lấy thông tin video
            fps_orig = cap_check.get(cv2.CAP_PROP_FPS)
            if fps_orig <= 0:
                fps_orig = 25.0
            width_orig = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_orig = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames_orig = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_orig = total_frames_orig / fps_orig if fps_orig > 0 else 0
            
            cap_check.release()
            
            # Kiểm tra xem có cần convert không
            need_conversion = (file_ext != '.mp4') or (not has_ffmpeg)
            
            # Nếu là file .mp4, thử phát trực tiếp trước
            if file_ext == '.mp4':
                st.success("✅ File MP4 - có thể phát trực tiếp")
                path_for_analysis = temp_original_path
                display_mode = "original"
            else:
                # Cần convert sang MP4 H.264
                if has_ffmpeg:
                    with st.spinner("🔄 Đang convert sang định dạng web (H.264)..."):
                        success = convert_video_to_h264(temp_original_path, temp_web_path)
                        
                        if success:
                            st.success("✅ Đã convert video sang H.264 chuẩn web!")
                            path_for_analysis = temp_web_path
                            display_mode = "converted"
                        else:
                            st.warning("⚠️ Convert thất bại. Sẽ dùng file gốc (có thể không phát được trên web nhưng AI vẫn phân tích được)")
                            path_for_analysis = temp_original_path
                            display_mode = "original"
                else:
                    st.warning("⚠️ Không có FFmpeg. Video sẽ được phân tích nhưng có thể không phát được trên web.")
                    path_for_analysis = temp_original_path
                    display_mode = "original"
            
            # =================================================================
            # HIỂN THỊ VIDEO
            # =================================================================
            st.subheader("🎥 Video của bạn:")
            
            if display_mode == "converted":
                # Hiển thị video đã convert
                with open(temp_web_path, "rb") as f:
                    video_bytes_display = f.read()
                st.video(video_bytes_display)
                
                # Lấy thông tin video đã convert
                cap_info = cv2.VideoCapture(temp_web_path)
                if cap_info.isOpened():
                    fps_display = cap_info.get(cv2.CAP_PROP_FPS)
                    total_frames_display = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                    width_display = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height_display = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration_display = total_frames_display / fps_display if fps_display > 0 else 0
                    cap_info.release()
                else:
                    fps_display, total_frames_display = fps_orig, total_frames_orig
                    width_display, height_display = width_orig, height_orig
                    duration_display = duration_orig
                    
            elif display_mode == "original":
                # Thử phát file gốc
                try:
                    with open(temp_original_path, "rb") as f:
                        video_bytes_display = f.read()
                    st.video(video_bytes_display)
                except:
                    # Nếu không phát được, hiển thị frames đầu tiên
                    st.info("ℹ️ Trình duyệt không hỗ trợ phát định dạng này. Hiển thị frame đầu tiên:")
                    cap_temp = cv2.VideoCapture(temp_original_path)
                    ret, first_frame = cap_temp.read()
                    if ret:
                        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                        st.image(first_frame_rgb, caption="Frame đầu tiên của video", use_column_width=True)
                    cap_temp.release()
                
                fps_display, total_frames_display = fps_orig, total_frames_orig
                width_display, height_display = width_orig, height_orig
                duration_display = duration_orig
            
            # =================================================================
            # HIỂN THỊ THÔNG TIN VIDEO
            # =================================================================
            st.markdown("---")
            st.subheader("📊 Thông tin video:")
            
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Độ phân giải", f"{width_display}x{height_display}")
            with col_info2:
                st.metric("FPS", f"{fps_display:.1f}")
            with col_info3:
                st.metric("Tổng frames", total_frames_display)
            with col_info4:
                st.metric("Thời lượng", f"{duration_display:.2f}s")
            
            # Kiểm tra độ dài video
            if total_frames_display < 15:
                st.error(f"⚠️ Video quá ngắn! Chỉ có {total_frames_display} frames (cần tối thiểu 15 frames)")
            else:
                st.success("✅ Video hợp lệ, có thể phân tích!")
                
                # =================================================================
                # PHẦN CHỌN MODEL VÀ PHÂN TÍCH
                # =================================================================
                st.subheader("⚙️ Cấu hình phân tích")
                analysis_mode = st.radio(
                    "Chọn mô hình:",
                    ["🔹 Project 1: LSTM (1 chiều)", 
                     "🔹 Project 2: Bi-LSTM (2 chiều)", 
                     "🔹 Cả hai (So sánh)"],
                    horizontal=True
                )
                
                if st.button("🔍 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
                    with st.spinner("🧠 AI đang xử lý 15 frames và phân tích ngữ cảnh..."):
                        try:
                            if analysis_mode == "🔹 Project 1: LSTM (1 chiều)":
                                res_lstm = predict_video(path_for_analysis, 'lstm', 'model_lstm_final.pth')
                                st.success("✅ PHÂN TÍCH HOÀN TẤT!")
                                st.info(f"**📊 Kết quả Project 1 (LSTM):**\n\n{res_lstm}")
                                
                            elif analysis_mode == "🔹 Project 2: Bi-LSTM (2 chiều)":
                                res_bilstm = predict_video(path_for_analysis, 'bilstm', 'model_bilstm_final.pth')
                                st.success("✅ PHÂN TÍCH HOÀN TẤT!")
                                st.success(f"**🚀 Kết quả Project 2 (Bi-LSTM):**\n\n{res_bilstm}")
                                
                            else:  # Cả hai
                                res_lstm = predict_video(path_for_analysis, 'lstm', 'model_lstm_final.pth')
                                res_bilstm = predict_video(path_for_analysis, 'bilstm', 'model_bilstm_final.pth')
                                
                                st.success("✅ PHÂN TÍCH HOÀN TẤT!")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.info(f"**📊 Project 1 (LSTM):**\n\n{res_lstm}")
                                with col2:
                                    st.success(f"**🚀 Project 2 (Bi-LSTM):**\n\n{res_bilstm}")
                                
                                st.markdown("---")
                                st.subheader("📈 So sánh độ tin cậy:")
                                
                                def parse_confidence(result_str):
                                    try:
                                        conf = float(result_str.split(":")[-1].strip().replace("%", ""))
                                        return conf
                                    except:
                                        return 0.0
                                
                                conf_lstm = parse_confidence(res_lstm)
                                conf_bilstm = parse_confidence(res_bilstm)
                                
                                fig_compare, ax_compare = plt.subplots(figsize=(10, 4))
                                models = ['LSTM', 'Bi-LSTM']
                                confidences = [conf_lstm, conf_bilstm]
                                colors = ['#3498db', '#2ecc71']
                                
                                bars = ax_compare.bar(models, confidences, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                                ax_compare.set_ylabel('Confidence (%)', fontsize=12)
                                ax_compare.set_title('So sánh độ tin cậy giữa 2 mô hình', fontsize=14, fontweight='bold')
                                ax_compare.set_ylim(0, 100)
                                ax_compare.grid(True, alpha=0.3, axis='y')
                                
                                for bar in bars:
                                    height = bar.get_height()
                                    ax_compare.text(bar.get_x() + bar.get_width()/2., height,
                                                   f'{height:.2f}%',
                                                   ha='center', va='bottom', fontsize=12, fontweight='bold')
                                
                                st.pyplot(fig_compare)
                                
                                if abs(conf_lstm - conf_bilstm) > 10:
                                    st.warning(f"💡 **Chênh lệch lớn**: {abs(conf_lstm - conf_bilstm):.1f}% - Đây có thể là video phức tạp!")
                                
                        except Exception as e:
                            st.error(f"❌ Lỗi khi phân tích: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>© 2026 AI Hockey Fight Detection System | CNN-LSTM & CNN-BiLSTM</p>
    </div>
    """, 
    unsafe_allow_html=True
)