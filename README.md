Giải thích các thành phần:
Face_Enhancer_Tool/: Thư mục gốc chứa toàn bộ công cụ.

inference_face_enhancer.py: Tệp thực thi chính để chạy quá trình cải thiện khuôn mặt.

install_setup.py: Tệp cài đặt môi trường Python.

enhancers/: Chứa mã nguồn và các mô hình .onnx của các công cụ cải thiện chất lượng (GFPGAN, Codeformer, v.v.).

faceID/ (Mới): Chứa mô hình faceID.py để bảo toàn đặc điểm nhận dạng của khuôn mặt sau khi cải thiện. Đây là một tính năng tùy chọn nhưng được khuyến khích để có kết quả tốt nhất.

inputs/: Thư mục chứa các video đầu vào.

results/: Thư mục lưu các video kết quả.

temp/: Thư mục chứa các tệp tạm thời.

README.md: Tệp hướng dẫn sử dụng.

Nội dung cho tệp README.md mới:
Bạn có thể tạo một tệp README.md mới trong thư mục Face_Enhancer_Tool với nội dung sau:

# Công cụ Cải thiện Chất lượng Khuôn mặt (Face Enhancer Tool)

Đây là một công cụ độc lập sử dụng các mô hình AI (GFPGAN, Codeformer, GPEN, v.v.) chạy trên ONNX Runtime để cải thiện chất lượng khuôn mặt trong video.

## Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
- Python 3.8+
- (Tùy chọn nhưng khuyến khích) Card đồ họa NVIDIA với CUDA Toolkit và cuDNN đã được cài đặt để tăng tốc GPU.

### 2. Cài đặt thư viện
Chạy tệp kịch bản cài đặt để tự động cài đặt tất cả các thư viện Python cần thiết:
```bash
python install_setup.py

3. Tải các mô hình ONNX
Tải các tệp mô hình .onnx cần thiết và đặt chúng vào các thư mục con tương ứng:

Đặt các mô hình enhancer (ví dụ: GFPGANv1.4.onnx) vào thư mục con trong enhancers/.

Đặt mô hình nhận dạng (ví dụ: arcface_w600k_r50.onnx) vào faceID/.

Cách sử dụng
Đặt các video bạn muốn cải thiện vào thư mục inputs/.

Chạy lệnh sau từ terminal:

python inference_face_enhancer.py --face "inputs/ten_video_cua_ban.mp4" --enhancer GFPGAN

Các tùy chọn chính:
--face: Đường dẫn đến video đầu vào.

--enhancer: Chọn mô hình để sử dụng. Các lựa chọn: GFPGAN, Codeformer, GPEN, RealESRGAN, Restoreformer.

--use_faceid: (Tùy chọn) Thêm cờ này nếu bạn muốn sử dụng mô hình FaceID để bảo toàn nhận dạng khuôn mặt.

--outfile: (Tùy chọn) Đường dẫn để lưu