import cv2
import argparse
import os
import subprocess
import sys
from tqdm import tqdm
import logging

# Import các enhancers (giả sử đã có implementation)
# from enhancers.GFPGAN.GFPGAN import GFPGANer
# from enhancers.Codeformer.Codeformer import Codeformer
# from enhancers.GPEN.GPEN import GPEN
# from enhancers.RealEsrgan.esrganONNX import RealESRGANer
# from enhancers.restoreformer.restoreformer32 import Restoreformer
# from enhancers.restoreformer.restoreformer16 import Restoreformer16
# from faceID.faceID import FaceID

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockEnhancer:
    """Mock enhancer for testing purposes"""
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        logger.info(f"Initialized enhancer with model: {model_path}")
    
    def enhance(self, frame):
        # Placeholder enhancement - in reality this would use the actual model
        return frame

class MockFaceID:
    """Mock FaceID for testing purposes"""
    def __init__(self, model_path):
        self.model_path = model_path
        logger.info(f"Initialized FaceID with model: {model_path}")
    
    def get_final_image(self, original, enhanced, reference):
        return enhanced

def get_video_details(video_path):
    """Lấy thông tin fps và kích thước của video."""
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise IOError(f"Không thể mở video: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video.release()
        
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video có thông số không hợp lệ")
        
        return fps, width, height, frame_count
    
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin video: {e}")
        raise

def cleanup_temp_files(*file_paths):
    """Dọn dẹp các file tạm thời"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Đã xóa file tạm: {file_path}")
        except Exception as e:
            logger.warning(f"Không thể xóa file {file_path}: {e}")

def main(args):
    """Hàm chính để chạy quá trình cải thiện khuôn mặt."""
    temp_files = []
    
    try:
        logger.info(f"Bắt đầu cải thiện video: {args.face}")
        
        # Kiểm tra tệp video đầu vào
        if not os.path.isfile(args.face):
            raise ValueError(f"Tệp video đầu vào không tồn tại: {args.face}")
        
        # Thiết lập đường dẫn đầu ra
        if args.outfile is None:
            basename = os.path.basename(args.face)
            name, ext = os.path.splitext(basename)
            output_dir = "/app/outputs"
            os.makedirs(output_dir, exist_ok=True)
            args.outfile = os.path.join(output_dir, f"{name}_enhanced_{args.enhancer}{ext}")
        
        # Thiết lập thư mục tạm
        temp_dir = "/app/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_file = os.path.join(temp_dir, f"enhanced_video_no_audio_{os.getpid()}.avi")
        temp_files.append(temp_video_file)
        
        # Lấy thông tin video
        fps, width, height, frame_count = get_video_details(args.face)
        logger.info(f"Video info: {width}x{height}, {fps}fps, {frame_count} frames")
        
        # Khởi tạo enhancer
        enhancer = None
        if args.enhancer == 'GFPGAN':
            model_path = '/app/enhancers/GFPGAN/GFPGANv1.4.onnx'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy model GFPGAN: {model_path}")
            enhancer = MockEnhancer(model_path=model_path, upscale=args.enhancer_w)
        elif args.enhancer == 'Codeformer':
            enhancer = MockEnhancer(model_path='enhancers/Codeformer/codeformer.onnx', upscale=args.enhancer_w)
        elif args.enhancer == 'GPEN':
            enhancer = MockEnhancer(model_path=f'enhancers/GPEN/GPEN-BFR-{args.gpen_type}.onnx')
        elif args.enhancer == 'RealESRGAN':
            enhancer = MockEnhancer(model_path=f'enhancers/RealEsrgan/RealESRGAN_x2plus.onnx')
        elif args.enhancer == 'Restoreformer':
            enhancer = MockEnhancer(model_path='enhancers/restoreformer/restoreformer.onnx')
        elif args.enhancer == 'Restoreformer32':
            enhancer = MockEnhancer(model_path='enhancers/restoreformer/restoreformer32.onnx')
        elif args.enhancer == 'Restoreformer16':
            enhancer = MockEnhancer(model_path='enhancers/restoreformer/restoreformer16.onnx')
        else:
            raise ValueError(f"Enhancer không hợp lệ: {args.enhancer}")
        
        # Khởi tạo FaceID nếu được yêu cầu
        faceid_model = None
        if args.use_faceid:
            faceid_model_path = '/app/faceID/arcface_w600k_r50.onnx'
            if not os.path.exists(faceid_model_path):
                raise FileNotFoundError(f"Không tìm thấy model FaceID: {faceid_model_path}")
            logger.info("Đang sử dụng FaceID để bảo toàn nhận dạng.")
            faceid_model = MockFaceID(model_path=faceid_model_path)
        
        # Mở video để đọc
        video_stream = cv2.VideoCapture(args.face)
        if not video_stream.isOpened():
            raise IOError(f"Không thể mở video stream: {args.face}")
        
        # Mở video để ghi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_video_file, fourcc, fps, (width, height))
        if not out.isOpened():
            raise IOError(f"Không thể tạo video output: {temp_video_file}")
        
        # Xử lý từng khung hình
        processed_frames = 0
        for frame_idx in tqdm(range(frame_count), desc=f"Đang cải thiện bằng {args.enhancer}..."):
            ret, frame = video_stream.read()
            if not ret:
                logger.warning(f"Không thể đọc frame {frame_idx}")
                break
            
            # Cải thiện khung hình
            try:
                original_frame = frame.copy()
                enhanced_frame = enhancer.enhance(frame)
                
                # Áp dụng FaceID nếu được bật
                if faceid_model:
                    enhanced_frame = faceid_model.get_final_image(original_frame, enhanced_frame, original_frame)
                
                processed_frames += 1
                
            except Exception as e:
                logger.error(f"Lỗi khi cải thiện frame {frame_idx}: {e}")
                enhanced_frame = frame  # Sử dụng khung hình gốc nếu có lỗi
            
            out.write(enhanced_frame)
        
        # Giải phóng tài nguyên
        video_stream.release()
        out.release()
        
        logger.info(f"Đã xử lý {processed_frames}/{frame_count} frames")
        
        # Kiểm tra file tạm có tồn tại không
        if not os.path.exists(temp_video_file):
            raise FileNotFoundError(f"Không tìm thấy file video tạm: {temp_video_file}")
        
        # Thêm âm thanh từ video gốc vào video đã cải thiện
        logger.info("Đang thêm âm thanh vào video đã cải thiện...")
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        
        # Sử dụng ffmpeg để merge video và audio
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video_file,
            '-i', args.face,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-shortest',
            args.outfile
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            # Fallback: copy temp file to output
            import shutil
            shutil.copy2(temp_video_file, args.outfile)
            logger.warning("Đã copy video không có âm thanh do lỗi FFmpeg")
        
        # Kiểm tra file output
        if not os.path.exists(args.outfile):
            raise FileNotFoundError(f"Không tạo được file output: {args.outfile}")
        
        logger.info(f"Hoàn thành! Video đã cải thiện được lưu tại: {args.outfile}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý: {e}")
        return False
    
    finally:
        # Dọn dẹp files tạm
        cleanup_temp_files(*temp_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cải thiện chất lượng khuôn mặt trong video bằng các mô hình AI.")
    parser.add_argument("--face", type=str, required=True, help="Đường dẫn đến video cần cải thiện.")
    parser.add_argument("--enhancer", type=str, default="GFPGAN", 
                       choices=['GFPGAN', 'Codeformer', 'GPEN', 'RealESRGAN', 'Restoreformer', 'Restoreformer32', 'Restoreformer16'],
                       help="Chọn mô hình để cải thiện khuôn mặt.")
    parser.add_argument("--enhancer_w", type=float, default=0.5, 
                       help="Trọng số hòa trộn của enhancer (chỉ một số enhancer hỗ trợ), từ 0 đến 1.")
    parser.add_argument("--gpen_type", type=str, default="256", choices=['256', '512'], 
                       help="Chọn loại mô hình GPEN (256 hoặc 512).")
    parser.add_argument("--use_faceid", action='store_true', 
                       help="Thêm cờ này để sử dụng FaceID nhằm bảo toàn nhận dạng khuôn mặt.")
    parser.add_argument("--outfile", type=str, default=None, 
                       help="Đường dẫn để lưu video kết quả.")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.face):
        logger.error(f"File video không tồn tại: {args.face}")
        sys.exit(1)
    
    if args.enhancer_w < 0 or args.enhancer_w > 1:
        logger.error("enhancer_w phải trong khoảng 0-1")
        sys.exit(1)
    
    success = main(args)
    sys.exit(0 if success else 1)
