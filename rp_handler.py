#!/usr/bin/env python3

import os
import subprocess
import logging
import requests
import tempfile
import shutil
import json
import traceback
from typing import Dict, Any
import runpod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
TIMEOUT_SECONDS = 1800  # 30 minutes

def validate_video_url(url: str) -> bool:
    """Validate video URL"""
    try:
        response = requests.head(url, timeout=30)
        content_type = response.headers.get('content-type', '')
        content_length = int(response.headers.get('content-length', 0))
        
        if content_length > MAX_FILE_SIZE:
            logger.error(f"File quá lớn: {content_length} bytes")
            return False
        
        if not any(fmt in content_type for fmt in ['video/', 'application/octet-stream']):
            logger.warning(f"Content-type có thể không phải video: {content_type}")
        
        return response.status_code == 200
    
    except Exception as e:
        logger.error(f"Lỗi khi validate URL: {e}")
        return False

def download_video(url: str, output_path: str) -> bool:
    """Download video from URL"""
    try:
        logger.info(f"Đang tải video từ: {url}")
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.getsize(output_path) == 0:
            raise ValueError("File tải về có kích thước 0")
        
        logger.info(f"Đã tải video thành công: {os.path.getsize(output_path)} bytes")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi tải video: {e}")
        return False

def upload_to_storage(file_path: str) -> str:
    """Upload file to storage service (implementation needed)"""
    # Placeholder - implement actual upload logic
    # For now, just return local path
    return file_path

def run_inference(video_path: str, output_path: str, enhancer: str = "GFPGAN", 
                 use_faceid: bool = True, enhancer_w: float = 0.5) -> bool:
    """Run face enhancement inference"""
    try:
        logger.info(f"Bắt đầu quá trình cải thiện khuôn mặt với {enhancer}...")
        
        # Validate input file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File input không tồn tại: {video_path}")
        
        # Prepare command
        cmd = [
            "python3", "/app/inference_face_enhancer.py",
            "--face", video_path,
            "--enhancer", enhancer,
            "--enhancer_w", str(enhancer_w),
            "--outfile", output_path
        ]
        
        if use_faceid:
            cmd.append("--use_faceid")
        
        logger.info(f"Chạy lệnh: {' '.join(cmd)}")
        
        # Run inference
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=TIMEOUT_SECONDS,
            cwd="/app"
        )
        
        if result.returncode == 0:
            logger.info("Quá trình cải thiện khuôn mặt hoàn tất thành công.")
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"File output không được tạo: {output_path}")
            
            return True
        else:
            logger.error(f"Lỗi inference: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Quá trình cải thiện khuôn mặt bị timeout")
        return False
    except Exception as e:
        logger.error(f"Lỗi trong quá trình inference: {e}")
        logger.error(traceback.format_exc())
        return False

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler for RunPod serverless"""
    temp_files = []
    
    try:
        # Parse input
        input_data = job.get("input", {})
        video_url = input_data.get("video_url")
        enhancer = input_data.get("enhancer", "GFPGAN")
        use_faceid = input_data.get("use_faceid", True)
        enhancer_w = input_data.get("enhancer_w", 0.5)
        
        # Validate input
        if not video_url:
            return {"error": "Thiếu tham số video_url"}
        
        if enhancer not in ['GFPGAN', 'Codeformer', 'GPEN', 'RealESRGAN', 'Restoreformer', 'Restoreformer32', 'Restoreformer16']:
            return {"error": f"Enhancer không hợp lệ: {enhancer}"}
        
        if not (0 <= enhancer_w <= 1):
            return {"error": "enhancer_w phải trong khoảng 0-1"}
        
        logger.info(f"Xử lý job với video: {video_url}, enhancer: {enhancer}")
        
        # Validate video URL
        if not validate_video_url(video_url):
            return {"error": "URL video không hợp lệ hoặc không thể truy cập"}
        
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
            input_path = temp_input.name
            temp_files.append(input_path)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            output_path = temp_output.name
            temp_files.append(output_path)
        
        # Step 1: Download video
        if not download_video(video_url, input_path):
            return {"error": "Không thể tải video"}
        
        # Step 2: Run face enhancement
        if not run_inference(input_path, output_path, enhancer, use_faceid, enhancer_w):
            return {"error": "Quá trình cải thiện khuôn mặt thất bại"}
        
        # Step 3: Upload result (implement as needed)
        # result_url = upload_to_storage(output_path)
        
        # For now, read file content and return as base64 (for small files)
        # In production, upload to cloud storage and return URL
        try:
            file_size = os.path.getsize(output_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit for base64
                logger.warning(f"File quá lớn để trả về trực tiếp: {file_size} bytes")
                return {"error": "File kết quả quá lớn, cần implement cloud storage"}
            
            import base64
            with open(output_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            
            return {
                "status": "success",
                "message": "Cải thiện video thành công",
                "output": {
                    "video_base64": video_data,
                    "file_size": file_size,
                    "enhancer_used": enhancer,
                    "faceid_used": use_faceid
                }
            }
        
        except Exception as e:
            logger.error(f"Lỗi khi xử lý output: {e}")
            return {"error": "Lỗi khi xử lý file kết quả"}
        
    except Exception as e:
        logger.error(f"Lỗi trong handler: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Lỗi server: {str(e)}"}
    
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Đã xóa file tạm: {temp_file}")
            except Exception as e:
                logger.warning(f"Không thể xóa file tạm {temp_file}: {e}")

if __name__ == "__main__":
    logger.info("Khởi động RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
