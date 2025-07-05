#!/usr/bin/env python3

import os
import subprocess
import logging
import requests
import tempfile
import json
import traceback
from typing import Dict, Any, Optional
import runpod
import time
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
TIMEOUT_SECONDS = 1800  # 30 minutes
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

class FaceEnhancerError(Exception):
    """Custom exception for face enhancer errors"""
    pass

def validate_video_url(url: str) -> tuple[bool, str]:
    """Validate video URL and return (is_valid, error_message)"""
    try:
        logger.info(f"Validating video URL: {url}")
        
        # Check if URL is accessible
        response = requests.head(url, timeout=30, allow_redirects=True)
        
        if response.status_code != 200:
            return False, f"URL không thể truy cập: HTTP {response.status_code}"
        
        # Check file size
        content_length = response.headers.get('content-length')
        if content_length:
            file_size = int(content_length)
            if file_size > MAX_FILE_SIZE:
                return False, f"File quá lớn: {file_size} bytes (max: {MAX_FILE_SIZE})"
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if content_type and not any(vid_type in content_type for vid_type in ['video/', 'application/octet-stream']):
            logger.warning(f"Content-type có thể không phải video: {content_type}")
        
        return True, "URL hợp lệ"
        
    except requests.exceptions.Timeout:
        return False, "Timeout khi kiểm tra URL"
    except requests.exceptions.RequestException as e:
        return False, f"Lỗi khi kiểm tra URL: {str(e)}"
    except Exception as e:
        return False, f"Lỗi không xác định: {str(e)}"

def download_video(url: str, output_path: str) -> bool:
    """Download video from URL with progress tracking"""
    try:
        logger.info(f"Bắt đầu tải video từ: {url}")
        
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 10MB
                    if downloaded % (10 * 1024 * 1024) == 0:
                        progress = (downloaded / total_size * 100) if total_size > 0 else 0
                        logger.info(f"Đã tải: {downloaded:,} bytes ({progress:.1f}%)")
        
        if os.path.getsize(output_path) == 0:
            raise FaceEnhancerError("File tải về có kích thước 0")
        
        logger.info(f"Tải video thành công: {os.path.getsize(output_path):,} bytes")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi khi tải video: {e}")
        return False
    except Exception as e:
        logger.error(f"Lỗi không xác định khi tải video: {e}")
        return False

def run_face_enhancement(input_path: str, output_path: str, 
                        enhancer: str = "GFPGAN", 
                        use_faceid: bool = True, 
                        enhancer_w: float = 0.5) -> tuple[bool, str]:
    """Run face enhancement with detailed logging"""
    try:
        logger.info(f"Bắt đầu cải thiện khuôn mặt với {enhancer}")
        
        # Validate input file
        if not os.path.exists(input_path):
            raise FaceEnhancerError(f"File input không tồn tại: {input_path}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare command
        cmd = [
            "python", "/app/inference_face_enhancer.py",
            "--face", input_path,
            "--enhancer", enhancer,
            "--enhancer_w", str(enhancer_w),
            "--outfile", output_path
        ]
        
        if use_faceid:
            cmd.append("--use_faceid")
        
        logger.info(f"Chạy lệnh: {' '.join(cmd)}")
        
        # Run with timeout
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app"
        )
        
        try:
            stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
            end_time = time.time()
            
            if process.returncode == 0:
                logger.info(f"Cải thiện khuôn mặt thành công trong {end_time - start_time:.2f}s")
                
                # Verify output file
                if not os.path.exists(output_path):
                    raise FaceEnhancerError(f"File output không được tạo: {output_path}")
                
                output_size = os.path.getsize(output_path)
                if output_size == 0:
                    raise FaceEnhancerError("File output có kích thước 0")
                
                logger.info(f"File output: {output_size:,} bytes")
                return True, "Thành công"
                
            else:
                error_msg = f"Lỗi inference (exit code {process.returncode}): {stderr}"
                logger.error(error_msg)
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            error_msg = f"Timeout sau {TIMEOUT_SECONDS}s"
            logger.error(error_msg)
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Lỗi trong quá trình cải thiện: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg

def cleanup_files(*file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Đã xóa file: {file_path}")
        except Exception as e:
            logger.warning(f"Không thể xóa file {file_path}: {e}")

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler"""
    job_id = job.get('id', 'unknown')
    logger.info(f"Bắt đầu xử lý job {job_id}")
    
    temp_files = []
    
    try:
        # Parse input
        input_data = job.get("input", {})
        
        # Required parameters
        video_url = input_data.get("video_url")
        if not video_url:
            return {"error": "Thiếu tham số video_url"}
        
        # Optional parameters with defaults
        enhancer = input_data.get("enhancer", "GFPGAN")
        use_faceid = input_data.get("use_faceid", True)
        enhancer_w = float(input_data.get("enhancer_w", 0.5))
        
        # Validate parameters
        valid_enhancers = ['GFPGAN', 'Codeformer', 'GPEN', 'RealESRGAN', 'Restoreformer', 'Restoreformer32', 'Restoreformer16']
        if enhancer not in valid_enhancers:
            return {"error": f"Enhancer không hợp lệ: {enhancer}. Chọn từ: {valid_enhancers}"}
        
        if not (0 <= enhancer_w <= 1):
            return {"error": "enhancer_w phải trong khoảng 0-1"}
        
        logger.info(f"Job {job_id}: video_url={video_url}, enhancer={enhancer}, use_faceid={use_faceid}, enhancer_w={enhancer_w}")
        
        # Validate video URL
        url_valid, url_error = validate_video_url(video_url)
        if not url_valid:
            return {"error": f"URL không hợp lệ: {url_error}"}
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
            input_path = temp_input.name
            temp_files.append(input_path)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            output_path = temp_output.name
            temp_files.append(output_path)
        
        logger.info(f"Job {job_id}: Temp files - input: {input_path}, output: {output_path}")
        
        # Step 1: Download video
        logger.info(f"Job {job_id}: Bắt đầu tải video...")
        if not download_video(video_url, input_path):
            return {"error": "Không thể tải video"}
        
        # Step 2: Run face enhancement
        logger.info(f"Job {job_id}: Bắt đầu cải thiện khuôn mặt...")
        success, message = run_face_enhancement(input_path, output_path, enhancer, use_faceid, enhancer_w)
        
        if not success:
            return {"error": f"Cải thiện khuôn mặt thất bại: {message}"}
        
        # Step 3: Prepare output
        try:
            output_size = os.path.getsize(output_path)
            logger.info(f"Job {job_id}: File output size: {output_size:,} bytes")
            
            # For RunPod, we typically return base64 for small files or upload to cloud storage
            if output_size <= 50 * 1024 * 1024:  # 50MB limit for base64
                with open(output_path, 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                
                result = {
                    "status": "success",
                    "message": "Cải thiện khuôn mặt thành công",
                    "output": {
                        "video_base64": video_data,
                        "file_size": output_size,
                        "enhancer_used": enhancer,
                        "faceid_used": use_faceid,
                        "enhancer_weight": enhancer_w
                    }
                }
                
                logger.info(f"Job {job_id}: Hoàn thành thành công")
                return result
                
            else:
                # For large files, you would upload to cloud storage here
                # and return the URL instead of base64
                logger.warning(f"Job {job_id}: File quá lớn cho base64: {output_size:,} bytes")
                return {
                    "error": "File kết quả quá lớn. Cần implement cloud storage upload.",
                    "file_size": output_size
                }
                
        except Exception as e:
            logger.error(f"Job {job_id}: Lỗi khi xử lý output: {e}")
            return {"error": f"Lỗi khi xử lý file kết quả: {str(e)}"}
        
    except Exception as e:
        logger.error(f"Job {job_id}: Lỗi trong handler: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Lỗi server: {str(e)}"}
    
    finally:
        # Always cleanup temp files
        cleanup_files(*temp_files)

if __name__ == "__main__":
    logger.info("🚀 Khởi động Face Enhancer RunPod serverless handler...")
    
    # Verify environment
    try:
        import onnxruntime
        logger.info(f"✅ ONNX Runtime version: {onnxruntime.__version__}")
        providers = onnxruntime.get_available_providers()
        logger.info(f"✅ Available providers: {providers}")
        
        if "CUDAExecutionProvider" in providers:
            logger.info("🎯 CUDA provider available - GPU acceleration enabled")
        else:
            logger.warning("⚠️ CUDA provider not available - running on CPU")
            
    except Exception as e:
        logger.error(f"❌ Error checking environment: {e}")
    
    # Start RunPod serverless
    runpod.serverless.start({"handler": handler})
