"""
通用结果处理模块
处理图片、视频、音频等文件的存储、缩略图生成、数据库更新、WS消息发送
"""
import os
import uuid
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from schemas import InferenceRequestParams, InferenceResponseParams, FileInfo
from .logger import get_logger
from typing import Protocol, runtime_checkable


@runtime_checkable
class ResultEgress(Protocol):
    async def send_result(self, result_message: dict) -> bool: ...
from .config_loader import load_inference_config, InferenceConfig
from .storage_backends import build_storage_backend, StorageBackendError

logger = get_logger(__name__)

# 尝试导入PIL，如果失败则使用占位符
from PIL import Image

# 尝试导入视频处理库
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available, video thumbnail generation will be disabled")


class ResultHandler:
    """通用结果处理器类"""
    
    def __init__(
        self, 
        ws_client: Optional[ResultEgress] = None,
        storage_base_path: str = "resources/outputs",
        inference_config: Optional[InferenceConfig] = None,
    ):
        """
        初始化结果处理器
        
        Args:
            ws_client: WebSocket客户端（可选，用于发送结果）
            storage_base_path: 存储基础路径（相对于项目根目录）
        
        注意：不再直接操作数据库，文件记录由WebSocket Server根据消息创建
        """
        self.ws_client = ws_client
        self.inference_config = inference_config or load_inference_config()
        # 统一使用绝对路径，避免后续 relative_to 时报路径类型不一致
        self.storage_base_path = Path(storage_base_path).resolve()
        self.storage_base_path.mkdir(parents=True, exist_ok=True)

    def _save_file_to_dir(self, file_data: Any, file_name: str, save_dir: Path) -> Path:
        """保存文件到指定目录（用于 local 或 staging）"""
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / file_name

        if isinstance(file_data, Image.Image):
            file_data.save(file_path)
        elif isinstance(file_data, str):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_data)
        elif isinstance(file_data, bytes):
            with open(file_path, "wb") as f:
                f.write(file_data)
        else:
            try:
                img = Image.fromarray(file_data)
                img.save(file_path)
            except Exception:
                if hasattr(file_data, "tobytes"):
                    with open(file_path, "wb") as f:
                        f.write(file_data.tobytes())
                else:
                    raise ValueError(f"Unsupported file_data type: {type(file_data)}")

        return file_path
    
    def generate_image_thumbnail(self, image_path: Path, thumbnail_size: tuple = (512, 512)) -> Optional[Path]:
        """
        生成图片缩略图
        
        Args:
            image_path: 原始图片路径
            thumbnail_size: 缩略图尺寸
        
        Returns:
            缩略图路径，如果失败则返回None
        """
        
        try:
            # 缩略图文件名：原文件名后加上_s
            thumbnail_path = image_path.parent / f"{image_path.stem}_s{image_path.suffix}"
            
            # 打开图片并生成缩略图
            with Image.open(image_path) as img:
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, quality=85)
            
            return thumbnail_path
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for {image_path}: {e}", exc_info=True)
            return None
    
    def generate_video_thumbnail(
        self, 
        video_path: Path, 
        thumbnail_size: tuple = (512, 512),
        frame_number: int = 10,
        reference_url: Optional[str] = None
    ) -> Optional[Path]:
        """
        生成视频缩略图
        
        Args:
            video_path: 视频文件路径
            thumbnail_size: 缩略图尺寸
            frame_number: 提取第几帧（默认第10帧）
            reference_url: 参考图片URL（如果提供，优先使用）
        
        Returns:
            缩略图路径，如果失败则返回None
        """
        # 如果提供了参考URL，尝试下载并使用
        if reference_url:
            try:
                # TODO: 实现从URL下载图片的逻辑
                # 这里先留空，后续实现
                logger.info(f"Reference URL provided: {reference_url}, but download not implemented yet")
            except Exception as e:
                logger.warning(f"Failed to use reference URL {reference_url}: {e}")
        
        if not CV2_AVAILABLE:
            logger.warning("cv2 not available, skipping video thumbnail generation")
            return None
        
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return None
            
            # 设置到指定帧
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frame = min(frame_number, total_frames - 1) if total_frames > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # 读取帧
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Failed to read frame {target_frame} from video: {video_path}")
                return None
            
            # 生成缩略图文件名：原文件名后加上_s
            thumbnail_path = video_path.parent / f"{video_path.stem}_s.jpg"
            
            # 调整大小并保存
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            img.save(thumbnail_path, quality=85)
            
            return thumbnail_path
        except Exception as e:
            logger.error(f"Failed to generate video thumbnail for {video_path}: {e}", exc_info=True)
            return None
    
    def save_file_local(
        self, 
        file_data: Any, 
        file_name: str,
        subdir: Optional[str] = None
    ) -> Optional[Path]:
        """
        保存文件到本地磁盘（通用方法，支持图片、视频、音频等）
        
        Args:
            file_data: 文件数据（PIL Image对象、numpy数组、字节数据等）
            file_name: 文件名
            subdir: 子目录（可选，例如按日期或用户ID）
        
        Returns:
            保存的文件路径，如果失败则返回None
        """
        try:
            # 确定保存目录
            if subdir:
                save_dir = self.storage_base_path / subdir
            else:
                save_dir = self.storage_base_path
            
            file_path = self._save_file_to_dir(file_data, file_name, save_dir)
            
            logger.info(f"File saved to: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Failed to save file: {e}", exc_info=True)
            return None
    
    def get_mime_type(self, file_ext: str, job_type: str) -> str:
        """
        根据文件扩展名和任务类型获取MIME类型
        
        Args:
            file_ext: 文件扩展名
            job_type: 任务类型（image/video/audio）
        
        Returns:
            MIME类型字符串
        """
        mime_map = {
            "image": {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
                "gif": "image/gif",
            },
            "video": {
                "mp4": "video/mp4",
                "avi": "video/x-msvideo",
                "mov": "video/quicktime",
                "webm": "video/webm",
            },
            "audio": {
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "ogg": "audio/ogg",
                "flac": "audio/flac",
            },
            "text": {
                "txt": "text/plain; charset=utf-8",
            },
            "mini": {
                "md": "text/markdown; charset=utf-8",
                "json": "application/json; charset=utf-8",
                "txt": "text/plain; charset=utf-8",
                # OCR text 模式的图文混排输出（document.md + images/ 打包）
                "zip": "application/zip",
            },
        }

        if file_ext.lower() == "txt":
            return "text/plain; charset=utf-8"
        return mime_map.get(job_type, {}).get(file_ext.lower(), "application/octet-stream")
    
    async def process_single_result(
        self,
        file_data: Any,
        request_params: InferenceRequestParams,
        generate_time: float,
        service_id: str,
        file_seed: Optional[int] = None,
        index: int = 0,
        total: Optional[int] = None,
        *,
        file_name_override: Optional[str] = None,
        key_override: Optional[str] = None,
        extra_message_fields: Optional[Dict[str, Any]] = None,
    ) -> InferenceResponseParams:
        """
        处理单张推理结果并立即发送（用于批处理和多图生成模式）
        
        Args:
            file_data: 文件数据（图片、视频、音频等）
            request_params: 推理请求参数
            generate_time: 生成耗时（秒）
            service_id: 服务ID
            file_seed: 当前文件使用的随机种子（可选）
            index: 当前文件在任务中的序号
            total: 任务总共要处理的图片数量（可选，用于前端显示进度）
        
        Returns:
            InferenceResponseParams对象
        """
        upload_start_time = time.time()
        
        # 确定任务类型（用于判断文件类型）
        task_type = request_params.type  # 任务大类：image/video/audio/text
        if not task_type:
            raise ValueError("type is required in request_params")
        
        # 确定存储目录：{storage_base_path}/{YYYY/MM/DD}
        today = time.strftime("%Y/%m/%d", time.gmtime())
        subdir = today
        
        file_ext = request_params.file_type if request_params.file_type else {
            "image": "png",
            "video": "mp4",
            "audio": "mp3",
            "mini": "md",
            "translate": "txt",
        }.get(task_type, "bin")
        
        # 生成文件名（支持覆盖：用于“长视频分段回传但覆盖同一输出文件”）
        file_name = file_name_override or f"{request_params.task_id}_{index}.{file_ext}"

        # 统一 key（相对路径）：前端/服务端依赖 key 做业务处理（支持覆盖）
        key = key_override or f"{subdir}/{file_name}"

        storage = request_params.storage or getattr(self.inference_config, "storage_default", "local")

        # 保存文件：local 直接落 outputs_dir；其他 storage 先落 staging 再上传
        if storage == "local":
            saved_path = self.save_file_local(file_data, file_name, subdir)
            if not saved_path:
                raise ValueError(f"Failed to save file at index {index}")
        else:
            staging_root = Path(tempfile.gettempdir()) / "vitoom_staging_outputs"
            staging_dir = staging_root / subdir
            saved_path = self._save_file_to_dir(file_data, file_name, staging_dir)
        
        # 生成缩略图（根据文件类型）
        thumbnail_path = None
        if task_type == "image":
            thumbnail_path = self.generate_image_thumbnail(saved_path)
        elif task_type == "video":
            thumbnail_path = self.generate_video_thumbnail(
                saved_path,
                reference_url=request_params.url,
                frame_number=10
            )
        elif task_type == "audio":
            thumbnail_path = None
        elif task_type in ("text", "mini", "translate"):
            # text / mini / translate 文本结果无需缩略图
            thumbnail_path = None
        else:
            logger.warning(f"Unknown task_type: {task_type}, skipping thumbnail generation")

        thumb_key: Optional[str] = f"{subdir}/{thumbnail_path.name}" if thumbnail_path else None

        # 获取文件大小（在清理 staging 前先取到）
        file_size = saved_path.stat().st_size
        
        # 创建文件记录ID
        file_id = str(uuid.uuid4())
        
        # 获取MIME类型
        mime_type = self.get_mime_type(file_ext, task_type)

        # 根据 storage 选择后端并上传（local backend 将确保文件落在 outputs_dir/key）
        backend = build_storage_backend(storage=storage, inference_config=self.inference_config)
        metadata = {
            "task_id": request_params.task_id,
            "user_id": request_params.user_id,
            "job_type": request_params.job_type,
            "type": request_params.type,
            "service_id": service_id,
        }
        try:
            await backend.put_file(key=key, local_path=saved_path, content_type=mime_type, metadata=metadata)
            if thumbnail_path and thumb_key:
                thumb_mime = "image/jpeg" if thumb_key.lower().endswith(".jpg") else mime_type
                await backend.put_file(key=thumb_key, local_path=thumbnail_path, content_type=thumb_mime, metadata=metadata)
        except StorageBackendError:
            raise
        except Exception as e:
            raise StorageBackendError(f"Upload failed: storage={storage}, key={key}, err={e}") from e
        finally:
            # 非 local：清理 staging 文件，避免占用磁盘
            if storage != "local":
                try:
                    if thumbnail_path and thumbnail_path.exists():
                        thumbnail_path.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    if saved_path and saved_path.exists():
                        saved_path.unlink(missing_ok=True)
                except Exception:
                    pass
        
        # 访问 URL 由 Backend 按 storage + key 解析；local 不提供 URL；推理侧不上传外链。
        url = None
        thumb_url = None

        # 构建FileInfo对象
        file_info = FileInfo(
            file_id=file_id,
            storage_path=key,
            file_name=file_name,
            file_size=file_size,
            mime_type=mime_type,
            seed=file_seed,
            index=index,
            thumbnail_path=thumb_key,
            width=request_params.width if task_type == "image" else None,
            height=request_params.height if task_type == "image" else None,
            thumb_url=thumb_url,
            url=url
        )
        
        upload_time = time.time() - upload_start_time
        
        # 构建InferenceResponseParams（包含完整的文件信息）
        message_dict = {
            "type": request_params.type,  # 任务大类：image/video/audio/text
            "job_type": request_params.job_type,  # 任务执行分类：MK, RBG...
            "storage": request_params.storage,
            "reference_id": request_params.reference_id,
            "seed": file_seed if file_seed is not None else request_params.seed,
            "user_id": request_params.user_id,
            "task_id": request_params.task_id,
            "index": index,
            # 多图模式：单张结果也会发送 result 消息；只有最后一张才是 completed。
            # 这样前端不会因为第一张 result.status=completed 就提前关闭 ws。
            "status": "completed",
            "progress": 100,
            "load_name": request_params.load_name,
            "duration": request_params.duration if request_params.type in ["video", "audio"] else 0,
        }

        # 如果提供 total，按 index/total 计算该条结果的状态与进度
        if isinstance(total, int) and total > 0:
            message_dict["total"] = total
            is_last = index >= (total - 1)
            message_dict["status"] = "completed" if is_last else "processing"
            # 进度：按已完成张数估算；非最后一张强制 < 100
            pct = int(round(((index + 1) / total) * 100))
            message_dict["progress"] = 100 if is_last else max(0, min(99, pct))
        
        response_params = InferenceResponseParams(message_dict, service_id, files=[file_info])
        response_params.generate_time = generate_time
        response_params.upload_time = upload_time
        # generate_time/upload_time 在初始化后赋值，需同步更新 used_time
        response_params.used_time = max(0.0, response_params.generate_time) + max(0.0, response_params.upload_time)
        # 如果提供了total参数，使用它；否则使用files列表长度（单图模式）
        if total is not None:
            response_params.total = total
    
        # 通过WebSocket发送结果（如果ws_client可用）
        if self.ws_client:
            result_dict = response_params.to_dict()
            result_dict["type"] = "result"  # 消息类型
            # 允许调用方在结果消息里追加内联字段（例如 mini/OCR 的 content 文本）。
            # 该字段不会影响现有 image/video/audio/text 任务（默认 None）。
            if extra_message_fields:
                for k, v in extra_message_fields.items():
                    result_dict[k] = v
            logger.info(f"任务 {request_params.task_id} 已处理完，回传消息")
            await self.ws_client.send_result(result_dict)
        
        return response_params

