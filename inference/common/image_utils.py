"""
图片处理工具函数库
提供图片加载、缩放、尺寸获取等公共功能
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Any
import requests
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from common.logger import get_logger

logger = get_logger(__name__)


MAX_IMAGE_PIXELS: int = 1280 * 1280
"""全局像素上限（宽×高）。超过此值的图片/尺寸将被等比缩小，宽高对齐到 16 的倍数。"""


def constrain_size(
    width: int,
    height: int,
    max_pixels: int = MAX_IMAGE_PIXELS,
    multiple: int = 16,
) -> Tuple[int, int]:
    """
    纯计算：若 width*height 超过 max_pixels，等比缩小使总像素不超过 max_pixels，
    并将宽高向下对齐到 multiple 的倍数。不超过上限时原样返回。
    """
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        return multiple, multiple
    if w * h <= max_pixels:
        return w, h
    scale = (max_pixels / (w * h)) ** 0.5
    w = max(multiple, (int(w * scale) // multiple) * multiple)
    h = max(multiple, (int(h * scale) // multiple) * multiple)
    return w, h


def resize_image_if_needed(img: Any, max_size: int = 1280) -> Any:
    """
    等比例缩小图片，使宽*高不超过 max_size*max_size，宽高对齐到 16 的倍数。

    Args:
        img: PIL Image对象
        max_size: 最大尺寸（默认1280）

    Returns:
        处理后的PIL Image对象
    """
    if not PIL_AVAILABLE or img is None:
        return img

    width, height = img.size
    new_width, new_height = constrain_size(width, height, max_pixels=max_size * max_size)
    if new_width == width and new_height == height:
        return img

    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    return img_resized


def load_image(image_path: str, resize: bool = True, max_size: int = 1280) -> Optional[Any]:
    """
    从URL或文件路径加载图片，并可选择性地等比例缩小
    
    Args:
        image_path: 图片URL或文件路径
        resize: 是否缩小图片（默认True）
        max_size: 最大尺寸（默认1280），仅在resize=True时生效
        
    Returns:
        PIL Image对象（RGB模式）或None
        
    Example:
        >>> # 从URL加载并缩小
        >>> img = load_image("https://example.com/image.jpg")
        
        >>> # 从文件路径加载，不缩小
        >>> img = load_image("/path/to/image.jpg", resize=False)
        
        >>> # 从相对路径加载（会自动查找resources目录）
        >>> img = load_image("images/test.jpg")
    """
    if not image_path:
        return None
    
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, cannot load image")
        return None
    
    try:
        if image_path.startswith(('http://', 'https://')):
            # 从URL加载
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            # 从文件路径加载
            if not os.path.isabs(image_path):
                # 相对路径，尝试从resources目录查找
                image_path = os.path.join("resources", image_path.lstrip("/"))
            img = Image.open(image_path)
        
        # 转换为RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 等比例缩小
        if resize:
            img = resize_image_if_needed(img, max_size=max_size)
        
        return img
    except Exception as e:
        logger.warning(f"Failed to load image from {image_path}: {e}")
        return None


def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    """
    获取图片尺寸（宽, 高）
    
    Args:
        image_path: 图片URL或文件路径
        
    Returns:
        (width, height) 元组或None
        
    Example:
        >>> size = get_image_size("https://example.com/image.jpg")
        >>> if size:
        ...     width, height = size
        ...     print(f"Image size: {width}x{height}")
    """
    img = load_image(image_path, resize=False)  # 获取原始尺寸，不缩小
    if img:
        return img.size
    return None


def resize_image_to_fit_with_multiple(
    img: Any, 
    target_width: int, 
    target_height: int, 
    multiple: int = 8
) -> Any:
    """
    将输入的图片按等比例缩放到指定宽*高像素大小（不能超过w*h值，只能稍小），
    同时保持宽和高都是指定值（如8，16，32，64）的倍数
    
    Args:
        img: PIL Image对象
        target_width: 目标宽度
        target_height: 目标高度
        multiple: 倍数（默认8），宽高必须是此值的倍数
        
    Returns:
        处理后的PIL Image对象
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("image.jpg")
        >>> # 缩放到不超过1024x1024，且宽高都是8的倍数
        >>> resized = resize_image_to_fit_with_multiple(img, 1024, 1024, multiple=8)
        >>> # 缩放到不超过512x768，且宽高都是16的倍数
        >>> resized = resize_image_to_fit_with_multiple(img, 512, 768, multiple=16)
    """
    if not PIL_AVAILABLE or img is None:
        return img
    
    original_width, original_height = img.size

    # 复用纯计算逻辑，确保与 calc_fit_size_with_multiple 行为一致
    new_width, new_height = calc_fit_size_with_multiple(
        int(original_width),
        int(original_height),
        int(target_width),
        int(target_height),
        multiple=int(multiple),
    )
    
    # 执行缩放
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    logger.debug(
        f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} "
        f"(target: {target_width}x{target_height}, multiple: {multiple})"
    )
    
    return img_resized


def calc_fit_size_with_multiple(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
    multiple: int = 8,
) -> Tuple[int, int]:
    """
    仅计算“等比例缩放到不超过 target_width/target_height 且宽高为 multiple 倍数”的尺寸，
    不对图片本身做 resize。
    计算逻辑与 `resize_image_to_fit_with_multiple` 保持一致。
    """
    if original_width <= 0 or original_height <= 0:
        return multiple, multiple

    scale_width = target_width / original_width
    scale_height = target_height / original_height
    scale = min(scale_width, scale_height)
    if scale > 1.0:
        scale = 1.0

    scaled_width = original_width * scale
    scaled_height = original_height * scale

    new_width = int(scaled_width // multiple) * multiple
    new_height = int(scaled_height // multiple) * multiple

    if new_width < multiple:
        new_width = multiple
    if new_height < multiple:
        new_height = multiple

    if new_width > target_width:
        new_width = (target_width // multiple) * multiple
    if new_height > target_height:
        new_height = (target_height // multiple) * multiple

    return int(new_width), int(new_height)


def load_images_from_list(image_paths: list, resize: bool = True, max_size: int = 1280) -> list:
    """
    批量加载图片列表
    
    Args:
        image_paths: 图片路径列表（URL或文件路径）
        resize: 是否缩小图片（默认True）
        max_size: 最大尺寸（默认1280）
        
    Returns:
        图片对象列表（PIL Image对象），失败的会跳过
        
    Example:
        >>> paths = ["image1.jpg", "image2.jpg", "https://example.com/image3.jpg"]
        >>> images = load_images_from_list(paths)
    """
    images = []
    for path in image_paths:
        img = load_image(path, resize=resize, max_size=max_size)
        if img:
            images.append(img)
    return images

