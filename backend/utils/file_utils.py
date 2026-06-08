"""
文件操作工具
"""
import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Union
import mimetypes


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        dir_path: 目录路径
    
    Returns:
        Path对象
    
    Example:
        >>> ensure_dir("test_dir")
        Path('test_dir')
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def ensure_file_dir(file_path: Union[str, Path]) -> Path:
    """
    确保文件所在目录存在，如果不存在则创建
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件所在目录的Path对象
    
    Example:
        >>> ensure_file_dir("test/subdir/file.txt")
        Path('test/subdir')
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    return file_path.parent


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件大小（字节）
    
    Example:
        >>> size = get_file_size("test.txt")
        >>> isinstance(size, int)
        True
    """
    return Path(file_path).stat().st_size


def get_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    计算文件的哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法（md5, sha1, sha256等），默认为sha256
    
    Returns:
        文件的哈希值（十六进制字符串）
    
    Example:
        >>> hash_value = get_file_hash("test.txt")
        >>> len(hash_value) == 64  # SHA256 produces 64 hex chars
        True
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def read_file(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径
        encoding: 文件编码，默认为utf-8
    
    Returns:
        文件内容字符串
    
    Example:
        >>> content = read_file("test.txt")
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_file(
    file_path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    ensure_dir_exists: bool = True
) -> Path:
    """
    写入文件内容
    
    Args:
        file_path: 文件路径
        content: 文件内容
        encoding: 文件编码，默认为utf-8
        ensure_dir_exists: 如果为True，确保目录存在
    
    Returns:
        Path对象
    
    Example:
        >>> write_file("test.txt", "Hello World")
        Path('test.txt')
    """
    file_path = Path(file_path)
    if ensure_dir_exists:
        ensure_file_dir(file_path)
    
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)
    
    return file_path


def copy_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    ensure_dir_exists: bool = True
) -> Path:
    """
    复制文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        ensure_dir_exists: 如果为True，确保目标目录存在
    
    Returns:
        目标文件Path对象
    
    Example:
        >>> copy_file("src.txt", "dst.txt")
        Path('dst.txt')
    """
    src = Path(src)
    dst = Path(dst)
    
    if ensure_dir_exists:
        ensure_file_dir(dst)
    
    shutil.copy2(src, dst)
    return dst


def move_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    ensure_dir_exists: bool = True
) -> Path:
    """
    移动文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        ensure_dir_exists: 如果为True，确保目标目录存在
    
    Returns:
        目标文件Path对象
    
    Example:
        >>> move_file("src.txt", "dst.txt")
        Path('dst.txt')
    """
    src = Path(src)
    dst = Path(dst)
    
    if ensure_dir_exists:
        ensure_file_dir(dst)
    
    shutil.move(str(src), str(dst))
    return dst


def delete_file(file_path: Union[str, Path]) -> bool:
    """
    删除文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        如果删除成功返回True，否则返回False
    
    Example:
        >>> delete_file("test.txt")
        True
    """
    try:
        Path(file_path).unlink()
        return True
    except Exception:
        return False


def list_files(
    dir_path: Union[str, Path],
    pattern: Optional[str] = None,
    recursive: bool = False
) -> List[Path]:
    """
    列出目录中的文件
    
    Args:
        dir_path: 目录路径
        pattern: 文件名模式（如 "*.txt"），如果为None则列出所有文件
        recursive: 是否递归搜索子目录
    
    Returns:
        文件路径列表
    
    Example:
        >>> files = list_files("test_dir", "*.txt")
        >>> isinstance(files, list)
        True
    """
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        return []
    
    if pattern:
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
    else:
        if recursive:
            files = [f for f in dir_path.rglob("*") if f.is_file()]
        else:
            files = [f for f in dir_path.iterdir() if f.is_file()]
    
    return sorted(files)


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    生成安全的文件名（移除非法字符）
    
    Args:
        filename: 原始文件名
        max_length: 最大长度，默认为255
    
    Returns:
        安全的文件名
    
    Example:
        >>> safe_filename("test/file.txt")
        'test_file.txt'
    """
    # 移除路径分隔符和非法字符
    safe_name = filename.replace("/", "_").replace("\\", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._-")
    
    # 限制长度
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:max_length - len(ext)] + ext
    
    return safe_name


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    获取文件扩展名（不含点号）
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件扩展名（不含点号）
    
    Example:
        >>> get_file_extension("test.txt")
        'txt'
    """
    return Path(file_path).suffix.lstrip(".")


def is_image_file(file_path: Union[str, Path]) -> bool:
    """
    判断是否为图片文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        如果是图片文件返回True，否则返回False
    
    Example:
        >>> is_image_file("test.jpg")
        True
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}
    return Path(file_path).suffix.lower() in image_extensions


def is_video_file(file_path: Union[str, Path]) -> bool:
    """
    判断是否为视频文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        如果是视频文件返回True，否则返回False
    
    Example:
        >>> is_video_file("test.mp4")
        True
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    return Path(file_path).suffix.lower() in video_extensions


def is_audio_file(file_path: Union[str, Path]) -> bool:
    """
    判断是否为音频文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        如果是音频文件返回True，否则返回False
    
    Example:
        >>> is_audio_file("test.mp3")
        True
    """
    audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}
    return Path(file_path).suffix.lower() in audio_extensions


def is_text_file(file_path: Union[str, Path]) -> bool:
    """
    判断是否为文本文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        如果是文本文件返回True，否则返回False
    
    Example:
        >>> is_text_file("test.txt")
        True
    """
    text_extensions = {".txt", ".md", ".json", ".xml", ".csv", ".log", ".py", ".js", ".html", ".css"}
    return Path(file_path).suffix.lower() in text_extensions

