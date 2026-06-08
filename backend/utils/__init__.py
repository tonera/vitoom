from .url_utils import get_public_base_url, to_absolute_outputs_url
"""
工具函数模块
提供通用的工具函数，包括UUID生成、时间处理、文件操作、加密工具、HTTP客户端等
"""
from .uuid_utils import (
    generate_uuid,
    generate_uuid4,
    is_valid_uuid,
)
from .datetime_utils import (
    now,
    utc_now,
    format_datetime,
    parse_datetime,
    datetime_to_timestamp,
    timestamp_to_datetime,
    add_days,
    add_hours,
    add_minutes,
    time_ago,
    is_expired,
)
from .file_utils import (
    ensure_dir,
    ensure_file_dir,
    get_file_size,
    get_file_hash,
    read_file,
    write_file,
    copy_file,
    move_file,
    delete_file,
    list_files,
    safe_filename,
    get_file_extension,
    is_image_file,
    is_video_file,
    is_audio_file,
    is_text_file,
)
from .crypto_utils import (
    hash_password,
    verify_password,
    generate_random_string,
    generate_token,
    hash_string,
)
from .http_utils import (
    HTTPClient,
    get_http_client,
    request,
    get,
    post,
    put,
    delete,
    download_file,
)

__all__ = [
    # UUID工具
    "generate_uuid",
    "generate_uuid4",
    "is_valid_uuid",
    # 时间处理工具
    "now",
    "utc_now",
    "format_datetime",
    "parse_datetime",
    "datetime_to_timestamp",
    "timestamp_to_datetime",
    "add_days",
    "add_hours",
    "add_minutes",
    "time_ago",
    "is_expired",
    # 文件操作工具
    "ensure_dir",
    "ensure_file_dir",
    "get_file_size",
    "get_file_hash",
    "read_file",
    "write_file",
    "copy_file",
    "move_file",
    "delete_file",
    "list_files",
    "safe_filename",
    "get_file_extension",
    "is_image_file",
    "is_video_file",
    "is_audio_file",
    "is_text_file",
    # 加密工具
    "hash_password",
    "verify_password",
    "generate_random_string",
    "generate_token",
    "hash_string",
    # HTTP客户端工具
    "HTTPClient",
    "get_http_client",
    "request",
    "get",
    "post",
    "put",
    "delete",
    "download_file",
]
