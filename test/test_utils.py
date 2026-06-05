"""
工具函数模块测试脚本
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils import (
    # UUID工具
    generate_uuid,
    generate_uuid4,
    is_valid_uuid,
    # 时间处理工具
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
    # 文件操作工具
    ensure_dir,
    ensure_file_dir,
    get_file_size,
    get_file_hash,
    read_file,
    write_file,
    copy_file,
    delete_file,
    list_files,
    safe_filename,
    get_file_extension,
    is_image_file,
    is_video_file,
    is_audio_file,
    is_text_file,
    # 加密工具
    hash_password,
    verify_password,
    generate_random_string,
    generate_token,
    hash_string,
)


def test_uuid_utils():
    """测试UUID工具"""
    print("=" * 50)
    print("Testing UUID Utils")
    print("=" * 50)
    
    # 生成UUID
    uuid1 = generate_uuid()
    uuid2 = generate_uuid4()
    
    assert len(uuid1) == 32
    assert len(uuid2) == 32
    assert uuid1 != uuid2
    
    # 验证UUID
    assert is_valid_uuid("550e8400-e29b-41d4-a716-446655440000")
    assert not is_valid_uuid("invalid-uuid")
    
    print(f"✓ Generated UUID: {uuid1}")
    print(f"✓ Generated UUID4: {uuid2}")
    print("UUID utils test passed!\n")


def test_datetime_utils():
    """测试时间处理工具"""
    print("=" * 50)
    print("Testing Datetime Utils")
    print("=" * 50)
    
    # 获取当前时间
    current_time = now()
    utc_time = utc_now()
    
    assert isinstance(current_time, datetime)
    assert isinstance(utc_time, datetime)
    
    # 格式化时间
    formatted = format_datetime(current_time)
    assert isinstance(formatted, str)
    
    # 解析时间
    parsed = parse_datetime("2023-12-07 14:30:00")
    assert parsed.year == 2023
    assert parsed.month == 12
    assert parsed.day == 7
    
    # 时间戳转换
    timestamp = datetime_to_timestamp(utc_time)
    assert isinstance(timestamp, float)
    
    dt_from_ts = timestamp_to_datetime(timestamp)
    assert isinstance(dt_from_ts, datetime)
    
    # 时间加减
    future_dt = add_days(current_time, 7)
    assert (future_dt - current_time).days == 7
    
    future_dt = add_hours(current_time, 2)
    assert (future_dt - current_time).total_seconds() >= 7200
    
    # 相对时间
    past_dt = current_time - timedelta(minutes=5)
    ago_str = time_ago(past_dt)
    assert "分钟前" in ago_str or "秒前" in ago_str
    
    # 过期检查
    expired_dt = current_time - timedelta(hours=2)
    assert is_expired(expired_dt, expire_seconds=3600)
    
    print(f"✓ Current time: {format_datetime(current_time)}")
    print(f"✓ UTC time: {format_datetime(utc_time)}")
    print(f"✓ Time ago: {ago_str}")
    print("Datetime utils test passed!\n")


def test_file_utils():
    """测试文件操作工具"""
    print("=" * 50)
    print("Testing File Utils")
    print("=" * 50)
    
    import tempfile
    import os
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_dir"
        test_file = test_dir / "test.txt"
        test_content = "Hello, World!"
        
        # 确保目录存在
        ensure_dir(test_dir)
        assert test_dir.exists()
        
        # 确保文件目录存在
        ensure_file_dir(test_file)
        assert test_file.parent.exists()
        
        # 写入文件
        write_file(test_file, test_content)
        assert test_file.exists()
        
        # 读取文件
        content = read_file(test_file)
        assert content == test_content
        
        # 获取文件大小
        size = get_file_size(test_file)
        assert size > 0
        
        # 获取文件哈希
        file_hash = get_file_hash(test_file)
        assert len(file_hash) == 64  # SHA256
        
        # 复制文件
        copied_file = test_dir / "copied.txt"
        copy_file(test_file, copied_file)
        assert copied_file.exists()
        assert read_file(copied_file) == test_content
        
        # 删除文件
        assert delete_file(copied_file)
        assert not copied_file.exists()
        
        # 列出文件
        files = list_files(test_dir)
        assert len(files) > 0
        
        # 安全文件名
        safe_name = safe_filename("test/file.txt")
        assert "/" not in safe_name
        
        # 文件扩展名
        ext = get_file_extension(test_file)
        assert ext == "txt"
        
        # 文件类型判断
        assert is_text_file(test_file)
        assert not is_image_file(test_file)
        
        # 创建图片文件测试
        image_file = test_dir / "test.jpg"
        write_file(image_file, "fake image")
        assert is_image_file(image_file)
        
        # 创建视频文件测试
        video_file = test_dir / "test.mp4"
        write_file(video_file, "fake video")
        assert is_video_file(video_file)
        
        # 创建音频文件测试
        audio_file = test_dir / "test.mp3"
        write_file(audio_file, "fake audio")
        assert is_audio_file(audio_file)
    
    print("✓ File operations test passed")
    print("File utils test passed!\n")


def test_crypto_utils():
    """测试加密工具"""
    print("=" * 50)
    print("Testing Crypto Utils")
    print("=" * 50)
    
    # 密码哈希（bcrypt限制密码长度不超过72字节）
    password = "mypassword123"
    hashed = hash_password(password)
    assert len(hashed) > 0
    assert hashed != password
    
    # 验证密码
    assert verify_password(password, hashed)
    assert not verify_password("wrongpassword", hashed)
    
    # 生成随机字符串
    random_str = generate_random_string(16)
    assert len(random_str) >= 16  # token_hex generates hex chars
    
    # 生成token
    token = generate_token(32)
    assert len(token) > 0
    
    # 字符串哈希
    hash_value = hash_string("hello")
    assert len(hash_value) == 64  # SHA256
    
    print(f"✓ Password hashed: {hashed[:20]}...")
    print(f"✓ Random string: {random_str}")
    print(f"✓ Token: {token[:20]}...")
    print(f"✓ String hash: {hash_value}")
    print("Crypto utils test passed!\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("Utils Module Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_uuid_utils()
        test_datetime_utils()
        test_file_utils()
        test_crypto_utils()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

