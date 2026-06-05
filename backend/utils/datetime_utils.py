"""
时间处理工具
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Union
import pytz


def now() -> datetime:
    """
    获取当前本地时间
    
    Returns:
        当前本地时间的datetime对象
    
    Example:
        >>> current_time = now()
        >>> isinstance(current_time, datetime)
        True
    """
    return datetime.now()


def utc_now() -> datetime:
    """
    获取当前UTC时间
    
    Returns:
        当前UTC时间的datetime对象
    
    Example:
        >>> utc_time = utc_now()
        >>> utc_time.tzinfo == timezone.utc
        True
    """
    return datetime.now(timezone.utc)


def format_datetime(
    dt: datetime,
    format_str: str = "%Y-%m-%d %H:%M:%S",
    timezone_str: Optional[str] = None
) -> str:
    """
    格式化datetime对象为字符串
    
    Args:
        dt: datetime对象
        format_str: 格式化字符串，默认为 "%Y-%m-%d %H:%M:%S"
        timezone_str: 时区字符串（如 "Asia/Shanghai"），如果为None则使用dt的时区
    
    Returns:
        格式化后的时间字符串
    
    Example:
        >>> dt = datetime(2023, 12, 7, 14, 30, 0)
        >>> format_datetime(dt)
        '2023-12-07 14:30:00'
        >>> format_datetime(dt, "%Y-%m-%d")
        '2023-12-07'
    """
    if timezone_str:
        tz = pytz.timezone(timezone_str)
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        dt = dt.astimezone(tz)
    
    return dt.strftime(format_str)


def parse_datetime(
    datetime_str: str,
    format_str: str = "%Y-%m-%d %H:%M:%S",
    timezone_str: Optional[str] = None
) -> datetime:
    """
    解析时间字符串为datetime对象
    
    Args:
        datetime_str: 时间字符串
        format_str: 格式化字符串，默认为 "%Y-%m-%d %H:%M:%S"
        timezone_str: 时区字符串（如 "Asia/Shanghai"），如果为None则解析为naive datetime
    
    Returns:
        datetime对象
    
    Example:
        >>> dt = parse_datetime("2023-12-07 14:30:00")
        >>> dt.year == 2023
        True
    """
    dt = datetime.strptime(datetime_str, format_str)
    
    if timezone_str:
        tz = pytz.timezone(timezone_str)
        dt = tz.localize(dt)
    
    return dt


def datetime_to_timestamp(dt: datetime) -> float:
    """
    将datetime对象转换为时间戳（秒）
    
    Args:
        dt: datetime对象
    
    Returns:
        时间戳（秒，浮点数）
    
    Example:
        >>> dt = datetime(2023, 12, 7, 14, 30, 0, tzinfo=timezone.utc)
        >>> timestamp = datetime_to_timestamp(dt)
        >>> isinstance(timestamp, float)
        True
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def timestamp_to_datetime(timestamp: float, tz: Optional[timezone] = None) -> datetime:
    """
    将时间戳转换为datetime对象
    
    Args:
        timestamp: 时间戳（秒）
        tz: 时区，如果为None则使用UTC
    
    Returns:
        datetime对象
    
    Example:
        >>> dt = timestamp_to_datetime(1701964200.0)
        >>> isinstance(dt, datetime)
        True
    """
    if tz is None:
        tz = timezone.utc
    return datetime.fromtimestamp(timestamp, tz=tz)


def add_days(dt: datetime, days: int) -> datetime:
    """
    给datetime对象添加天数
    
    Args:
        dt: datetime对象
        days: 要添加的天数（可以是负数）
    
    Returns:
        新的datetime对象
    
    Example:
        >>> dt = datetime(2023, 12, 7)
        >>> new_dt = add_days(dt, 7)
        >>> new_dt.day == 14
        True
    """
    return dt + timedelta(days=days)


def add_hours(dt: datetime, hours: int) -> datetime:
    """
    给datetime对象添加小时数
    
    Args:
        dt: datetime对象
        hours: 要添加的小时数（可以是负数）
    
    Returns:
        新的datetime对象
    
    Example:
        >>> dt = datetime(2023, 12, 7, 14, 0, 0)
        >>> new_dt = add_hours(dt, 2)
        >>> new_dt.hour == 16
        True
    """
    return dt + timedelta(hours=hours)


def add_minutes(dt: datetime, minutes: int) -> datetime:
    """
    给datetime对象添加分钟数
    
    Args:
        dt: datetime对象
        minutes: 要添加的分钟数（可以是负数）
    
    Returns:
        新的datetime对象
    
    Example:
        >>> dt = datetime(2023, 12, 7, 14, 30, 0)
        >>> new_dt = add_minutes(dt, 15)
        >>> new_dt.minute == 45
        True
    """
    return dt + timedelta(minutes=minutes)


def time_ago(dt: datetime) -> str:
    """
    获取相对时间描述（如"5分钟前"、"2小时前"）
    
    Args:
        dt: datetime对象
    
    Returns:
        相对时间描述字符串
    
    Example:
        >>> dt = now() - timedelta(minutes=5)
        >>> time_ago(dt)
        '5分钟前'
    """
    now_dt = now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    
    delta = now_dt - dt
    
    if delta.total_seconds() < 0:
        return "未来"
    
    seconds = int(delta.total_seconds())
    
    if seconds < 60:
        return f"{seconds}秒前"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}分钟前"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours}小时前"
    elif seconds < 2592000:
        days = seconds // 86400
        return f"{days}天前"
    elif seconds < 31536000:
        months = seconds // 2592000
        return f"{months}个月前"
    else:
        years = seconds // 31536000
        return f"{years}年前"


def is_expired(dt: datetime, expire_seconds: Optional[int] = None) -> bool:
    """
    检查datetime对象是否已过期
    
    Args:
        dt: datetime对象
        expire_seconds: 过期时间（秒），如果为None则检查是否小于当前时间
    
    Returns:
        如果已过期返回True，否则返回False
    
    Example:
        >>> dt = now() - timedelta(hours=1)
        >>> is_expired(dt, expire_seconds=3600)
        True
    """
    now_dt = now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    
    if expire_seconds is None:
        return dt < now_dt
    
    delta = (now_dt - dt).total_seconds()
    return delta > expire_seconds

