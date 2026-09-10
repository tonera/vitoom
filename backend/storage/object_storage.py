"""
S3 / R2 / OSS 对象存储适配器（Backend 写入与删除）。
上传仅需 endpoint/bucket/密钥；访问 URL 由 artifact_storage.resolve_artifact_public_url 生成。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from backend.core.config import get_config
from backend.core.logger import get_app_logger
from backend.storage.adapter import StorageAdapter
from backend.utils import generate_uuid
from backend.utils.artifact_storage import resolve_artifact_public_url

logger = get_app_logger(__name__)


class _ObjectStorageAdapterBase(StorageAdapter):
    """S3/R2/OSS 适配器公共逻辑（按 key 存取）。"""

    storage_name: str = "s3"

    def _generate_storage_path(self, category: str, filename: str) -> str:
        now = datetime.utcnow()
        date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
        ext = Path(filename).suffix or ""
        unique = f"{generate_uuid()}{ext}"
        return f"{category}/{date_path}/{unique}"

    def _public_url(self, key: str) -> Optional[str]:
        return resolve_artifact_public_url(self.storage_name, key)

    async def get_file_url(self, storage_path: str) -> str:
        url = self._public_url(storage_path)
        return url or ""

    async def put_bytes(self, key: str, file_data: bytes, content_type: str) -> None:
        raise NotImplementedError

    async def delete_object(self, key: str) -> bool:
        raise NotImplementedError

    async def object_exists(self, key: str) -> bool:
        raise NotImplementedError

    async def get_object_size(self, key: str) -> int:
        raise NotImplementedError

    async def get_object_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    async def save_file(
        self,
        file_data: bytes,
        category: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        key = self._generate_storage_path(category, filename)
        await self.put_bytes(key, file_data, _guess_content_type(filename))
        http_url = self._public_url(key) or ""
        return {
            "storage_path": key,
            "http_url": http_url,
            "file_size": len(file_data),
        }

    async def delete_file(self, storage_path: str) -> bool:
        return await self.delete_object(storage_path)

    async def file_exists(self, storage_path: str) -> bool:
        return await self.object_exists(storage_path)

    async def get_file_size(self, storage_path: str) -> int:
        return await self.get_object_size(storage_path)

    async def read_file(self, storage_path: str) -> bytes:
        return await self.get_object_bytes(storage_path)


def _guess_content_type(filename: str) -> str:
    import mimetypes

    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def _create_boto3_s3_client(
    *,
    access_key_id: str,
    secret_access_key: str,
    region: Optional[str],
    endpoint: Optional[str],
    r2_compat: bool = False,
    storage_name: str = "s3",
):
    try:
        import boto3  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"boto3 not installed, cannot use {storage_name} storage") from e

    session = boto3.session.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
    )
    client_kwargs: Dict[str, Any] = {"endpoint_url": endpoint}
    if r2_compat:
        from botocore.config import Config  # type: ignore

        checksum_kwargs = {
            "signature_version": "s3v4",
            "s3": {"addressing_style": "path"},
            "request_checksum_calculation": "when_required",
            "response_checksum_validation": "when_required",
        }
        try:
            client_kwargs["config"] = Config(**checksum_kwargs)
        except TypeError:
            client_kwargs["config"] = Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            )
    return session.client("s3", **client_kwargs)


class S3StorageAdapter(_ObjectStorageAdapterBase):
    storage_name = "s3"
    _config_key = "s3"
    _r2_compat = False
    _require_endpoint = False
    _default_region: Optional[str] = None

    def __init__(self) -> None:
        key = self._config_key
        self.bucket = str(get_config(f"storage.{key}.bucket", "") or "").strip()
        self.access_key_id = str(get_config(f"storage.{key}.access_key_id", "") or "")
        self.secret_access_key = str(get_config(f"storage.{key}.secret_access_key", "") or "")
        region = get_config(f"storage.{key}.region", None)
        region_str = str(region).strip() if region else ""
        self.region = region_str or self._default_region
        endpoint = get_config(f"storage.{key}.endpoint", None)
        self.endpoint = str(endpoint).strip() if endpoint else None
        if not self.bucket:
            raise ValueError(f"storage.{key}.bucket is required for {self.storage_name} storage")
        if self._require_endpoint and not self.endpoint:
            raise ValueError(f"storage.{key}.endpoint is required for {self.storage_name} storage")
        self._client = _create_boto3_s3_client(
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            region=self.region,
            endpoint=self.endpoint,
            r2_compat=self._r2_compat,
            storage_name=self.storage_name,
        )
        logger.info("%s initialized: bucket=%s", type(self).__name__, self.bucket)

    async def put_bytes(self, key: str, file_data: bytes, content_type: str) -> None:
        extra: Dict[str, Any] = {"ContentType": content_type}

        def _upload() -> None:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=file_data,
                **extra,
            )

        await asyncio.to_thread(_upload)

    async def delete_object(self, key: str) -> bool:
        def _delete() -> None:
            self._client.delete_object(Bucket=self.bucket, Key=key)

        try:
            await asyncio.to_thread(_delete)
            return True
        except Exception as e:
            logger.warning("%s delete failed: key=%s err=%s", self.storage_name, key, e)
            return False

    async def object_exists(self, key: str) -> bool:
        def _head() -> bool:
            try:
                self._client.head_object(Bucket=self.bucket, Key=key)
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_head)

    async def get_object_size(self, key: str) -> int:
        def _head() -> int:
            resp = self._client.head_object(Bucket=self.bucket, Key=key)
            return int(resp.get("ContentLength") or 0)

        return await asyncio.to_thread(_head)

    async def get_object_bytes(self, key: str) -> bytes:
        def _get() -> bytes:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()

        return await asyncio.to_thread(_get)


class R2StorageAdapter(S3StorageAdapter):
    """Cloudflare R2：S3 兼容协议，独立配置段 storage.r2。"""

    storage_name = "r2"
    _config_key = "r2"
    _r2_compat = True
    _require_endpoint = True
    _default_region = "auto"


class OSSStorageAdapter(_ObjectStorageAdapterBase):
    storage_name = "oss"

    def __init__(self) -> None:
        self.endpoint = str(get_config("storage.oss.endpoint", "") or "").strip()
        self.bucket = str(get_config("storage.oss.bucket", "") or "").strip()
        self.access_key_id = str(get_config("storage.oss.access_key_id", "") or "")
        self.access_key_secret = str(get_config("storage.oss.access_key_secret", "") or "")
        if not self.endpoint or not self.bucket:
            raise ValueError("storage.oss.endpoint and storage.oss.bucket are required for OSS storage")
        try:
            import oss2  # type: ignore
        except ImportError as e:
            raise RuntimeError("oss2 not installed, cannot use oss storage") from e
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self._bucket = oss2.Bucket(auth, self.endpoint, self.bucket)
        logger.info("OSSStorageAdapter initialized: bucket=%s", self.bucket)

    async def put_bytes(self, key: str, file_data: bytes, content_type: str) -> None:
        headers = {"Content-Type": content_type}

        def _put() -> None:
            self._bucket.put_object(key, file_data, headers=headers)

        await asyncio.to_thread(_put)

    async def delete_object(self, key: str) -> bool:
        try:
            await asyncio.to_thread(self._bucket.delete_object, key)
            return True
        except Exception as e:
            logger.warning("OSS delete failed: key=%s err=%s", key, e)
            return False

    async def object_exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._bucket.object_exists, key)

    async def get_object_size(self, key: str) -> int:
        meta = await asyncio.to_thread(self._bucket.head_object, key)
        return int(getattr(meta, "content_length", 0) or 0)

    async def get_object_bytes(self, key: str) -> bytes:
        result = await asyncio.to_thread(self._bucket.get_object, key)
        return result.read()


async def put_bytes_at_key(storage: str, key: str, file_data: bytes, content_type: str) -> None:
    """按指定 storage 将字节写入 key（用于用户上传等固定路径场景）。"""
    from backend.storage.factory import create_storage_adapter
    from backend.utils.artifact_storage import normalize_storage_for_write

    mode = normalize_storage_for_write(storage)
    if mode == "server":
        base = Path(get_config("storage.local.base_path", "resources/outputs"))
        if not base.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            base = (project_root / base).resolve()
        dest = (base / key).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.relative_to(base.resolve())
        except ValueError as e:
            raise ValueError("invalid storage key") from e
        dest.write_bytes(file_data)
        return

    adapter = create_storage_adapter(mode)
    if not isinstance(adapter, _ObjectStorageAdapterBase):
        raise TypeError(f"adapter does not support put_bytes_at_key: {type(adapter)}")
    await adapter.put_bytes(key, file_data, content_type)
