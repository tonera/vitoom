"""
存储后端抽象与实现

目标：
- 按 request_params.storage 将推理产物上传到 local / server / s3 / oss
- 上层永远只使用 key（相对路径）做业务处理，不直接依赖 URL 或绝对路径
"""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Literal

import httpx

from .logger import get_logger

logger = get_logger(__name__)

StorageTarget = Literal["local", "server", "s3", "oss"]


class StorageBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class PutResult:
    key: str
    size: int
    content_type: str
    public_url: Optional[str] = None
    etag: Optional[str] = None


class StorageBackend:
    """存储后端抽象类（接口尽量小，便于实现）"""

    storage: StorageTarget

    async def put_file(
        self,
        *,
        key: str,
        local_path: Path,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PutResult:
        raise NotImplementedError


class LocalBackend(StorageBackend):
    storage: StorageTarget = "local"

    def __init__(self, outputs_dir: str):
        self.outputs_dir = Path(outputs_dir).resolve()
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    async def put_file(
        self,
        *,
        key: str,
        local_path: Path,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PutResult:
        """
        local 模式不做“上传”，只确保文件已落到 outputs_dir/key。
        若 local_path 不在目标位置，则拷贝过去。
        """

        dest = (self.outputs_dir / key).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        src = local_path.resolve()
        if src != dest:
            # 用线程避免阻塞事件循环
            await asyncio.to_thread(dest.write_bytes, src.read_bytes())

        size = dest.stat().st_size
        return PutResult(key=key, size=size, content_type=content_type)


class ServerBackend(StorageBackend):
    storage: StorageTarget = "server"

    def __init__(
        self,
        *,
        api_base_url: str,
        upload_path: str = "/api/inference/upload",
        timeout_seconds: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        auth_secret: str = "",
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.upload_url = f"{self.api_base_url}{upload_path}"
        self.timeout_seconds = timeout_seconds
        self.headers = headers or {}
        self.auth_secret = (auth_secret or "").strip()

    async def put_file(
        self,
        *,
        key: str,
        local_path: Path,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PutResult:
        meta = metadata or {}
        # 约定：后端实现先不做，这里先按 multipart 上传 key + file
        # 若后端未来字段名调整，仅需改这里，不影响上层。
        data: Dict[str, Any] = {"key": key, "content_type": content_type}
        for k, v in meta.items():
            if v is None:
                continue
            data[str(k)] = str(v)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                with open(local_path, "rb") as f:
                    files = {"file": (Path(key).name, f, content_type)}
                    # 可选签权：通过 header 传递时间戳与签名（避免污染 form 字段）
                    headers = dict(self.headers)
                    if self.auth_secret:
                        ts = str(int(time.time()))
                        canonical = f"{ts}\n{key}\n{content_type}\n".encode("utf-8")
                        sig = hmac.new(self.auth_secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
                        headers["X-Vitoom-Upload-Timestamp"] = ts
                        headers["X-Vitoom-Upload-Signature"] = sig
                    resp = await client.post(self.upload_url, data=data, files=files, headers=headers)
            resp.raise_for_status()
        except Exception as e:
            raise StorageBackendError(f"Server upload failed: key={key}, url={self.upload_url}, err={e}") from e

        size = local_path.stat().st_size
        return PutResult(key=key, size=size, content_type=content_type)


class S3Backend(StorageBackend):
    storage: StorageTarget = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        region: Optional[str] = None,
        endpoint: Optional[str] = None,
        public_base_url: Optional[str] = None,
    ):
        self.bucket = bucket
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.endpoint = endpoint
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None

        try:
            import boto3  # type: ignore
        except Exception as e:
            raise StorageBackendError("boto3 not installed, cannot use s3 storage") from e

        session = boto3.session.Session(
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region,
        )
        self._client = session.client("s3", endpoint_url=self.endpoint)

    async def put_file(
        self,
        *,
        key: str,
        local_path: Path,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PutResult:
        extra: Dict[str, Any] = {"ContentType": content_type}
        if metadata:
            # S3 metadata 只接受 str->str
            extra["Metadata"] = {str(k): str(v) for k, v in metadata.items() if v is not None}

        try:
            await asyncio.to_thread(
                self._client.upload_file,
                str(local_path),
                self.bucket,
                key,
                ExtraArgs=extra,
            )
        except Exception as e:
            raise StorageBackendError(f"S3 upload failed: bucket={self.bucket}, key={key}, err={e}") from e

        size = local_path.stat().st_size
        public_url = f"{self.public_base_url}/{key}" if self.public_base_url else None
        return PutResult(key=key, size=size, content_type=content_type, public_url=public_url)


class OSSBackend(StorageBackend):
    storage: StorageTarget = "oss"

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key_id: str,
        access_key_secret: str,
        public_base_url: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.bucket = bucket
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None

        try:
            import oss2  # type: ignore
        except Exception as e:
            raise StorageBackendError("oss2 not installed, cannot use oss storage") from e

        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self._bucket = oss2.Bucket(auth, self.endpoint, self.bucket)

    async def put_file(
        self,
        *,
        key: str,
        local_path: Path,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PutResult:
        headers: Dict[str, str] = {"Content-Type": content_type}
        if metadata:
            # OSS 允许 x-oss-meta-*，oss2 支持直接传 headers
            for k, v in metadata.items():
                if v is None:
                    continue
                headers[f"x-oss-meta-{k}"] = str(v)

        try:
            await asyncio.to_thread(self._bucket.put_object_from_file, key, str(local_path), headers=headers)
        except Exception as e:
            raise StorageBackendError(f"OSS upload failed: bucket={self.bucket}, key={key}, err={e}") from e

        size = local_path.stat().st_size
        public_url = f"{self.public_base_url}/{key}" if self.public_base_url else None
        return PutResult(key=key, size=size, content_type=content_type, public_url=public_url)


def build_storage_backend(
    *,
    storage: StorageTarget,
    inference_config: Any,
) -> StorageBackend:
    """
    inference_config：来自 inference/common/config_loader.py 的 InferenceConfig
    """
    if storage == "local":
        return LocalBackend(outputs_dir=getattr(inference_config, "outputs_dir", "resources/outputs"))

    if storage == "server":
        return ServerBackend(
            api_base_url=getattr(inference_config, "api_base_url", "http://127.0.0.1:8888"),
            upload_path=getattr(inference_config, "server_upload_path", "/api/inference/upload"),
            timeout_seconds=float(getattr(inference_config, "server_upload_timeout_seconds", 60.0)),
            auth_secret=getattr(inference_config, "server_upload_auth_secret", "") or "",
        )

    if storage == "s3":
        return S3Backend(
            bucket=getattr(inference_config, "s3_bucket", ""),
            access_key_id=getattr(inference_config, "s3_access_key_id", ""),
            secret_access_key=getattr(inference_config, "s3_secret_access_key", ""),
            region=getattr(inference_config, "s3_region", None),
            endpoint=getattr(inference_config, "s3_endpoint", None),
            public_base_url=getattr(inference_config, "s3_public_base_url", None),
        )

    if storage == "oss":
        return OSSBackend(
            endpoint=getattr(inference_config, "oss_endpoint", ""),
            bucket=getattr(inference_config, "oss_bucket", ""),
            access_key_id=getattr(inference_config, "oss_access_key_id", ""),
            access_key_secret=getattr(inference_config, "oss_access_key_secret", ""),
            public_base_url=getattr(inference_config, "oss_public_base_url", None),
        )

    raise StorageBackendError(f"Unsupported storage type: {storage}")


