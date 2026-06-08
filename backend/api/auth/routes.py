"""
用户认证API路由
"""
from fastapi import APIRouter, Depends
from backend.api.auth.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    RefreshTokenRequest,
    UpdateProfileRequest,
)
from backend.api.auth.service import (
    register_user,
    login_user,
    get_user_by_id,
    refresh_user_token,
    update_user_profile,
)
from backend.auth import get_current_user_id
from backend.core.logger import get_app_logger
from backend.core.response import ok

logger = get_app_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", status_code=201)
async def register(request: UserRegisterRequest):
    """
    用户注册
    
    - **email**: 用户邮箱（唯一）
    - **password**: 密码（最少6位）
    - **nickname**: 用户昵称（可选）
    """
    user_dict = register_user(
        email=request.email,
        password=request.password,
        nickname=request.nickname
    )
    
    return ok(
        data={
            "id": user_dict["id"],
            "email": user_dict["email"],
            "nickname": user_dict.get("nickname"),
            "status": user_dict["status"],
            "is_admin": user_dict.get("is_admin", False),
            "created_at": user_dict["created_at"],
        },
        msg="registered",
    )


@router.post("/login")
async def login(request: UserLoginRequest):
    """
    用户登录
    
    - **email**: 用户邮箱
    - **password**: 密码
    
    返回访问Token和刷新Token
    """
    result = login_user(request.email, request.password)
    
    return ok(
        data={
            "access_token": result["access_token"],
            "refresh_token": result["refresh_token"],
            "token_type": result["token_type"],
        },
        msg="ok",
    )


@router.post("/refresh")
async def refresh_token(request: RefreshTokenRequest):
    """
    刷新Token
    
    - **refresh_token**: 刷新Token
    
    返回新的访问Token
    """
    result = refresh_user_token(request.refresh_token)
    
    return ok(
        data={
            "access_token": result["access_token"],
            "token_type": result["token_type"],
        },
        msg="ok",
    )


@router.get("/me")
async def get_current_user(user_id: str = Depends(get_current_user_id)):
    """
    获取当前用户信息
    
    需要认证
    """
    user_dict = get_user_by_id(user_id)
    
    if not user_dict:
        from backend.core.exceptions import UserNotFoundException
        raise UserNotFoundException(user_id)
    
    return ok(
        data={
            "id": user_dict["id"],
            "email": user_dict["email"],
            "nickname": user_dict.get("nickname"),
            "status": user_dict["status"],
            "is_admin": user_dict.get("is_admin", False),
            "created_at": user_dict["created_at"],
        },
        msg="ok",
    )


def _user_info_payload(user_dict: dict) -> dict:
    return {
        "id": user_dict["id"],
        "email": user_dict["email"],
        "nickname": user_dict.get("nickname"),
        "status": user_dict["status"],
        "is_admin": user_dict.get("is_admin", False),
        "created_at": user_dict["created_at"],
    }


@router.put("/me")
async def update_current_user(
    request: UpdateProfileRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    更新当前用户资料

    - **nickname**: 用户昵称（可选）
    - **new_password**: 新密码（可选，最少6位；两次输入一致由前端校验）
    """
    user_dict = update_user_profile(user_id, request.model_dump(exclude_unset=True))
    return ok(data=_user_info_payload(user_dict), msg="updated")


@router.post("/logout")
async def logout(user_id: str = Depends(get_current_user_id)):
    """
    用户登出
    
    注意：由于使用JWT Token，服务端无法主动使Token失效
    客户端应删除存储的Token
    """
    logger.info(f"User logged out: {user_id}")
    return ok(data={"user_id": user_id}, msg="logged_out")

