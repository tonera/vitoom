"""
用户认证相关的Pydantic模型
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional


class UserRegisterRequest(BaseModel):
    """用户注册请求"""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="Password (minimum 6 characters)")
    nickname: Optional[str] = Field(None, max_length=100, description="User nickname")
    
    @validator('password')
    def validate_password(cls, v):
        """验证密码强度"""
        if len(v) < 6:
            raise ValueError('密码长度至少6位')
        return v


class UserLoginRequest(BaseModel):
    """用户登录请求"""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Token响应"""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")


class UserInfoResponse(BaseModel):
    """用户信息响应"""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    nickname: Optional[str] = Field(None, description="User nickname")
    status: str = Field(..., description="User status")
    is_admin: bool = Field(..., description="Whether the user is an admin")
    created_at: str = Field(..., description="Creation timestamp")


class RefreshTokenRequest(BaseModel):
    """刷新Token请求"""
    refresh_token: str = Field(..., description="Refresh token")


class UpdateProfileRequest(BaseModel):
    """当前用户更新资料"""
    nickname: Optional[str] = Field(None, max_length=100, description="User nickname")
    new_password: Optional[str] = Field(None, min_length=6, description="New password (minimum 6 characters)")

    @validator("new_password")
    def validate_new_password(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if len(value) < 6:
            raise ValueError("密码长度至少6位")
        return value
