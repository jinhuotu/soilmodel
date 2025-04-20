# src/utils/auth.py
import os
from datetime import datetime, timedelta
from typing import Optional, Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

# 项目内部依赖
from src.db.database import get_db
from src.db.models import User
from src.config.settings import settings
from src.api.utils.logger import logger


# ------------------------------------------
# 认证核心配置
# ------------------------------------------

class TokenData(BaseModel):
    """JWT 令牌数据模型"""
    username: Optional[str] = None


class AuthConfig:
    """认证配置类"""
    SECRET_KEY = settings.SECRET_KEY  # 从环境变量读取
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30


# 密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 方案
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    auto_error=False  # 允许匿名访问
)


# ------------------------------------------
# 工具函数
# ------------------------------------------

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码哈希"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return pwd_context.hash(password)


def create_access_token(
        data: dict,
        expires_delta: Optional[timedelta] = None
) -> str:
    """生成JWT访问令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(
        minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES
    ))
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        AuthConfig.SECRET_KEY,
        algorithm=AuthConfig.ALGORITHM
    )


async def validate_token(
        db: Session,
        token: Optional[str] = Depends(oauth2_scheme)
) -> User:
    """验证JWT令牌并返回用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        if not token:
            raise credentials_exception

        payload = jwt.decode(
            token,
            AuthConfig.SECRET_KEY,
            algorithms=[AuthConfig.ALGORITHM]
        )
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        logger.error(f"JWT验证失败: {str(e)}")
        raise credentials_exception

    user = db.query(User).filter(User.username == token_data.username).first()
    if not user:
        logger.warning(f"用户不存在: {token_data.username}")
        raise credentials_exception

    return user


# ------------------------------------------
# FastAPI 依赖项
# ------------------------------------------

async def get_current_user(
        db: Session = Depends(get_db),
        token: str = Depends(oauth2_scheme)
) -> User:
    """获取当前认证用户（严格模式）"""
    return await validate_token(db, token)


async def get_current_active_user(
        current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """验证用户是否激活"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用"
        )
    return current_user


async def get_current_admin_user(
        current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """验证管理员权限"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return current_user


# ------------------------------------------
# 辅助功能
# ------------------------------------------

def authenticate_user(
        db: Session,
        username: str,
        password: str
) -> Optional[User]:
    """用户认证逻辑"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        # 防止计时攻击
        pwd_context.dummy_verify()
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user