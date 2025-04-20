# src/db/models.py
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from passlib.context import CryptContext

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(Base):
    """系统用户模型"""
    __tablename__ = "users"

    # 基础字段
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(200), nullable=False)

    # 状态标识
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系映射
    files = relationship("FileMetadata", back_populates="owner")

    def verify_password(self, plain_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, self.hashed_password)

    def generate_access_token(self) -> str:
        """生成JWT访问令牌（示例）"""
        from src.api.utils.auth import create_access_token  # 延迟导入避免循环依赖
        return create_access_token(data={"sub": self.username})


class FileMetadata(Base):
    """文件元数据模型"""
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(200), nullable=False)
    file_type = Column(String(10))
    file_size = Column(Integer)
    storage_path = Column(String(300), unique=True)

    # 关联用户
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="files")

    # 处理信息
    chunk_count = Column(Integer, default=0)
    processed_at = Column(DateTime)