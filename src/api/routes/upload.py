# src/api/routes/upload.py
import os
import hashlib
from pathlib import Path
from typing import List, Annotated
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel

# 导入项目内部模块
from src.core.data_processing.loader import DocumentLoaderFactory
from src.core.data_processing.splitter import RecursiveTextSplitter
from src.core.models.vector_db import get_vector_store
from src.db.database import get_db
from src.db.models import User, FileMetadata
from src.utils.auth import validate_token
from src.utils.logger import logger
from src.config.settings import settings

router = APIRouter(prefix="/api/v1/upload", tags=["file_upload"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    message: str
    chunk_count: int = 0


class ErrorResponse(BaseModel):
    error: str
    detail: str


async def get_current_user(
        token: Annotated[str, Depends(oauth2_scheme)],
        db: Session = Depends(get_db)
) -> User:
    """验证JWT并获取用户"""
    return await validate_token(db, token)


def generate_file_id(file: UploadFile, user_id: str) -> str:
    """生成唯一文件标识"""
    raw_id = f"{user_id}-{file.filename}-{file.size}"
    return hashlib.sha256(raw_id.encode()).hexdigest()


async def validate_file(file: UploadFile):
    """文件验证逻辑"""
    # 允许的文件类型
    ALLOWED_TYPES = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "text/plain": "txt",
        "text/markdown": "md"
    }

    # 检查文件类型
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"不支持的文件类型: {file.content_type}"
        )

    # 检查文件大小 (最大100MB)
    max_size = 100 * 1024 * 1024
    if file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件大小超过限制 ({max_size // 1024 // 1024}MB)"
        )


async def save_upload_file(file: UploadFile, user_dir: Path) -> Path:
    """保存上传文件到用户目录"""
    try:
        file_path = user_dir / file.filename
        with file_path.open("wb") as buffer:
            while content := await file.read(1024 * 1024):  # 分块读取
                buffer.write(content)
        return file_path
    except Exception as e:
        logger.error(f"文件保存失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文件保存失败"
        )
    finally:
        await file.close()


async def process_file_content(
        file_path: Path,
        user_id: str,
        db: Session,
        vector_store
) -> int:
    """处理文件内容的核心逻辑"""
    try:
        # 1. 加载文档
        loader = DocumentLoaderFactory.get_loader(file_path)
        docs = loader.load()

        # 2. 分块处理
        splitter = RecursiveTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        chunks = splitter.split("\n".join(docs))

        # 3. 生成嵌入并存储
        vector_store.add_documents(
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[f"{user_id}-{file_path.name}-{i}" for i in range(len(chunks))]
        )

        # 4. 保存元数据到数据库
        db_file = FileMetadata(
            user_id=user_id,
            file_id=generate_file_id(file_path.name, user_id),
            filename=file_path.name,
            file_type=file_path.suffix,
            chunk_count=len(chunks),
            storage_path=str(file_path)
        )
        db.add(db_file)
        db.commit()

        return len(chunks)
    except Exception as e:
        db.rollback()
        logger.error(f"文件处理失败: {str(e)}")
        raise


@router.post(
    "",
    response_model=List[UploadResponse],
    responses={
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def upload_files(
        background_tasks: BackgroundTasks,
        files: Annotated[List[UploadFile], File(description="支持多文件上传")],
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
        vector_store=Depends(get_vector_store)
):
    """上传并处理文档文件"""
    user_dir = Path(settings.UPLOAD_DIR) / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    responses = []
    for file in files:
        file_id = generate_file_id(file, current_user.id)
        try:
            # 1. 验证文件
            await validate_file(file)

            # 2. 保存文件
            file_path = await save_upload_file(file, user_dir)

            # 3. 后台处理内容
            background_tasks.add_task(
                process_file_content,
                file_path,
                current_user.id,
                db,
                vector_store
            )

            responses.append(UploadResponse(
                file_id=file_id,
                filename=file.filename,
                status="accepted",
                message="文件已接收，正在处理"
            ))
        except HTTPException as e:
            responses.append(UploadResponse(
                file_id=file_id,
                filename=file.filename,
                status="error",
                message=e.detail
            ))

    return responses