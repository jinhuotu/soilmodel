# src/api/routes/chat.py
import logging
from typing import List, Optional, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

# 项目内部依赖
from src.core.models.base import BaseLLM, GenerationError
from src.core.models.vector_db import get_vector_store
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
from src.db.models import User
from src.api.utils.auth import get_current_user
from src.api.utils.logger import logger

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[str]] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: Optional[float]


PROMPT_TEMPLATE = """基于以下上下文和对话历史，请专业、简洁地回答用户问题。如果无法确定答案，请说明原因。
上下文：
{context}

对话历史：
{history}

问题：{question}
答案："""


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"description": "无效请求"},
        503: {"description": "服务暂时不可用"}
    }
)
async def chat(
        request: ChatRequest,
        llm: BaseLLM = Depends(get_llm),  # 需要实现get_llm依赖
        vector_store: BaseRetriever = Depends(get_vector_store),
        user: User = Depends(get_current_user)
):
    """处理用户问题并返回智能回答"""
    try:
        # 1. 检索相关文档
        docs = vector_store.get_relevant_documents(request.question)
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="未找到相关文档"
            )

        # 2. 构建提示词
        context = "\n".join([doc.page_content for doc in docs])
        history = "\n".join(request.chat_history or [])
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context,
            history=history,
            question=request.question
        )

        # 3. 生成回答
        answer = llm.generate(
            context=context,
            question=request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return {
            "answer": answer,
            "sources": list(set([doc.metadata.get("source", "") for doc in docs])),
            "confidence": calculate_confidence(answer, docs)  # 需实现置信度计算
        }

    except GenerationError as e:
        logger.error(f"生成失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型服务暂时不可用"
        )
    except Exception as e:
        logger.error(f"聊天处理异常: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="内部服务器错误"
        )


@router.post("/stream")
async def chat_stream(
        request: ChatRequest,
        llm: BaseLLM = Depends(get_llm),
        vector_store: BaseRetriever = Depends(get_vector_store)
) -> StreamingResponse:
    """流式问答接口"""

    async def event_generator():
        try:
            # 检索文档
            docs = vector_store.get_relevant_documents(request.question)
            context = "\n".join([doc.page_content for doc in docs])

            # 流式生成
            async for token in llm.astream_generate(
                    context=context,
                    question=request.question
            ):
                yield f"data: {token}\n\n"

            # 发送来源信息
            sources = list(set([doc.metadata.get("source", "") for doc in docs]))
            yield f"event: sources\ndata: {sources}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# 需要实现的工具函数
def calculate_confidence(answer: str, docs: List) -> float:
    """简单置信度计算（示例实现）"""
    from rapidfuzz import fuzz
    scores = [fuzz.token_sort_ratio(answer, doc.page_content) for doc in docs]
    return max(scores) / 100 if scores else 0.0


def get_llm() -> BaseLLM:
    """获取LLM实例（需根据配置实现）"""
    from src.core.models import OpenAIModel  # 示例使用OpenAI
    return OpenAIModel(config=load_llm_config())


def load_llm_config():
    """加载LLM配置（需结合项目配置系统实现）"""
    from src.config import settings
    return ModelConfig(
        model_name=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY
    )