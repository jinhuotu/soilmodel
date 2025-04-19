# src/core/models/base.py
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
from pydantic import BaseModel, Field, ValidationError, validator
import logging

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """大语言模型基础配置"""
    model_name: str = Field(..., min_length=1, description="模型标识名称")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="生成温度控制")
    max_tokens: int = Field(512, ge=1, description="最大生成token数")
    api_key: Optional[str] = Field(None, description="API访问密钥")
    api_base: Optional[str] = Field(None, description="自定义API地址")
    timeout: int = Field(30, ge=5, description="API调用超时时间(秒)")

    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0.1:
            logger.warning("极低温度值(%s)可能导致生成结果过于确定", v)
        return v


class BaseLLM(ABC):
    """大语言模型抽象基类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """执行模型特定的配置校验"""
        try:
            ModelConfig(**self.config.dict())
        except ValidationError as e:
            logger.error("模型配置校验失败: %s", e)
            raise

    @abstractmethod
    def generate(
            self,
            context: str,
            question: str,
            **kwargs: Any
    ) -> str:
        """同步生成答案
        Args:
            context: 检索到的上下文文本
            question: 用户问题
            kwargs: 模型特定参数
        Returns:
            生成的答案文本
        """
        raise NotImplementedError

    @abstractmethod
    async def agenerate(
            self,
            context: str,
            question: str,
            **kwargs: Any
    ) -> str:
        """异步生成答案"""
        raise NotImplementedError

    @abstractmethod
    def stream_generate(
            self,
            context: str,
            question: str,
            **kwargs: Any
    ) -> Generator[str, None, None]:
        """流式生成（逐token产出）"""
        yield ""

    @abstractmethod
    async def astream_generate(
            self,
            context: str,
            question: str,
            **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """异步流式生成"""
        yield ""

    @abstractmethod
    def get_embeddings(
            self,
            texts: List[str]
    ) -> List[List[float]]:
        """获取文本嵌入向量（用于检索步骤）
        Args:
            texts: 需要编码的文本列表
        Returns:
            各文本的嵌入向量列表
        """
        raise NotImplementedError


class GenerationError(Exception):
    """模型生成异常基类"""

    def __init__(
            self,
            model: str,
            error_type: str,
            message: str,
            status_code: Optional[int] = None
    ):
        super().__init__(f"[{model}] {error_type}: {message}")
        self.error_type = error_type
        self.status_code = status_code


# 示例实现：OpenAI 适配器
class OpenAIModel(BaseLLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from openai import OpenAI, AsyncOpenAI

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout
        )

    def _validate_config(self):
        if not self.config.api_key:
            raise ValueError("OpenAI 模型需要有效的 API Key")

    def _build_prompt(self, context: str, question: str) -> str:
        """构建OpenAI格式提示词"""
        return f"基于以下上下文：\n{context}\n\n请回答：{question}"

    def generate(self, context: str, question: str, **kwargs) -> str:
        prompt = self._build_prompt(context, question)
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise GenerationError(
                model="OpenAI",
                error_type="API_ERROR",
                message=str(e)
            )
            # 其他方法实现（agenerate, stream_generate等）