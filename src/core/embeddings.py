# src/core/embeddings.py
import logging
from functools import lru_cache
from typing import Union, Optional, List
from pydantic import BaseModel, Field
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    CohereEmbeddings
)
from src.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""
    model_type: str = Field("openai", description="模型类型 (openai/huggingface/cohere/deepseek)")
    model_name: str = Field("text-embedding-3-small", description="模型名称或路径")
    api_key: Optional[str] = Field(None, description="API访问密钥")
    api_base: Optional[str] = Field(None, description="自定义API地址")
    model_kwargs: dict = Field(default_factory=dict, description="模型额外参数")


class DeepSeekEmbeddings:
    """DeepSeek 自定义嵌入模型"""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入"""
        # 实际调用 DeepSeek API 的实现
        # 示例伪代码：
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            "https://api.deepseek.com/v1/embeddings",
            json={"inputs": texts, "model": self.model_name},
            headers=headers
        )
        return [item['embedding'] for item in response.json()['data']]

    def embed_query(self, text: str) -> List[float]:
        """生成查询嵌入"""
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings, CohereEmbeddings]:
    """获取嵌入模型实例（单例模式）"""
    config = EmbeddingConfig(
        model_type=settings.EMBED_MODEL_TYPE,
        model_name=settings.EMBED_MODEL_NAME,
        api_key=settings.EMBED_API_KEY,
        api_base=settings.EMBED_API_BASE,
        model_kwargs=settings.EMBED_MODEL_KWARGS
    )

    try:
        if config.model_type == "openai":
            if not config.api_key:
                raise ValueError("OpenAI 模型需要 API 密钥")

            return OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=config.api_key,
                openai_api_base=config.api_base,
                **config.model_kwargs
            )

        elif config.model_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=config.model_name,
                model_kwargs=config.model_kwargs
            )

        elif config.model_type == "cohere":
            if not config.api_key:
                raise ValueError("Cohere 模型需要 API 密钥")

            return CohereEmbeddings(
                cohere_api_key=config.api_key,
                model=config.model_name,
                **config.model_kwargs
            )

        elif config.model_type == "deepseek":
            if not config.api_key:
                raise ValueError("DeepSeek 模型需要 API 密钥")

            return DeepSeekEmbeddings(
                model_name=config.model_name,
                api_key=config.api_key
            )

        else:
            raise ValueError(f"不支持的嵌入模型类型: {config.model_type}")

    except ImportError as e:
        logger.error(f"依赖库未安装: {str(e)}")
        raise RuntimeError("请安装所需的模型依赖库") from e
    except Exception as e:
        logger.error(f"嵌入模型初始化失败: {str(e)}")
        raise