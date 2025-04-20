import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, Field
from langchain.vectorstores import Chroma, FAISS, Pinecone
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from src.config.settings import settings

logger = logging.getLogger(__name__)


class VectorDBConfig(BaseModel):
    """向量数据库基础配置"""
    db_type: str = Field("chroma", description="向量数据库类型 (chroma/faiss/pinecone)")
    persist_dir: str = Field("data/vector_db", description="本地持久化目录")
    index_name: str = Field("default", description="索引名称（Pinecone专用）")
    api_key: Optional[str] = Field(None, description="云服务API密钥")
    host: Optional[str] = Field(None, description="数据库服务器地址")
    embedding: Embeddings = Field(..., description="嵌入模型实例")


class VectorDBFactory:
    """向量数据库工厂类"""

    _registry: Dict[str, Type[VectorStore]] = {
        "chroma": Chroma,
        "faiss": FAISS,
        "pinecone": Pinecone
    }

    @classmethod
    def create(
            cls,
            config: VectorDBConfig,
            documents: Optional[list] = None,
            metadatas: Optional[list] = None,
            **kwargs: Any
    ) -> VectorStore:
        """创建或加载向量数据库"""
        try:
            db_class = cls._registry[config.db_type.lower()]
        except KeyError:
            raise ValueError(f"不支持的向量数据库类型: {config.db_type}")

        logger.info(f"正在初始化 {config.db_type} 向量数据库...")

        if config.db_type.lower() == "chroma":
            return cls._init_chroma(db_class, config, documents, metadatas)
        elif config.db_type.lower() == "faiss":
            return cls._init_faiss(db_class, config, documents, metadatas)
        elif config.db_type.lower() == "pinecone":
            return cls._init_pinecone(db_class, config, documents, metadatas)
        else:
            raise NotImplementedError(f"{config.db_type} 尚未实现")

    @staticmethod
    def _init_chroma(
            db_class: Type[Chroma],
            config: VectorDBConfig,
            documents: list,
            metadatas: list
    ) -> Chroma:
        """初始化 Chroma 数据库"""
        persist_dir = Path(config.persist_dir)
        collection_name = config.index_name

        if persist_dir.exists() and any(persist_dir.iterdir()):
            logger.info(f"从 {persist_dir} 加载已有 Chroma 数据库")
            return db_class(
                persist_directory=str(persist_dir),
                embedding_function=config.embedding,
                collection_name=collection_name
            )

        logger.info(f"在 {persist_dir} 创建新的 Chroma 数据库")
        return db_class.from_texts(
            texts=documents,
            metadatas=metadatas,
            embedding=config.embedding,
            persist_directory=str(persist_dir),
            collection_name=collection_name
        )

    @staticmethod
    def _init_faiss(
            db_class: Type[FAISS],
            config: VectorDBConfig,
            documents: list,
            metadatas: list
    ) -> FAISS:
        """初始化 FAISS 数据库"""
        persist_path = Path(config.persist_dir) / "faiss_index"

        if persist_path.exists():
            logger.info(f"从 {persist_path} 加载已有 FAISS 索引")
            return db_class.load_local(
                folder_path=str(persist_path),
                embeddings=config.embedding
            )

        logger.info(f"在 {persist_path} 创建新的 FAISS 索引")
        db = db_class.from_texts(
            texts=documents,
            metadatas=metadatas,
            embedding=config.embedding
        )
        db.save_local(str(persist_path))
        return db

    @staticmethod
    def _init_pinecone(
            db_class: Type[Pinecone],
            config: VectorDBConfig,
            documents: list,
            metadatas: list
    ) -> Pinecone:
        """初始化 Pinecone 数据库"""
        if not config.api_key:
            raise ValueError("Pinecone 需要有效的 API 密钥")

        import pinecone
        pinecone.init(
            api_key=config.api_key,
            environment=config.host  # Pinecone 使用 host 字段作为环境
        )

        if config.index_name not in pinecone.list_indexes():
            logger.info(f"创建新的 Pinecone 索引: {config.index_name}")
            pinecone.create_index(
                name=config.index_name,
                dimension=1536,  # OpenAI 嵌入维度
                metric="cosine"
            )

        logger.info(f"连接到 Pinecone 索引: {config.index_name}")
        return db_class.from_texts(
            texts=documents,
            metadatas=metadatas,
            embedding=config.embedding,
            index_name=config.index_name
        )


# 依赖注入函数
def get_vector_store() -> VectorStore:
    """获取向量数据库实例（FastAPI 依赖项）"""
    from src.core.embeddings import get_embeddings  # 延迟导入避免循环依赖

    db_config = VectorDBConfig(
        db_type=settings.VECTOR_DB_TYPE,
        persist_dir=settings.VECTOR_DB_PERSIST_DIR,
        index_name=settings.VECTOR_DB_INDEX_NAME,
        api_key=settings.VECTOR_DB_API_KEY,
        host=settings.VECTOR_DB_HOST,
        embedding=get_embeddings()
    )

    return VectorDBFactory.create(db_config)