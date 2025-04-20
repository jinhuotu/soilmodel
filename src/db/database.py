# src/db/database.py
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from src.config.settings import settings
from src.api.utils.logger import logger

# 配置日志
logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库连接管理器"""

    def __init__(self):
        self.engine = None
        self.session_factory = None

    def init_db(self):
        """初始化数据库连接"""
        try:
            self.engine = create_engine(
                settings.DATABASE_URL,
                pool_size=20,  # 连接池大小
                max_overflow=30,  # 最大溢出连接数
                pool_pre_ping=True,  # 连接前健康检查
                pool_recycle=3600,  # 连接回收时间（秒）
                connect_args={
                    "connect_timeout": 10  # 连接超时时间
                }
            )
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            logger.info("数据库连接池初始化成功")
        except Exception as e:
            logger.critical(f"数据库连接失败: {str(e)}")
            raise

    @contextmanager
    def get_session(self):
        """获取数据库会话上下文管理器"""
        session = scoped_session(self.session_factory)()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"数据库操作异常: {str(e)}")
            raise
        finally:
            session.close()


# 初始化全局数据库管理器
db_manager = DatabaseManager()
db_manager.init_db()


def get_db():
    """FastAPI 依赖项：获取数据库会话"""
    with db_manager.get_session() as session:
        yield session


def test_connection():
    """数据库连接测试函数"""
    try:
        with db_manager.engine.connect() as conn:
            conn.execute("SELECT 1")
            logger.info("数据库连接测试成功")
            return True
    except Exception as e:
        logger.error(f"数据库连接测试失败: {str(e)}")
        return False


# 服务启动时自动测试连接
if __name__ == "__main__":
    test_connection()