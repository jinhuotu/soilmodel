# src/core/data_processing/loader.py
import os
import logging
from pathlib import Path
from typing import List, Union, Optional
from abc import ABC, abstractmethod
import magic  # 需要python-magic库

# 第三方依赖
try:
    from PyPDF2 import PdfReader
    from docx import Document
    import markdown
    from bs4 import BeautifulSoup
except ImportError as e:
    raise RuntimeError("请先安装文档解析依赖：pip install PyPDF2 python-docx markdown beautifulsoup4") from e

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """文档加载异常基类"""

    def __init__(self, file_path: str, message: str):
        super().__init__(f"无法加载文件 {file_path}: {message}")
        self.file_path = file_path


class BaseDocumentLoader(ABC):
    """文档加载器抽象基类"""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

    @abstractmethod
    def load(self) -> List[str]:
        """加载文档内容为文本块列表"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> List[str]:
        """支持的文件扩展名"""
        raise NotImplementedError

    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """检查是否支持该文件类型"""
        return Path(file_path).suffix.lower() in cls.supported_extensions()


class PDFLoader(BaseDocumentLoader):
    """PDF文档加载器"""

    def load(self) -> List[str]:
        try:
            with open(self.file_path, 'rb') as f:
                reader = PdfReader(f)
                return [page.extract_text() for page in reader.pages]
        except Exception as e:
            raise DocumentLoadError(str(self.file_path), f"PDF解析失败: {str(e)}")

    @classmethod
    def supported_extensions(cls) -> List[str]:
        return ['.pdf']


class WordLoader(BaseDocumentLoader):
    """Word文档加载器"""

    def load(self) -> List[str]:
        try:
            doc = Document(self.file_path)
            return [para.text for para in doc.paragraphs if para.text.strip()]
        except Exception as e:
            raise DocumentLoadError(str(self.file_path), f"Word解析失败: {str(e)}")

    @classmethod
    def supported_extensions(cls) -> List[str]:
        return ['.docx']


class TextLoader(BaseDocumentLoader):
    """纯文本加载器"""

    def load(self) -> List[str]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return [f.read()]
        except UnicodeDecodeError:
            try:
                with open(self.file_path, 'r', encoding='gbk') as f:
                    return [f.read()]
            except Exception as e:
                raise DocumentLoadError(str(self.file_path), f"文本解码失败: {str(e)}")

    @classmethod
    def supported_extensions(cls) -> List[str]:
        return ['.txt', '.md']


class MarkdownLoader(TextLoader):
    """Markdown文档加载器"""

    def load(self) -> List[str]:
        raw_text = super().load()[0]
        html = markdown.markdown(raw_text)
        soup = BeautifulSoup(html, 'html.parser')
        return [soup.get_text()]

    @classmethod
    def supported_extensions(cls) -> List[str]:
        return ['.md']


class DocumentLoaderFactory:
    """文档加载器工厂"""

    _loaders = {
        'pdf': PDFLoader,
        'word': WordLoader,
        'text': TextLoader,
        'markdown': MarkdownLoader
    }

    @classmethod
    def get_loader(
            cls,
            file_path: Union[str, Path],
            mime_type: Optional[str] = None
    ) -> BaseDocumentLoader:
        """根据文件类型获取加载器"""
        path = Path(file_path)

        # 优先使用MIME类型检测
        if mime_type:
            for loader in cls._loaders.values():
                if mime_type in loader.supported_mime_types():
                    return loader(file_path)

        # 根据扩展名匹配
        for loader in cls._loaders.values():
            if loader.is_supported(path):
                return loader(path)

        # 使用magic库进行文件类型检测
        mime = magic.Magic(mime=True)
        detected_type = mime.from_file(str(path))
        for loader in cls._loaders.values():
            if detected_type in loader.supported_mime_types():
                return loader(path)

        raise ValueError(f"不支持的文件类型: {path.suffix} (MIME: {detected_type})")


# 扩展MIME类型支持
PDFLoader.supported_mime_types = lambda: ['application/pdf']
WordLoader.supported_mime_types = lambda: [
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
]
TextLoader.supported_mime_types = lambda: ['text/plain']
MarkdownLoader.supported_mime_types = lambda: ['text/markdown']


def load_documents(file_path: Union[str, Path]) -> List[str]:
    """统一文档加载入口"""
    try:
        loader = DocumentLoaderFactory.get_loader(file_path)
        return loader.load()
    except Exception as e:
        logger.error("文档加载失败: %s", e)
        raise


# 使用示例
if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(level=logging.INFO)

    # 加载PDF示例
    pdf_content = load_documents("sample.pdf")
    print(f"Loaded {len(pdf_content)} pages from PDF")

    # 加载Word示例
    docx_content = load_documents("sample.docx")
    print(f"Loaded {len(docx_content)} paragraphs from Word")