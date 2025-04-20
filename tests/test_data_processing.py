import unittest
import tempfile
import logging
from pathlib import Path

# 被测试模块
from src.core.data_processing.loader import (
    DocumentLoaderFactory,
    PDFLoader,
    WordLoader,
    DocumentLoadError
)
from src.core.data_processing.splitter import (
    RecursiveTextSplitter,
    Chunk,
    SplitterError
)

# 初始化日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestDocumentLoaders(unittest.TestCase):
    """文档加载器测试套件"""

    @classmethod
    def setUpClass(cls):
        # 创建测试用临时文件
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_data = {
            "valid.pdf": b"%PDF-1.4 minimal file...",  # 简化的PDF结构
            "valid.docx": b"PK\x03\x04...",  # 最小化的Word文件
            "empty.txt": b"",
            "invalid.pdf": b"NOT_A_PDF",
            "large.txt": b"Hello World\n" * 1000
        }

        # 生成测试文件
        for filename, content in cls.test_data.items():
            path = Path(cls.temp_dir.name) / filename
            with open(path, "wb") as f:
                f.write(content)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_pdf_loader_success(self):
        """测试PDF文件正常加载"""
        loader = PDFLoader(Path(self.temp_dir.name) / "valid.pdf")
        pages = loader.load()
        self.assertGreater(len(pages), 0)
        self.assertIsInstance(pages, list)

    def test_word_loader_success(self):
        """测试Word文件正常加载"""
        loader = WordLoader(Path(self.temp_dir.name) / "valid.docx")
        paragraphs = loader.load()
        self.assertTrue(all(isinstance(p, str) for p in paragraphs))

    def test_invalid_file_type(self):
        """测试不支持的文件类型"""
        with self.assertRaises(ValueError):
            DocumentLoaderFactory.get_loader("image.jpg")

    def test_corrupted_pdf_handling(self):
        """测试损坏的PDF文件处理"""
        path = Path(self.temp_dir.name) / "invalid.pdf"
        with self.assertRaises(DocumentLoadError):
            PDFLoader(path).load()

    def test_large_text_file(self):
        """测试大文本文件加载性能"""
        path = Path(self.temp_dir.name) / "large.txt"
        loader = DocumentLoaderFactory.get_loader(path)
        content = loader.load()
        self.assertEqual(len(content), 1)  # 单块加载
        self.assertGreater(len(content[0]), 10000)


class TestTextSplitters(unittest.TestCase):
    """文本分块器测试套件"""

    def setUp(self):
        self.splitter = RecursiveTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n", "。", " "]
        )

    def test_basic_splitting(self):
        """测试基础分块逻辑"""
        text = "a b c d " * 100  # 400字符
        chunks = self.splitter.split(text)

        # 验证分块数量
        expected_chunks = len(text) // (500 - 50) + 1
        self.assertGreaterEqual(len(chunks), expected_chunks)

        # 验证块长度限制
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 500)

        # 验证重叠处理
        for i in range(1, len(chunks)):
            overlap = set(chunks[i - 1].text[-50:].split()) & set(chunks[i].text[:50].split())
            self.assertGreater(len(overlap), 0)

    def test_edge_cases(self):
        """测试边界条件"""
        cases = [
            ("", []),  # 空文本
            ("a" * 300, [1]),  # 小于块大小
            ("a" * 600, [2]),  # 刚好需要分块
            ("a\n" * 1000, [10, 20])  # 多分隔符
        ]

        for text, expected_range in cases:
            with self.subTest(text=text):
                chunks = self.splitter.split(text)
                self.assertTrue(
                    expected_range[0] <= len(chunks) <= expected_range[-1],
                    f"分块数量异常: {len(chunks)}"
                )

    def test_metadata_propagation(self):
        """测试元数据传递"""
        metadata = {"source": "test.md", "page": 42}
        chunks = self.splitter.split("text", metadata)

        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "test.md")
            self.assertEqual(chunk.metadata["page"], 42)

    def test_invalid_parameters(self):
        """测试非法参数校验"""
        with self.assertRaises(ValueError):
            RecursiveTextSplitter(chunk_size=0)

        with self.assertRaises(ValueError):
            RecursiveTextSplitter(chunk_overlap=500, chunk_size=400)


class TestIntegration(unittest.TestCase):
    """集成测试：完整数据处理流程"""

    def test_end_to_end_processing(self):
        """从文件加载到分块的全流程测试"""
        # 1. 加载文档
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            f.write(b"Line 1\nLine 2\n" * 100)
            f.seek(0)

            loader = DocumentLoaderFactory.get_loader(f.name)
            text = loader.load()[0]

        # 2. 分块处理
        splitter = RecursiveTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        chunks = splitter.split(text)

        # 验证结果完整性
        reconstructed = " ".join(chunk.text for chunk in chunks)
        self.assertEqual(reconstructed.replace(" ", ""), text.replace(" ", ""))


if __name__ == "__main__":
    unittest.main(
        verbosity=2,
        testRunner=unittest.TextTestRunner(
            descriptions=True,
            resultclass=unittest.TestResult
        )
    )