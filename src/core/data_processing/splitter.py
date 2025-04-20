import re
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    """文本块数据模型"""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_id: Optional[str] = None


class SplitterError(Exception):
    """分块处理异常基类"""


class BaseSplitter(ABC):
    """文本分块器抽象基类"""

    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            separators: Optional[List[str]] = None,
            length_function: callable = len
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "？", "！", "；", " ", ""]
        self.length_function = length_function

        self._validate_params()

    def _validate_params(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("重叠长度必须小于分块长度")
        if self.chunk_size <= 0:
            raise ValueError("分块长度必须大于0")

    @abstractmethod
    def split(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """分割文本为块序列"""
        raise NotImplementedError


class RecursiveTextSplitter(BaseSplitter):
    """递归语义分块器"""

    def split(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        try:
            return self._recursive_split(
                text.strip(),
                separators=self.separators,
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"分块失败: {str(e)}")
            raise SplitterError(f"文本分块错误: {str(e)}") from e

    def _recursive_split(
            self,
            text: str,
            separators: List[str],
            metadata: Dict
    ) -> List[Chunk]:
        # 终止条件：找到可用分隔符或遍历完所有分隔符
        if not text:
            return []

        # 尝试当前分隔符集合
        separator = separators[0]
        other_separators = separators[1:]

        if separator:
            splits = self._split_text(text, separator)
        else:
            splits = [text]

        # 合并小段文本
        merged_splits = self._merge_splits(splits, separator)

        # 递归处理剩余分隔符
        if len(merged_splits) > 1:
            return merged_splits
        elif other_separators:
            return self._recursive_split(text, other_separators, metadata)
        else:
            return [self._create_chunk(text, metadata)]

    def _split_text(self, text: str, separator: str) -> List[str]:
        # 使用正则表达式保留分隔符
        if separator:
            regex_pattern = f"({re.escape(separator)})"
            parts = re.split(regex_pattern, text)

            # 重新组合分割结果
            return [
                parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
                for i in range(0, len(parts), 2)
            ]
        else:
            return [text]

    def _merge_splits(self, splits: List[str], separator: str) -> List[Chunk]:
        chunks = []
        current_chunk = []
        current_length = 0

        for s in splits:
            s_length = self.length_function(s)

            if current_length + s_length > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text))

                    # 处理重叠
                    if chunks:
                        last_chunk = chunks[-1].text
                        overlap_start = max(0, len(last_chunk) - self.chunk_overlap)
                        current_chunk = [last_chunk[overlap_start:]]
                        current_length = self.length_function(current_chunk[0])
                current_chunk.append(s)
                current_length += s_length
            else:
                current_chunk.append(s)
                current_length += s_length

        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text))

        return chunks

    def _create_chunk(self, text: str, metadata: Optional[Dict] = None) -> Chunk:
        return Chunk(
            text=text.strip(),
            metadata=metadata or {},
            chunk_id=f"chunk_{hash(text)}"
        )


class SemanticSplitter(BaseSplitter):
    """基于语义的分块器（需配合NLP模型使用）"""

    def __init__(
            self,
            model,
            chunk_size: int = 512,
            chunk_overlap: int = 50
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.model = model  # 句子嵌入模型

    def split(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        # 实现基于语义相似度的分块逻辑
        pass


# 使用示例
if __name__ == "__main__":
    # 初始化分块器
    splitter = RecursiveTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "?", "!", ";", ",", " "]
    )

    # 示例文本
    sample_text = "自然语言处理是人工智能的重要领域...（长文本）..."

    # 执行分块
    chunks = splitter.split(sample_text, {"source": "sample_doc.pdf"})

    # 输出结果
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(chunk.text)
        print("-" * 50)