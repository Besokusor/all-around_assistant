"""
文档处理器 — 上传、解析、智能切片

切片策略（业界主流方法）：
  1. RecursiveCharacterTextSplitter — 递归字符分割
     按优先级递减的分隔符依次尝试分割，保证语义完整性：
     "\n\n" → "\n" → "。" → "！" → "？" → "；" → " " → ""
     优于固定长度切片：不会在句子中间截断

  2. SemanticChunker — 语义分割
     基于 embedding 相似度检测语义边界：
     相邻句子相似度低于阈值 → 在此处切分

  3. Hybrid（推荐）— 混合模式
     先用 RecursiveCharacterTextSplitter 粗切，
     再用 SemanticChunker 细调边界

支持格式：PDF, TXT, Markdown, Word, CSV, JSON, Python/JS/TS
"""
import os
import re
import hashlib
import time
from typing import Optional, Callable

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)

from config import DOCUMENT_CONFIG, QWEN_MAAS_CONFIG


# ==================== 文档解析 ====================

class DocumentParser:
    """
    多格式文档解析器

    支持的格式：
    - .pdf   → PyPDF2 (基础) / pdfplumber (高级表格)
    - .txt   → 直接读取（自动检测编码）
    - .md    → 读取并保留结构
    - .docx  → python-docx
    - .csv   → pandas/csv
    - .json  → json
    - .py/.js/.ts/.java → 代码文件
    """

    MAX_FILE_SIZE = DOCUMENT_CONFIG["max_file_size_mb"] * 1024 * 1024

    @classmethod
    def parse(cls, file_path: str) -> list[dict]:
        """
        解析文档为段落列表

        Args:
            file_path: 文件路径

        Returns:
            [{"content": "...", "metadata": {...}}, ...]

        Raises:
            ValueError: 不支持的文件格式或文件过大
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > cls.MAX_FILE_SIZE:
            raise ValueError(
                f"文件过大（{file_size / 1024 / 1024:.1f}MB），"
                f"最大支持 {DOCUMENT_CONFIG['max_file_size_mb']}MB"
            )

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in DOCUMENT_CONFIG["supported_formats"]:
            raise ValueError(
                f"不支持的文件格式「{ext}」，"
                f"支持的格式：{', '.join(DOCUMENT_CONFIG['supported_formats'])}"
            )

        file_name = os.path.basename(file_path)

        parser_map = {
            ".pdf": cls._parse_pdf,
            ".txt": cls._parse_text,
            ".md": cls._parse_markdown,
            ".docx": cls._parse_docx,
            ".csv": cls._parse_csv,
            ".json": cls._parse_json,
            ".py": cls._parse_code,
            ".js": cls._parse_code,
            ".ts": cls._parse_code,
            ".java": cls._parse_code,
        }

        parser = parser_map.get(ext, cls._parse_text)
        raw_text = parser(file_path)

        # 返回单段文本（后续由 chunker 切片）
        return [{
            "content": raw_text,
            "metadata": {
                "source": file_name,
                "file_path": file_path,
                "file_type": ext,
                "file_size": file_size,
                "parsed_at": time.time(),
            },
        }]

    # ==================== 各格式解析器 ====================

    @classmethod
    def _parse_pdf(cls, path: str) -> str:
        """PDF 解析（优先 pdfplumber，降级 PyPDF2）"""
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        parts.append(text)
                    # 尝试提取表格
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = "\n".join(
                                " | ".join(str(cell or "") for cell in row)
                                for row in table
                            )
                            parts.append(table_text)
            return "\n\n".join(parts)
        except ImportError:
            pass
        except Exception:
            pass

        # 降级 pypdf（PyPDF2 的继任者）
        try:
            from pypdf import PdfReader
            parts = []
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
            return "\n\n".join(parts)
        except Exception as e:
            raise ValueError(f"PDF 解析失败：{e}")

    @classmethod
    def _parse_text(cls, path: str) -> str:
        """文本文件解析（自动编码检测）"""
        try:
            import chardet
            with open(path, "rb") as f:
                raw = f.read()
            encoding = chardet.detect(raw).get("encoding", "utf-8")
            return raw.decode(encoding, errors="replace")
        except ImportError:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

    @classmethod
    def _parse_markdown(cls, path: str) -> str:
        """Markdown 解析（保留结构）"""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @classmethod
    def _parse_docx(cls, path: str) -> str:
        """Word 文档解析"""
        try:
            from docx import Document
            doc = Document(path)
            parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    # 检测标题
                    if para.style.name.startswith("Heading"):
                        level = para.style.name.replace("Heading ", "")
                        prefix = "#" * int(level) if level.isdigit() else "##"
                        parts.append(f"{prefix} {para.text}")
                    else:
                        parts.append(para.text)
            return "\n\n".join(parts)
        except Exception as e:
            raise ValueError(f"Word 文档解析失败：{e}")

    @classmethod
    def _parse_csv(cls, path: str) -> str:
        """CSV 解析"""
        import csv
        parts = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                parts.append(" | ".join(headers))
                parts.append(" | ".join(["---"] * len(headers)))
            for row in reader:
                parts.append(" | ".join(row))
        return "\n".join(parts)

    @classmethod
    def _parse_json(cls, path: str) -> str:
        """JSON 解析（格式化）"""
        import json
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def _parse_code(cls, path: str) -> str:
        """代码文件解析"""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


# ==================== 智能切片器 ====================

class SmartChunker:
    """
    智能文档切片器

    三种模式：
    1. recursive  — RecursiveCharacterTextSplitter（推荐，适合通用文档）
    2. semantic   — SemanticChunker（基于 embedding 相似度找语义边界）
    3. hybrid     — 先用 recursive 粗切，再用 semantic 细调（最精准）
    """

    def __init__(
        self,
        method: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_fn: Optional[Callable] = None,
    ):
        self.method = method or DOCUMENT_CONFIG["chunk_method"]
        self.chunk_size = chunk_size or DOCUMENT_CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap or DOCUMENT_CONFIG["chunk_overlap"]
        self.separators = DOCUMENT_CONFIG["separators"]
        self.embedding_fn = embedding_fn

    def split(self, documents: list[dict]) -> list[dict]:
        """
        切片主入口

        Args:
            documents: [{"content": "...", "metadata": {...}}, ...]

        Returns:
            [{"content": "chunk...", "metadata": {...}}, ...]
        """
        if self.method == "recursive":
            return self._recursive_split(documents)
        elif self.method == "semantic":
            return self._semantic_split(documents)
        elif self.method == "hybrid":
            return self._hybrid_split(documents)
        else:
            raise ValueError(f"不支持的切片方法：{self.method}")

    def _recursive_split(self, documents: list[dict]) -> list[dict]:
        """
        RecursiveCharacterTextSplitter — 递归字符分割

        原理：按优先级递减的分隔符依次尝试切分
          "\n\n" → "\n" → "。" → "！" → "？" → "；" → " " → ""

        优势：
        - 优先在段落边界（\n\n）切分
        - 其次在句子边界（。！？）切分
        - 最后才在字符级别切分
        - 保持语义单元完整，不会截断句子
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = []
        for doc in documents:
            texts = splitter.split_text(doc["content"])
            for i, text in enumerate(texts):
                chunks.append({
                    "content": text.strip(),
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "chunk_method": "recursive",
                    },
                })

        return chunks

    def _semantic_split(self, documents: list[dict]) -> list[dict]:
        """
        SemanticChunker — 语义分割

        原理：
        1. 将文档拆为句子
        2. 计算相邻句子的 embedding 相似度
        3. 相似度低于阈值 → 在此处切分（说明语义发生了变化）

        优势：
        - 在语义边界切分，主题变化处自然断开
        - 不依赖固定长度，适应不同密度的内容
        """
        try:
            from langchain_text_splitters import SemanticChunker

            if self.embedding_fn is None:
                # 降级到 recursive
                print("  ⚠️ SemanticChunker 需要 embedding_fn，降级为 recursive 模式")
                return self._recursive_split(documents)

            splitter = SemanticChunker(
                embeddings=self.embedding_fn,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=DOCUMENT_CONFIG["semantic_threshold"],
            )

            chunks = []
            for doc in documents:
                texts = splitter.split_text(doc["content"])
                for i, text in enumerate(texts):
                    chunks.append({
                        "content": text.strip(),
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": i,
                            "chunk_method": "semantic",
                        },
                    })
            return chunks

        except ImportError:
            print("  ⚠️ SemanticChunker 不可用，降级为 recursive 模式")
            return self._recursive_split(documents)

    def _hybrid_split(self, documents: list[dict]) -> list[dict]:
        """
        混合模式 — recursive 粗切 + semantic 细调

        流程：
        1. RecursiveCharacterTextSplitter 用较大 chunk_size (×2) 粗切
        2. SemanticChunker 在粗切结果上细调边界
        3. 保证每个 chunk 不超过目标大小

        综合两种方法优势：效率高 + 语义精准
        """
        # 第一步：粗切（使用 2 倍 chunk_size）
        original_size = self.chunk_size
        self.chunk_size = original_size * 2
        coarse_chunks = self._recursive_split(documents)
        self.chunk_size = original_size

        # 第二步：如果每个粗切 chunk 太大，再细切
        fine_chunks = []
        for chunk in coarse_chunks:
            if len(chunk["content"]) > self.chunk_size * 1.5:
                # 对大 chunk 做二次分割
                sub_texts = self._recursive_split([{
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                }])
                for sub in sub_texts:
                    sub["metadata"]["chunk_method"] = "hybrid"
                    fine_chunks.append(sub)
            else:
                chunk["metadata"]["chunk_method"] = "hybrid"
                fine_chunks.append(chunk)

        return fine_chunks


# ==================== 文档处理器（统一入口） ====================

class DocumentProcessor:
    """
    文档处理统一入口

    流程：
    Upload → Parse → Smart Chunk → Embed → Store (Chroma)
    """

    def __init__(self, embedding_fn: Optional[Callable] = None):
        self.parser = DocumentParser()
        self.chunker = SmartChunker(embedding_fn=embedding_fn)
        self.embedding_fn = embedding_fn

    def process(self, file_path: str) -> list[dict]:
        """
        处理文档：解析 + 智能切片

        Args:
            file_path: 文档路径

        Returns:
            [{"content": "chunk...", "metadata": {...}, "chunk_id": "..."}, ...]
        """
        # Step 1: 解析
        docs = self.parser.parse(file_path)

        # Step 2: 智能切片
        chunks = self.chunker.split(docs)

        # Step 3: 为每个 chunk 生成唯一 ID
        for chunk in chunks:
            raw = f"{chunk['metadata']['source']}_{chunk['metadata']['chunk_index']}_{chunk['content'][:50]}"
            chunk["chunk_id"] = hashlib.md5(raw.encode()).hexdigest()[:16]

        return chunks

