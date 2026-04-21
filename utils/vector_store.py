"""F-16-1: Chroma Vector DB 인덱스 구축 유틸.

컬렉션:
  prompt_clinic_diagnosis  — Chain 2 진단 보조 (행위축 필터)
  prompt_clinic_rewrite    — Chain 3 재작성 보조 (학문축 필터)

RAG_ENABLED=false(기본) 시 build_index/search_* 모두 no-op.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GUIDES_DIR = DATA_DIR / "guides"
WIKI_DIR = DATA_DIR / "wiki"
PROMPT_RUNS_PATH = DATA_DIR / "prompt_runs.jsonl"
FEWSHOT_PATH = DATA_DIR / "fewshot_examples.json"
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

COLLECTION_DIAGNOSIS = "prompt_clinic_diagnosis"
COLLECTION_REWRITE = "prompt_clinic_rewrite"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# data/wiki/ 하위 폴더명 → 학문축 레이블 매핑
_WIKI_FOLDER_TO_DOMAIN: dict[str, str] = {
    "medical": "의학",
    "law": "법률",
    "design": "디자인",
    "coding": "코딩",
    "science": "과학",
    "marketing": "마케팅",
    "general": "일반",
}


def _rag_enabled() -> bool:
    return os.environ.get("RAG_ENABLED", "false").lower() == "true"


def _get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_chroma_client():
    import chromadb
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def _chunk_text(text: str) -> list[str]:
    """RecursiveCharacterTextSplitter 방식으로 청킹."""
    try:
        from langchain_community.document_loaders import TextLoader  # noqa: F401
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        return [doc.page_content for doc in splitter.create_documents([text])]
    except Exception:
        # fallback: 단순 슬라이싱
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks


def _pdf_to_markdown(pdf_path: Path) -> str:
    """markitdown으로 PDF → Markdown 변환."""
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(str(pdf_path))
        return result.text_content or ""
    except Exception:
        return ""


def _load_docs_from_runs() -> list[tuple[str, dict[str, str]]]:
    """prompt_runs.jsonl → (text, metadata) 리스트."""
    if not PROMPT_RUNS_PATH.exists():
        return []
    docs: list[tuple[str, dict[str, str]]] = []
    with PROMPT_RUNS_PATH.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            user_prompt = str(record.get("user_prompt") or "").strip()
            improved = str(record.get("improved_prompt") or "").strip()
            summary = str(record.get("analysis_summary") or "").strip()
            text = "\n".join(part for part in [user_prompt, improved, summary] if part)
            if not text:
                continue
            meta: dict[str, str] = {
                "source": "internal",
                "type": "fewshot",
                "domain_action": str(record.get("domain_action") or ""),
                "domain_knowledge": str(record.get("domain_knowledge") or ""),
            }
            docs.append((text, meta))
    return docs


def _load_docs_from_fewshot() -> list[tuple[str, dict[str, str]]]:
    """fewshot_examples.json → (text, metadata) 리스트."""
    if not FEWSHOT_PATH.exists():
        return []
    try:
        examples = json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(examples, list):
        return []
    docs: list[tuple[str, dict[str, str]]] = []
    for item in examples:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        analysis = str(item.get("analysis") or "").strip()
        text = "\n".join(part for part in [prompt, analysis] if part)
        if not text:
            continue
        meta: dict[str, str] = {
            "source": "internal",
            "type": "fewshot",
            "domain_action": str(item.get("domain_action") or ""),
            "domain_knowledge": "",
        }
        docs.append((text, meta))
    return docs


def _load_docs_from_guides() -> list[tuple[str, dict[str, str]]]:
    """data/guides/ PDF/MD/TXT → 청킹 후 (text, metadata). domain_knowledge: '일반'."""
    if not GUIDES_DIR.exists():
        return []
    allowed_ext = {".pdf", ".md", ".txt"}
    docs: list[tuple[str, dict[str, str]]] = []
    for file_path in sorted(GUIDES_DIR.iterdir()):
        if file_path.name.startswith(".") or file_path.suffix not in allowed_ext:
            continue
        if file_path.suffix == ".pdf":
            text = _pdf_to_markdown(file_path)
        else:
            try:
                text = file_path.read_text(encoding="utf-8")
            except OSError:
                continue
        if not text.strip():
            continue
        for chunk in _chunk_text(text):
            if not chunk.strip():
                continue
            meta: dict[str, str] = {
                "source": "external",
                "type": "guide",
                "domain_action": "",
                "domain_knowledge": "일반",
            }
            docs.append((chunk, meta))
    return docs


def _strip_yaml_frontmatter(text: str) -> str:
    """문서 맨 위의 YAML frontmatter(`---` ... `---`) 블록 제거."""
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return text
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :]).lstrip("\n")
    return text


def _load_docs_from_wiki() -> list[tuple[str, dict[str, str]]]:
    """data/wiki/{domain}/ 파일들 → 청킹 후 (text, metadata) 리스트."""
    if not WIKI_DIR.exists():
        return []
    docs: list[tuple[str, dict[str, str]]] = []
    for folder, domain_label in _WIKI_FOLDER_TO_DOMAIN.items():
        folder_path = WIKI_DIR / folder
        if not folder_path.exists():
            continue
        allowed_ext = {".pdf", ".txt", ".md"}
        for file_path in sorted(folder_path.iterdir()):
            if file_path.name.startswith(".") or file_path.suffix not in allowed_ext:
                continue
            if file_path.suffix == ".pdf":
                text = _pdf_to_markdown(file_path)
            else:
                try:
                    text = file_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                if file_path.suffix == ".md":
                    text = _strip_yaml_frontmatter(text)
            if not text.strip():
                continue
            for chunk in _chunk_text(text):
                if not chunk.strip():
                    continue
                meta: dict[str, str] = {
                    "source": "external",
                    "type": "wiki",
                    "domain_action": "",
                    "domain_knowledge": domain_label,
                }
                docs.append((chunk, meta))
    return docs


def _upsert_collection(
    client: Any,
    collection_name: str,
    docs: list[tuple[str, dict[str, str]]],
    embeddings_fn: Any,
) -> int:
    """컬렉션 재구축 (기존 컬렉션 삭제 후 재생성). 저장된 문서 수 반환."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    if not docs:
        return 0

    texts = [text for text, _ in docs]
    metadatas = [meta for _, meta in docs]
    ids = [f"{collection_name}_{i}" for i in range(len(texts))]

    # 임베딩 배치 처리 (최대 100개씩)
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_meta = metadatas[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        vectors = embeddings_fn.embed_documents(batch_texts)
        collection.add(
            embeddings=vectors,
            documents=batch_texts,
            metadatas=batch_meta,
            ids=batch_ids,
        )
    return len(texts)


def build_index() -> dict[str, int]:
    """내부+외부 데이터를 청킹·임베딩해 Chroma에 저장한다.

    반환: {"diagnosis": n_docs, "rewrite": m_docs}
    RAG_ENABLED=false 시 {"diagnosis": 0, "rewrite": 0} 반환.
    """
    if not _rag_enabled():
        return {"diagnosis": 0, "rewrite": 0}

    client = _get_chroma_client()
    embeddings = _get_embeddings()

    # 진단 컬렉션: runs + fewshot
    diag_docs = _load_docs_from_runs() + _load_docs_from_fewshot()
    n_diag = _upsert_collection(client, COLLECTION_DIAGNOSIS, diag_docs, embeddings)

    # 재작성 컬렉션: guides + wiki + fewshot
    rewrite_docs = (
        _load_docs_from_guides() + _load_docs_from_wiki() + _load_docs_from_fewshot()
    )
    n_rewrite = _upsert_collection(client, COLLECTION_REWRITE, rewrite_docs, embeddings)

    return {"diagnosis": n_diag, "rewrite": n_rewrite}


def _query_collection(
    client: Any,
    collection_name: str,
    query: str,
    where: dict[str, str] | None,
    k: int,
    embeddings_fn: Any,
) -> list[dict[str, Any]]:
    """컬렉션 검색 후 결과 리스트 반환. 컬렉션 없거나 오류 시 빈 리스트."""
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []
    try:
        query_vec = embeddings_fn.embed_query(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results": min(k, collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            # where 조건에 해당하는 문서가 없을 수 있으므로 안전하게 처리
            kwargs["where"] = where
        results = collection.query(**kwargs)
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]
        output: list[dict[str, Any]] = []
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            output.append({"text": doc, "metadata": meta, "distance": dist})
        return output
    except Exception:
        return []


def search_diagnosis(
    query: str,
    domain_action: str = "",
    k: int = 3,
) -> list[dict[str, Any]]:
    """행위축 필터로 진단 인덱스 검색.

    RAG_ENABLED=false 시 빈 리스트 반환.
    """
    if not _rag_enabled():
        return []
    client = _get_chroma_client()
    embeddings = _get_embeddings()
    where = {"domain_action": domain_action} if domain_action else None
    return _query_collection(client, COLLECTION_DIAGNOSIS, query, where, k, embeddings)


def search_rewrite(
    query: str,
    domain_knowledge: str = "",
    k: int = 3,
) -> list[dict[str, Any]]:
    """학문축 필터로 재작성 인덱스 검색.

    RAG_ENABLED=false 시 빈 리스트 반환.
    """
    if not _rag_enabled():
        return []
    client = _get_chroma_client()
    embeddings = _get_embeddings()
    where = {"domain_knowledge": domain_knowledge} if domain_knowledge else None
    return _query_collection(client, COLLECTION_REWRITE, query, where, k, embeddings)
