"""Fast local embeddings exposed through LangChain's Embeddings interface."""

from __future__ import annotations

from threading import Lock
from typing import Iterable

from langchain_core.embeddings import Embeddings

from .config import EMBEDDING_MODEL


class FastBGEEmbeddings(Embeddings):
    """BAAI BGE-small embeddings using FastEmbed/ONNX on CPU.

    The model is downloaded once by FastEmbed and then cached locally. A single
    model instance is shared per Python process so query-time embedding stays
    fast and does not repeatedly reload model weights.
    """

    _model = None
    _model_name = None
    _lock = Lock()

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._ensure_model()

    def _ensure_model(self):
        if self.__class__._model is not None and self.__class__._model_name == self.model_name:
            return self.__class__._model
        with self.__class__._lock:
            if self.__class__._model is None or self.__class__._model_name != self.model_name:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise RuntimeError(
                        "fastembed is required for local RAG embeddings. Run: pip install fastembed"
                    ) from exc
                self.__class__._model = TextEmbedding(model_name=self.model_name)
                self.__class__._model_name = self.model_name
        return self.__class__._model

    @property
    def model(self):
        return self._ensure_model()

    @staticmethod
    def _as_lists(vectors: Iterable) -> list[list[float]]:
        return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Retrieval models can use a passage-specific representation. FastEmbed
        # exposes this directly for BGE and falls back cleanly when unavailable.
        if hasattr(self.model, "passage_embed"):
            vectors = self.model.passage_embed(texts)
        else:
            vectors = self.model.embed(texts)
        return self._as_lists(vectors)

    def embed_query(self, text: str) -> list[float]:
        # Query-specific embedding improves retrieval for BGE-style models.
        if hasattr(self.model, "query_embed"):
            vectors = self._as_lists(self.model.query_embed(text))
        else:
            vectors = self._as_lists(self.model.embed([text]))
        if not vectors:
            raise RuntimeError("Embedding model returned no query vector.")
        return vectors[0]
