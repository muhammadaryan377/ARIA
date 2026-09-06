"""PDF RAG capability used by ARIA's Insight Agent.

This is intentionally a capability/module, not a fifth autonomous agent.
"""

from .service import InsightPDFRAG

__all__ = ["InsightPDFRAG"]
