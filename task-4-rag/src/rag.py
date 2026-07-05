"""端到端 RAG:answer(query) -> {"answer": str, "sources": List[dict]}。"""
from .generator import Generator
from .retriever import Retriever

_retriever = None
_generator = None


def _get_pipeline():
    global _retriever, _generator
    if _retriever is None:
        _retriever = Retriever()
    if _generator is None:
        _generator = Generator()
    return _retriever, _generator


def answer(query, k=5):
    retriever, generator = _get_pipeline()
    sources = retriever.retrieve(query, k=k)
    text = generator.generate(query, sources)
    return {"answer": text, "sources": sources}
