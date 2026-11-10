"""Section-aware chunking.

Each chunk is one complete section plus its ancestor headings. Retrieval units
therefore never straddle a heading boundary, and every hit carries a path that
can be shown to a user as a citation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import Document, Section


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    path: tuple[str, ...]
    text: str

    @property
    def display_path(self) -> str:
        return " > ".join(self.path)


def _walk(doc: Document, section: Section, parents: tuple[str, ...], out: list[Chunk]) -> None:
    path = parents + (section.heading,)
    if section.text.strip():
        out.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{'/'.join(path)}",
                doc_id=doc.doc_id,
                path=path,
                text=section.text,
            )
        )
    for child in section.children:
        _walk(doc, child, path, out)


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        for section in doc.sections:
            _walk(doc, section, (doc.title,), out)
    return out
