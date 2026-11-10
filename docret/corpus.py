"""Synthetic document corpus with exactly-gradable labels.

Documents have nested sections. A fact is planted inside one specific subsection,
and the labelled query asks for exactly that fact. Because many documents share
near-identical wording, a retriever has to use the discriminative detail (the
number, the entity) rather than the topic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

COMPANIES = [
    "Acme Holdings", "Northwind Bank", "Bluepeak Logistics", "Cedarline Health",
    "Harbour Freight", "Meridian Retail", "Orchard Energy", "Silverton Media",
    "Tessera Pharma", "Vantage Insurance",
]

TOPICS = [
    ("Data handling", "Retention", "retention period"),
    ("Data handling", "Encryption", "encryption standard"),
    ("Commercial terms", "Liability", "liability cap"),
    ("Commercial terms", "Payment", "payment window"),
    ("Operations", "Escalation", "escalation threshold"),
]

UNITS = {
    "retention period": "days",
    "encryption standard": "bits",
    "liability cap": "million",
    "payment window": "days",
    "escalation threshold": "hours",
}


@dataclass
class Section:
    heading: str
    text: str
    children: list["Section"] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str
    title: str
    sections: list[Section]


def build(seed: int = 7, n_docs: int = 30) -> tuple[list[Document], list[dict]]:
    """Return (documents, labelled queries).

    Each query's label is the tuple (doc_id, heading path) of the one chunk that
    actually holds the answer. That makes every metric exact.
    """
    rng = random.Random(seed)
    docs: list[Document] = []
    labels: list[dict] = []

    for i in range(n_docs):
        company = COMPANIES[i % len(COMPANIES)]
        suffix = f" (agreement {i + 1:02d})"
        sections: list[Section] = []
        planted: dict[str, tuple[str, str]] = {}
        planted_value: dict[str, int] = {}

        for parent, child, fact in TOPICS:
            value = {
                "retention period": rng.choice([30, 45, 90, 180, 365, 730]),
                "encryption standard": rng.choice([128, 192, 256, 512]),
                "liability cap": rng.choice([1, 2, 5, 10, 25]),
                "payment window": rng.choice([15, 30, 45, 60, 90]),
                "escalation threshold": rng.choice([2, 4, 8, 24, 48]),
            }[fact]
            sentence = (
                f"For {company}{suffix}, the {fact} is {value} {UNITS[fact]}, "
                f"and any deviation requires written approval from the governance board."
            )
            child_section = Section(heading=child, text=sentence)
            parent_section = next((s for s in sections if s.heading == parent), None)
            if parent_section is None:
                parent_section = Section(
                    heading=parent,
                    text=f"This section governs {parent.lower()} for {company}{suffix}.",
                )
                sections.append(parent_section)
            parent_section.children.append(child_section)
            planted[fact] = (parent, child)
            planted_value[fact] = value

        docs.append(Document(doc_id=f"doc{i:02d}", title=f"{company} master agreement", sections=sections))

        # one labelled query per planted fact, pointing at its exact section path
        for parent, child, fact in TOPICS:
            value = next(
                s for s in sections
                for c in s.children
                if c.heading == child and s.heading == parent
            )
            labels.append({
                "query_id": f"q{i:02d}-{fact.split()[0]}",
                "query": (f"Which {company} agreement sets the {fact} "
                          f"to {planted_value[fact]} {UNITS[fact]}?"),
                "doc_id": f"doc{i:02d}",
                "path": [f"{company} master agreement", parent, child],
                "gold_text": value.children[0].text if value.children else "",
            })

    # Digest documents restate a handful of facts from other agreements.
    # They are deliberately near-duplicates: a retriever that only matches
    # keywords will surface the digest instead of the authoritative clause,
    # which is exactly the failure mode a ranking metric should expose.
    for d in range(0, n_docs, 3):
        company = COMPANIES[d % len(COMPANIES)]
        notes = []
        for k in range(3):
            src = d + k
            if src >= n_docs:
                break
            c = COMPANIES[src % len(COMPANIES)]
            notes.append(
                f"In a prior review, the {TOPICS[k][2]} for {c} (agreement {src + 1:02d}) "
                f"was recorded as a key obligation."
            )
        docs.append(Document(
            doc_id=f"dig{d:02d}",
            title=f"{company} obligations digest",
            sections=[Section(heading="Summary of obligations",
                              text=" ".join(notes) or "No obligations recorded.")],
        ))

    return docs, labels
