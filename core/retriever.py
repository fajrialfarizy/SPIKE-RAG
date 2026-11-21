"""
Retrieval for the two semantic modes and the catalog mode (Brief 5.5, PRD 8.1).

Guarantees:
- The query is embedded through the SAME e5 code path as ingestion, with exactly
  one "query: " prefix (the embedder adds it; this module never prefixes).
- Metadata filters are always applied: "$and" for several conditions, "$in" per
  condition, no where clause only when there is genuinely no filter.
- Similarity is 1 - cosine distance; candidates below the threshold are dropped.
- Greedy selection applies a per-report cap so one report cannot dominate.
- The catalog mode never embeds and dedupes by report_id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

import config
from core.knowledge_base import ChunkMatch, KnowledgeBase, StoredChunk
from enums import QueryMode
from ingestion.embedder import E5Embedder

FILTER_FIELDS = ("competition_type", "competition_year", "achievement_level")
CATALOG_METADATA_FIELDS = (
    "report_id",
    "competition_name",
    "competition_type",
    "competition_year",
    "achievement_level",
)


class RetrievalError(RuntimeError):
    """Raised when retrieval is configured with an unsupported mode."""


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk chosen for the context, with its similarity when it has one."""

    chunk_id: str
    report_id: str
    document: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    # Set only by retrieve_grouped, so the context can label the group.
    group_label: Optional[str] = None


def merge_filter_dicts(
    base: Optional[Dict[str, Sequence[Any]]],
    extra: Optional[Dict[str, Sequence[Any]]],
) -> Dict[str, List[Any]]:
    """Combine two filter sets; a field in both keeps only the intersection."""
    merged: Dict[str, List[Any]] = {
        field: list(values) for field, values in (base or {}).items() if values
    }
    for field, values in (extra or {}).items():
        incoming = list(values or [])
        if field in merged:
            merged[field] = [value for value in merged[field] if value in incoming]
        else:
            merged[field] = incoming
    return merged


def build_where_filter(
    filters: Optional[Dict[str, Sequence[Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Translate validated analyzer filters into a Chroma where clause.

    Empty filter sets are dropped as a whole; a single remaining condition is
    returned without "$and" because Chroma rejects a one-element $and.
    """
    conditions: List[Dict[str, Any]] = []

    for field in FILTER_FIELDS:
        values = list((filters or {}).get(field) or [])
        if not values:
            continue

        if field == "competition_year":
            normalized = sorted({int(value) for value in values})
        else:
            normalized = sorted({str(value) for value in values})

        if normalized:
            conditions.append({field: {"$in": normalized}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class Retriever:
    """Selects chunks (specific / cross_report) or report metadata (catalog)."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedder: Optional[E5Embedder] = None,
        similarity_threshold: float = config.RETRIEVAL_SIMILARITY_THRESHOLD,
    ):
        self.knowledge_base = knowledge_base
        self.embedder = embedder or E5Embedder()
        self.similarity_threshold = similarity_threshold
        # Latency of the last query embedding, for the debug object and smoke run.
        self.last_embedding_latency_ms = 0.0

    def _embed_query(self, search_query: str) -> Sequence[float]:
        started = time.perf_counter()
        embedding = self.embedder.embed_query(search_query)
        self.last_embedding_latency_ms = round(
            (time.perf_counter() - started) * 1000.0, 1
        )
        return embedding

    # -- semantic modes ----------------------------------------------------
    def _select_with_cap(
        self,
        candidates: Sequence[RetrievedChunk],
        k: int,
        max_per_report: int,
    ) -> List[RetrievedChunk]:
        """
        Greedy pick by score, capped per report, deterministic on ties.

        The sort key is (score, report_id, chunk_id) so an identical query always
        produces the same ordered set even when Chroma returns equal-distance hits
        in a different order (fix pack K1c).
        """
        ordered = sorted(
            candidates,
            key=lambda item: (-(item.score or 0.0), item.report_id, item.chunk_id),
        )

        selected: List[RetrievedChunk] = []
        per_report: Dict[str, int] = {}
        for candidate in ordered:
            if len(selected) >= k:
                break
            used = per_report.get(candidate.report_id, 0)
            if used >= max_per_report:
                continue
            per_report[candidate.report_id] = used + 1
            selected.append(candidate)

        return selected

    def retrieve(
        self,
        search_query: str,
        mode: QueryMode,
        filters: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve context chunks for specific or cross_report mode."""
        if mode not in (QueryMode.SPECIFIC, QueryMode.CROSS_REPORT):
            raise RetrievalError(
                f"Mode '{mode}' has no semantic retrieval; use list_catalog()."
            )

        limits = config.RETRIEVAL_LIMITS[mode.value]
        k = int(limits["k"])
        max_per_report = int(limits["max_per_report"])

        total = self.knowledge_base.count_chunks()
        if total == 0:
            return []

        query_embedding = self._embed_query(search_query)
        return self._query_chunks(
            query_embedding, build_where_filter(filters), k, max_per_report, total
        )

    def _query_chunks(
        self,
        query_embedding: Sequence[float],
        where: Optional[Dict[str, Any]],
        k: int,
        max_per_report: int,
        total: int,
    ) -> List[RetrievedChunk]:
        if k <= 0 or total == 0:
            return []

        candidate_count = min(k * config.RETRIEVAL_CANDIDATE_MULTIPLIER, total)
        matches: List[ChunkMatch] = self.knowledge_base.query(
            query_embedding=query_embedding,
            where=where,
            n_results=candidate_count,
        )

        candidates: List[RetrievedChunk] = []
        for match in matches:
            score = 1.0 - match.distance
            if score < self.similarity_threshold:
                continue
            candidates.append(
                RetrievedChunk(
                    chunk_id=match.chunk_id,
                    report_id=match.report_id,
                    document=match.document,
                    metadata=match.metadata,
                    score=score,
                )
            )

        return self._select_with_cap(candidates, k, max_per_report)

    @staticmethod
    def _chunk_rank(chunk: RetrievedChunk) -> tuple:
        """Prefer evaluasi_tim/saran_pengembangan chunks, then the higher score."""
        section = str(chunk.metadata.get("section_type") or "")
        preferred = config.COMPARE_PREFERRED_SECTIONS
        section_rank = (
            preferred.index(section) if section in preferred else len(preferred)
        )
        return (section_rank, -(chunk.score or 0.0), chunk.chunk_id)

    def _select_one_chunk_per_report(
        self,
        candidates: Sequence[RetrievedChunk],
        max_reports: int,
    ) -> List[RetrievedChunk]:
        """One best chunk per distinct report, at most max_reports reports."""
        best_per_report: Dict[str, RetrievedChunk] = {}
        for candidate in candidates:
            current = best_per_report.get(candidate.report_id)
            if current is None or self._chunk_rank(candidate) < self._chunk_rank(current):
                best_per_report[candidate.report_id] = candidate

        ordered = sorted(
            best_per_report.values(),
            key=lambda chunk: (-(chunk.score or 0.0), chunk.report_id, chunk.chunk_id),
        )
        return ordered[: int(max_reports)]

    def _query_group_reports(
        self,
        query_embedding: Sequence[float],
        where: Optional[Dict[str, Any]],
        max_reports: int,
        total: int,
    ) -> List[RetrievedChunk]:
        """Grouped comparison: distinct reports, one chunk each (fix pack H3a)."""
        if max_reports <= 0 or total == 0:
            return []

        candidate_count = min(
            int(max_reports) * config.RETRIEVAL_COMPARE_CANDIDATE_MULTIPLIER, total
        )
        matches: List[ChunkMatch] = self.knowledge_base.query(
            query_embedding=query_embedding,
            where=where,
            n_results=candidate_count,
        )

        # Normalize the provider's order before selection so only the explicit
        # tie-break decides (fix pack K1c).
        matches = sorted(matches, key=lambda match: match.chunk_id)
        candidates = [
            RetrievedChunk(
                chunk_id=match.chunk_id,
                report_id=match.report_id,
                document=match.document,
                metadata=match.metadata,
                score=1.0 - match.distance,
            )
            for match in matches
            if (1.0 - match.distance) >= self.similarity_threshold
        ]
        return self._select_one_chunk_per_report(candidates, max_reports)

    def retrieve_grouped(
        self,
        search_query: str,
        mode: QueryMode,
        filters: Optional[Dict[str, Sequence[Any]]],
        compare_by: str,
    ) -> tuple[List[RetrievedChunk], List[Dict[str, Any]]]:
        """
        One sub-retrieval per comparison group.

        Each group contributes up to COMPARE_REPORTS_PER_GROUP distinct reports
        with exactly one chunk per report, so one team cannot be represented twice
        while another team disappears. A group whose own filter contradicts the
        analyzer filters (an empty intersection) matches nothing, and the returned
        status says so instead of silently dropping the constraint. Results are
        deduplicated by chunk id.
        """
        if mode is not QueryMode.CROSS_REPORT:
            raise RetrievalError(
                f"compare_by is only supported for cross_report, not '{mode}'."
            )
        if compare_by not in config.COMPARISON_GROUPS:
            raise RetrievalError(f"Unknown comparison grouping '{compare_by}'.")

        groups = config.COMPARISON_GROUPS[compare_by]
        max_reports = int(config.COMPARE_REPORTS_PER_GROUP)

        total = self.knowledge_base.count_chunks()
        if total == 0:
            return [], []

        query_embedding = self._embed_query(search_query)

        merged: List[RetrievedChunk] = []
        seen: set = set()
        status: List[Dict[str, Any]] = []

        for group in groups:
            group_filters = merge_filter_dicts(filters, group.get("filter"))

            if any(not values for values in group_filters.values()):
                status.append({"label": group["label"], "count": 0})
                continue

            chunks = self._query_group_reports(
                query_embedding,
                build_where_filter(group_filters),
                max_reports,
                total,
            )
            status.append({"label": group["label"], "count": len(chunks)})

            for chunk in chunks:
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                merged.append(replace(chunk, group_label=group["label"]))

        return merged, status

    # -- exact ID route ----------------------------------------------------
    def retrieve_by_report_ids(self, report_ids: Sequence[str]) -> List[RetrievedChunk]:
        """Fetch every chunk of the given reports with an exact where filter."""
        stored: List[StoredChunk] = self.knowledge_base.get_by_report_ids(report_ids)
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                report_id=chunk.report_id,
                document=chunk.document,
                metadata=chunk.metadata,
                score=None,
            )
            for chunk in stored
        ]

    # -- catalog -----------------------------------------------------------
    def list_catalog(
        self,
        filters: Optional[Dict[str, Sequence[Any]]] = None,
        max_reports: int = config.RETRIEVAL_CATALOG_MAX_REPORTS,
    ) -> List[Dict[str, Any]]:
        """
        Return a deduped, bounded list of reports (metadata only, no embedding).

        When several chunks of a report qualify, the first one seen wins; all of
        them carry the same report metadata.
        """
        where = build_where_filter(filters)
        metadatas = self.knowledge_base.list_report_metadatas(where=where)

        by_report: Dict[str, Dict[str, Any]] = {}
        for metadata in metadatas:
            report_id = metadata.get("report_id")
            if not report_id or report_id in by_report:
                continue
            by_report[report_id] = {
                field: metadata.get(field) for field in CATALOG_METADATA_FIELDS
            }

        ordered = [by_report[report_id] for report_id in sorted(by_report)]
        return ordered[: int(max_reports)]
