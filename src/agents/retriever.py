"""
agents/retriever.py — Retriever agent: literature and patent mining.

Fixed bugs:
  - ChromaDB upsert: embeddings must be list[list[float]], IDs must be strings,
    documents must be non-empty strings. Added full validation before upsert.
  - Semantic search: guard against top_k=0 and empty result sets.
  - Batch summarisation: abstracts passed as dicts; must extract 'text' field
    before joining into a string for the LLM prompt.
  - ArXiv query string: passed indication directly, not as drug discovery query.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from config import (
    FeatureFlags,
    ModelConfig,
    RetrievalConfig,
    VectorStoreConfig,
    PipelineConfig,
)
from schemas import BatchSummary, RetrieverResponse, TargetCandidate
from utils.ollama_client import OllamaClient
from utils.prompts import (
    RETRIEVER_SUMMARISE_PROMPT,
    RETRIEVER_RANK_PROMPT,
    SYSTEM_ANALYST,
    JSON_ENFORCEMENT,
    build_agent_prompt,
)
from utils.helpers import log_json_event, utc_now_iso, safe_json_dumps
from utils.json_validator import JSONValidator
from utils.context_manager import ContextManager


class RetrieverAgent:
    """
    Mines scientific literature and patents to identify druggable targets.

    Batch strategy:
      - Abstracts are processed in groups, each batch summarised by the LLM.
      - All batch summaries are aggregated then re-ranked by a second LLM call.
      - ChromaDB stores embeddings for semantic search across the corpus.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        self.client = ollama_client
        self.ctx = ContextManager()
        self._chroma_client: Any = None
        self._literature_collection: Any = None

    # ── ChromaDB initialisation ───────────────────────────────────────────────
    def _init_chromadb(self) -> bool:
        """
        Initialise ChromaDB persistent client and literature collection.

        Returns:
            bool: True if ChromaDB was successfully initialised.
        """
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(
                path=VectorStoreConfig.PERSIST_DIR
            )
            self._literature_collection = self._chroma_client.get_or_create_collection(
                name=VectorStoreConfig.LITERATURE_COLLECTION,
                metadata={"hnsw:space": VectorStoreConfig.DISTANCE_METRIC},
            )
            logger.debug(
                f"[Retriever] ChromaDB initialised at '{VectorStoreConfig.PERSIST_DIR}'"
            )
            return True
        except ImportError:
            logger.warning(
                "[Retriever] chromadb not installed. Semantic search disabled."
            )
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[Retriever] ChromaDB init failed: {exc}")
            return False

    # ── Main run method ───────────────────────────────────────────────────────
    async def run(self, state: dict) -> dict:
        """
        Execute the retrieval phase.

        Steps:
            1. Fetch literature from PubMed + arXiv.
            2. Optionally fetch patents.
            3. Embed valid abstracts into ChromaDB.
            4. Semantic search for most relevant abstracts.
            5. Batch-summarise via LLM.
            6. Rank targets via LLM.
            7. Enrich with UniProt data.

        Args:
            state: Pipeline state dict.

        Returns:
            dict: Validated RetrieverResponse as a dictionary.
        """
        indication = state["indication_or_target"]
        logger.info(f"[Retriever] Starting retrieval for: '{indication}'")

        # ── Step 1: Fetch literature ──────────────────────────────────────────
        from tools.literature_search import LiteratureSearch
        lit_search = LiteratureSearch()

        abstracts = await lit_search.search_pubmed(
            query=indication,
            max_results=RetrievalConfig.PUBMED_MAX_RESULTS,
        )
        logger.info(f"[Retriever] Fetched {len(abstracts)} PubMed abstracts.")

        arxiv_abstracts = await lit_search.search_arxiv(
            query=indication,
            max_results=RetrievalConfig.ARXIV_MAX_RESULTS,
        )
        logger.info(f"[Retriever] Fetched {len(arxiv_abstracts)} arXiv abstracts.")

        all_abstracts = abstracts + arxiv_abstracts
        total_papers = len(all_abstracts)
        logger.info(f"[Retriever] Fetched {total_papers} abstracts total.")

        # ── Step 2: Patents ───────────────────────────────────────────────────
        total_patents = 0
        patent_summary_text = "Patent search disabled."

        if FeatureFlags.ENABLE_PATENT_SEARCH:
            from tools.patent_search import PatentSearch
            patent_search = PatentSearch()
            patents = await patent_search.search(
                query=indication,
                max_results=RetrievalConfig.PATENTS_MAX_RESULTS,
            )
            total_patents = len(patents)
            patent_summary_text = self._summarise_patents(patents)
            logger.info(f"[Retriever] Patents: {total_patents} found.")
        else:
            logger.info("[Retriever] Patent search disabled.")

        # ── Step 3: Embed into ChromaDB ───────────────────────────────────────
        chroma_ok = self._init_chromadb()
        if chroma_ok and all_abstracts:
            await self._embed_documents(all_abstracts)

        # ── Step 4: Semantic search for top-k relevant abstracts ──────────────
        top_abstracts = await self._semantic_search(
            query=indication,
            all_abstracts=all_abstracts,
            top_k=VectorStoreConfig.TOP_K,
        )
        logger.info(
            f"[Retriever] Using {len(top_abstracts)} abstracts for LLM summarisation."
        )

        # ── Step 5: Batch-summarise ───────────────────────────────────────────
        batch_summaries = await self._summarise_literature_batches(
            abstracts=top_abstracts,
            indication=indication,
        )
        logger.info(
            f"[Retriever] Processed {len(batch_summaries)} literature batches."
        )

        # ── Step 6: Rank targets ──────────────────────────────────────────────
        retriever_response = await self._rank_targets(
            batch_summaries=batch_summaries,
            patent_summary=patent_summary_text,
            indication=indication,
            total_papers=total_papers,
            total_patents=total_patents,
        )

        # ── Step 7: UniProt enrichment ────────────────────────────────────────
        retriever_response = await self._enrich_with_uniprot(retriever_response)

        top_gene = (
            retriever_response.target_candidates[0].gene_name
            if retriever_response.target_candidates else "none"
        )
        logger.success(
            f"[Retriever] Found {len(retriever_response.target_candidates)} candidates. "
            f"Top: {top_gene}"
        )

        log_json_event(
            PipelineConfig.PIPELINE_LOG,
            {
                "event": "retriever_output",
                "task_id": state.get("task_id"),
                "total_papers": total_papers,
                "total_patents": total_patents,
                "candidate_count": len(retriever_response.target_candidates),
                "top_targets": [
                    c.gene_name for c in retriever_response.get_top_candidates(3)
                ],
                "timestamp": utc_now_iso(),
            },
        )

        return retriever_response.model_dump()

    # ── Embedding ─────────────────────────────────────────────────────────────
    async def _embed_documents(self, abstracts: list[dict]) -> None:
        """
        Embed abstract texts into ChromaDB.

        Validates every field before calling upsert:
          - text must be a non-empty string
          - id must be a unique string
          - embeddings must be list[list[float]] with correct dimensionality

        Args:
            abstracts: List of abstract dicts with 'text', 'pmid' keys.
        """
        if self._literature_collection is None:
            return

        # ── Build validated parallel lists ────────────────────────────────────
        valid_texts: list[str] = []
        valid_ids: list[str] = []
        valid_metas: list[dict] = []
        seen_ids: set[str] = set()

        for i, abstract in enumerate(abstracts):
            text = abstract.get("text", "")

            # Skip non-string or empty texts
            if not isinstance(text, str) or not text.strip():
                logger.debug(f"[Retriever] Skipping abstract {i}: empty text.")
                continue

            # Build unique string ID
            raw_id = abstract.get("pmid", f"doc_{i}")
            uid = str(raw_id).strip()
            if not uid:
                uid = f"doc_{i}"

            # Deduplicate IDs
            if uid in seen_ids:
                uid = f"{uid}_{i}"
            seen_ids.add(uid)

            valid_texts.append(text[:2000])   # cap at 2000 chars
            valid_ids.append(uid)
            valid_metas.append({
                "source": str(abstract.get("source", "pubmed")),
                "title":  str(abstract.get("title", ""))[:200],
                "year":   str(abstract.get("year", "unknown")),
            })

        if not valid_texts:
            logger.warning("[Retriever] No valid texts to embed.")
            return

        logger.debug(f"[Retriever] Embedding {len(valid_texts)} documents …")

        # ── Embed in batches ──────────────────────────────────────────────────
        batch_size = VectorStoreConfig.EMBEDDING_BATCH_SIZE

        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = batch_start + batch_size
            b_texts = valid_texts[batch_start:batch_end]
            b_ids   = valid_ids[batch_start:batch_end]
            b_metas = valid_metas[batch_start:batch_end]

            try:
                # embed() returns list[list[float]]
                embeddings: list[list[float]] = await self.client.embed(b_texts)

                # Validate embedding shape before upsert
                if len(embeddings) != len(b_texts):
                    logger.warning(
                        f"[Retriever] Embedding count mismatch: "
                        f"got {len(embeddings)}, expected {len(b_texts)}. "
                        f"Skipping batch."
                    )
                    continue

                # Every element must be a list of floats
                clean_embeddings: list[list[float]] = []
                clean_texts:      list[str]         = []
                clean_ids:        list[str]         = []
                clean_metas:      list[dict]        = []

                for emb, text, uid, meta in zip(
                    embeddings, b_texts, b_ids, b_metas
                ):
                    if not isinstance(emb, list) or len(emb) == 0:
                        logger.debug(
                            f"[Retriever] Skipping invalid embedding for id={uid}"
                        )
                        continue
                    # Ensure all values are plain Python floats
                    try:
                        emb_floats = [float(v) for v in emb]
                    except (TypeError, ValueError) as e:
                        logger.debug(
                            f"[Retriever] Cannot coerce embedding to float: {e}"
                        )
                        continue

                    clean_embeddings.append(emb_floats)
                    clean_texts.append(text)
                    clean_ids.append(uid)
                    clean_metas.append(meta)

                if not clean_embeddings:
                    logger.warning(
                        f"[Retriever] Batch {batch_start // batch_size + 1}: "
                        f"no valid embeddings after validation."
                    )
                    continue

                self._literature_collection.upsert(
                    documents=clean_texts,
                    embeddings=clean_embeddings,
                    ids=clean_ids,
                    metadatas=clean_metas,
                )
                logger.debug(
                    f"[Retriever] Upserted batch "
                    f"{batch_start // batch_size + 1}: "
                    f"{len(clean_embeddings)} embeddings."
                )

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[Retriever] Embedding batch "
                    f"{batch_start // batch_size + 1} failed: {exc}"
                )

    # ── Semantic search ───────────────────────────────────────────────────────
    async def _semantic_search(
        self,
        query: str,
        all_abstracts: list[dict],
        top_k: int = 20,
    ) -> list[dict]:
        """
        Query ChromaDB for the most relevant abstracts.

        Falls back to returning the first top_k abstracts from the full
        list if ChromaDB is unavailable or the collection is empty.

        Args:
            query: Semantic search query string.
            all_abstracts: Full list of fetched abstract dicts (fallback).
            top_k: Number of top results to retrieve.

        Returns:
            list[dict]: Top-k abstract dicts (from ChromaDB or fallback).
        """
        # Ensure top_k is a valid positive integer
        effective_top_k = max(1, int(top_k))

        if self._literature_collection is None:
            logger.info(
                "[Retriever] ChromaDB unavailable. "
                f"Using first {effective_top_k} abstracts as fallback."
            )
            return all_abstracts[:effective_top_k]

        try:
            # Check how many documents are actually in the collection
            collection_count = self._literature_collection.count()
            if collection_count == 0:
                logger.warning(
                    "[Retriever] ChromaDB collection is empty. "
                    f"Using first {effective_top_k} abstracts as fallback."
                )
                return all_abstracts[:effective_top_k]

            # Clamp top_k to collection size
            safe_top_k = min(effective_top_k, collection_count)

            # Embed the query with the same model used to embed documents.
            # Do NOT pass query_texts — that would trigger ChromaDB's default
            # embedding function (all-MiniLM-L6-v2, 384-dim), which mismatches
            # any collection built with a different model (e.g. nomic-embed-text,
            # 768-dim), causing "Collection expecting dimension X, got Y".
            query_embeddings: list[list[float]] = await self.client.embed([query])
            if not query_embeddings or not query_embeddings[0]:
                logger.warning(
                    "[Retriever] Failed to embed query. Using fallback."
                )
                return all_abstracts[:effective_top_k]

            results = self._literature_collection.query(
                query_embeddings=query_embeddings,
                n_results=safe_top_k,
            )

            # results["documents"] is list[list[str]] — one list per query
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            if not docs:
                logger.warning(
                    "[Retriever] ChromaDB query returned no documents. "
                    "Using fallback."
                )
                return all_abstracts[:effective_top_k]

            # Reconstruct abstract dicts from ChromaDB results
            reconstructed: list[dict] = []
            for doc, meta in zip(docs, metadatas):
                if isinstance(doc, str) and doc.strip():
                    reconstructed.append({
                        "text":   doc,
                        "title":  meta.get("title", "") if meta else "",
                        "source": meta.get("source", "unknown") if meta else "unknown",
                        "pmid":   "",
                        "year":   meta.get("year", "") if meta else "",
                        "authors":  [],
                        "journal":  "",
                    })

            logger.info(
                f"[Retriever] Semantic search returned {len(reconstructed)} abstracts."
            )
            return reconstructed

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Retriever] Semantic search failed: {exc}. "
                f"Using first {effective_top_k} abstracts as fallback."
            )
            return all_abstracts[:effective_top_k]

    # ── Batch summarisation ───────────────────────────────────────────────────
    async def _summarise_literature_batches(
        self,
        abstracts: list[dict],
        indication: str,
    ) -> list[BatchSummary]:
        """
        Process abstracts in batches, summarising each batch via LLM.

        Each element in `abstracts` is a dict with a 'text' key.
        This method extracts the text strings before grouping into batches.

        Args:
            abstracts: List of abstract dicts (must have 'text' key).
            indication: Disease/target query for prompt context.

        Returns:
            list[BatchSummary]: One BatchSummary per processed batch.
        """
        # ── Extract plain text strings from abstract dicts ────────────────────
        abstract_texts: list[str] = []
        for item in abstracts:
            if isinstance(item, dict):
                text = item.get("text", "")
            elif isinstance(item, str):
                text = item
            else:
                continue

            if isinstance(text, str) and text.strip():
                abstract_texts.append(text.strip())

        if not abstract_texts:
            logger.warning("[Retriever] No valid abstract texts for summarisation.")
            return []

        logger.info(
            f"[Retriever] Summarising {len(abstract_texts)} abstracts in batches …"
        )

        # ── Group into token-safe batches ─────────────────────────────────────
        batches = self.ctx.chunk_documents(
            abstract_texts,
            max_tokens_per_batch=3500,
        )
        total_batches = len(batches)
        logger.info(f"[Retriever] {total_batches} batch(es) to process.")

        summaries: list[BatchSummary] = []

        for batch_idx, batch in enumerate(batches, start=1):
            logger.debug(
                f"[Retriever] Summarising batch {batch_idx}/{total_batches} "
                f"({len(batch)} abstracts) …"
            )
            summary = await self._summarise_single_batch(
                batch=batch,
                batch_number=batch_idx,
                total_batches=total_batches,
                indication=indication,
            )
            if summary is not None:
                summaries.append(summary)

        return summaries

    async def _summarise_single_batch(
        self,
        batch: list[str],
        batch_number: int,
        total_batches: int,
        indication: str,
    ) -> Optional[BatchSummary]:
        """
        Summarise a single batch of abstract text strings via one LLM call.

        Args:
            batch: List of plain abstract text strings.
            batch_number: 1-indexed batch number.
            total_batches: Total batch count.
            indication: Disease/target query for context.

        Returns:
            BatchSummary | None: Parsed summary or None on failure.
        """
        # Join plain text strings with separator
        abstracts_text = "\n---\n".join(
            f"[{i + 1}] {text}" for i, text in enumerate(batch)
        )
        abstracts_text = self.ctx.truncate_to_tokens(abstracts_text, max_tokens=3000)

        user_prompt = RETRIEVER_SUMMARISE_PROMPT.format(
            indication_or_target=indication,
            batch_number=batch_number,
            total_batches=total_batches,
            abstracts_text=abstracts_text,
            json_enforcement=JSON_ENFORCEMENT,
        )

        messages = build_agent_prompt(
            system=SYSTEM_ANALYST,
            user=user_prompt,
            assistant_primer="{",
        )

        try:
            raw_response = await self.client.chat(
                messages=messages,
                schema=BatchSummary,
                temperature=ModelConfig.JSON_TEMPERATURE,
                context_label=f"retriever_batch_{batch_number}",
            )
            parsed = JSONValidator.assert_valid(raw_response, BatchSummary)
            parsed["batch_number"] = batch_number
            parsed["papers_processed"] = len(batch)
            return BatchSummary.model_validate(parsed)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Retriever] Batch {batch_number} summarisation failed: {exc}. "
                "Skipping batch."
            )
            return None

    # ── Target ranking ────────────────────────────────────────────────────────
    async def _rank_targets(
        self,
        batch_summaries: list[BatchSummary],
        patent_summary: str,
        indication: str,
        total_papers: int,
        total_patents: int,
    ) -> RetrieverResponse:
        """
        Aggregate batch summaries and rank targets via a final LLM call.

        Falls back to a minimal constructed response if the LLM call fails,
        so the pipeline can continue to the hypothesis phase.

        Args:
            batch_summaries: All BatchSummary objects.
            patent_summary: Compact patent summary string.
            indication: Disease/target query.
            total_papers: Total papers processed.
            total_patents: Total patents reviewed.

        Returns:
            RetrieverResponse: Validated and ranked target list.
        """
        all_findings: list[dict] = []
        all_key_findings: list[str] = []

        for summary in batch_summaries:
            for target in summary.targets_found:
                all_findings.append(target.model_dump())
            all_key_findings.extend(summary.key_findings[:3])

        evidence_summary = self._build_evidence_summary(
            findings=all_findings,
            key_findings=all_key_findings[:15],
            patent_summary=patent_summary,
        )
        evidence_summary = self.ctx.truncate_to_tokens(evidence_summary, 4000)

        user_prompt = RETRIEVER_RANK_PROMPT.format(
            indication_or_target=indication,
            total_papers=total_papers,
            total_patents=total_patents,
            evidence_summary=evidence_summary,
            timestamp=utc_now_iso(),
            json_enforcement=JSON_ENFORCEMENT,
        )

        messages = build_agent_prompt(
            system=SYSTEM_ANALYST,
            user=user_prompt,
            assistant_primer="{",
        )

        try:
            raw_response = await self.client.chat(
                messages=messages,
                schema=RetrieverResponse,
                temperature=ModelConfig.JSON_TEMPERATURE,
                context_label="retriever_ranking",
            )
            parsed = JSONValidator.assert_valid(raw_response, RetrieverResponse)
            parsed["total_papers_reviewed"] = total_papers
            parsed["total_patents_reviewed"] = total_patents
            parsed["retrieval_timestamp"] = utc_now_iso()
            return RetrieverResponse.model_validate(parsed)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[Retriever] Target ranking LLM call failed: {exc}. "
                "Building fallback response from raw findings."
            )
            return self._build_fallback_response(
                findings=all_findings,
                indication=indication,
                total_papers=total_papers,
                total_patents=total_patents,
            )

    @staticmethod
    def _build_fallback_response(
        findings: list[dict],
        indication: str,
        total_papers: int,
        total_patents: int,
    ) -> RetrieverResponse:
        """
        Build a minimal RetrieverResponse from raw findings when LLM ranking fails.

        Deduplicates by gene name and constructs TargetCandidate objects
        with default druggability scores.

        Args:
            findings: Raw TargetFinding dicts from batch summaries.
            indication: Disease/target indication string.
            total_papers: Total papers reviewed.
            total_patents: Total patents reviewed.

        Returns:
            RetrieverResponse: Minimal but valid response.
        """
        seen: dict[str, dict] = {}
        for f in findings:
            gene = f.get("gene_name", "UNKNOWN").upper().strip()
            if gene and gene not in seen:
                seen[gene] = f

        candidates: list[TargetCandidate] = []
        for gene, f in list(seen.items())[:10]:
            try:
                candidates.append(
                    TargetCandidate(
                        gene_name=gene,
                        uniprot_id=None,
                        pdb_ids=[],
                        evidence_summary=f.get(
                            "evidence_summary",
                            f"Found in literature for {indication}.",
                        )[:500],
                        literature_citations=f.get("citations", []),
                        patent_count=0,
                        druggability_score=0.5,
                        novelty_score=0.5,
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        if not candidates:
            # Absolute last resort — one generic candidate so pipeline continues
            candidates = [
                TargetCandidate(
                    gene_name="UNKNOWN",
                    uniprot_id=None,
                    pdb_ids=[],
                    evidence_summary=(
                        f"Target identification from literature for {indication}."
                    ),
                    literature_citations=[],
                    patent_count=0,
                    druggability_score=0.5,
                    novelty_score=0.5,
                )
            ]

        return RetrieverResponse(
            target_candidates=candidates,
            total_papers_reviewed=total_papers,
            total_patents_reviewed=total_patents,
            retrieval_timestamp=utc_now_iso(),
        )

    # ── UniProt enrichment ────────────────────────────────────────────────────
    async def _enrich_with_uniprot(
        self,
        response: RetrieverResponse,
    ) -> RetrieverResponse:
        """
        Enrich target candidates with UniProt accessions and PDB IDs.

        Args:
            response: RetrieverResponse to enrich.

        Returns:
            RetrieverResponse: Enriched response.
        """
        try:
            from tools.target_lookup import TargetLookup
            lookup = TargetLookup()

            for candidate in response.target_candidates:
                if candidate.uniprot_id:
                    continue
                if candidate.gene_name in {"UNKNOWN", ""}:
                    continue

                logger.debug(
                    f"[Retriever] Fetching UniProt for {candidate.gene_name} …"
                )
                uniprot_data = await lookup.get_uniprot_info(candidate.gene_name)

                if uniprot_data:
                    candidate.uniprot_id = uniprot_data.get("uniprot_id")
                    pdb_ids = uniprot_data.get("pdb_ids", [])
                    if pdb_ids and not candidate.pdb_ids:
                        candidate.pdb_ids = pdb_ids[:5]

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Retriever] UniProt enrichment failed: {exc}. "
                "Continuing with available data."
            )

        return response

    # ── Static helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _summarise_patents(patents: list[dict]) -> str:
        """Build a compact patent summary string."""
        if not patents:
            return "No patents found."
        lines = [f"Total patents found: {len(patents)}\n"]
        for i, patent in enumerate(patents[:10], start=1):
            title    = str(patent.get("title", "Unknown title"))[:100]
            assignee = str(patent.get("assignee", "Unknown"))[:50]
            lines.append(f"{i}. {title} (Assignee: {assignee})")
        if len(patents) > 10:
            lines.append(f"… and {len(patents) - 10} more patents.")
        return "\n".join(lines)

    @staticmethod
    def _build_evidence_summary(
        findings: list[dict],
        key_findings: list[str],
        patent_summary: str,
    ) -> str:
        """Aggregate findings into a structured evidence summary string."""
        gene_evidence: dict[str, list[str]] = {}
        for finding in findings:
            gene = str(finding.get("gene_name", "UNKNOWN")).upper().strip()
            summary = str(finding.get("evidence_summary", ""))
            ev_type = str(finding.get("evidence_type", "unknown"))
            if gene not in gene_evidence:
                gene_evidence[gene] = []
            gene_evidence[gene].append(f"[{ev_type}] {summary}")

        target_lines = ["IDENTIFIED TARGETS:"]
        for gene, ev_list in sorted(
            gene_evidence.items(), key=lambda x: -len(x[1])
        )[:15]:
            target_lines.append(f"\n  {gene} ({len(ev_list)} mentions):")
            for ev in ev_list[:2]:
                target_lines.append(f"    - {ev[:150]}")

        findings_lines = ["\nKEY MECHANISTIC FINDINGS:"]
        for i, finding in enumerate(key_findings[:10], start=1):
            findings_lines.append(f"  {i}. {finding[:150]}")

        return (
            "\n".join(target_lines)
            + "\n"
            + "\n".join(findings_lines)
            + f"\n\nPATENT LANDSCAPE:\n{patent_summary}"
        )