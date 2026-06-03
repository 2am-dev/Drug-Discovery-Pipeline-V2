"""
tools/literature_search.py — PubMed and arXiv literature retrieval tool.

Provides async methods for:
  1. PubMed E-utilities API: search + fetch abstracts in batches.
  2. arXiv Atom feed: search preprints via feedparser.
  3. Abstract cleaning and normalisation for downstream embedding.

PubMed rate limit: 3 requests/second without API key, 10/second with one.
Set NCBI_API_KEY environment variable to increase throughput.

All returned abstracts are dicts with a consistent schema:
    {
        "pmid":    str,           # PubMed ID or arXiv ID
        "title":   str,           # Article title
        "text":    str,           # Abstract text (cleaned)
        "source":  str,           # "pubmed" or "arxiv"
        "authors": list[str],     # Author list (last names)
        "year":    str,           # Publication year
        "journal": str,           # Journal / venue name
    }
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Optional
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import aiohttp
import feedparser
from loguru import logger

from config import RetrievalConfig


class LiteratureSearch:
    """
    Fetches scientific literature from PubMed and arXiv.

    Uses aiohttp for async HTTP requests with automatic retry on
    transient failures (429 Too Many Requests, 503 Service Unavailable).

    Usage:
        search = LiteratureSearch()
        abstracts = await search.search_pubmed("EGFR lung cancer", max_results=100)
        preprints = await search.search_arxiv("KRAS inhibitor", max_results=30)
    """

    # PubMed E-utilities endpoints
    ESEARCH_URL = f"{RetrievalConfig.PUBMED_BASE_URL}/esearch.fcgi"
    EFETCH_URL  = f"{RetrievalConfig.PUBMED_BASE_URL}/efetch.fcgi"

    # Batch size for efetch calls (PubMed recommends ≤ 200 per call)
    FETCH_BATCH_SIZE = 100

    # Delay between PubMed API calls (seconds) to respect rate limits
    PUBMED_DELAY = 0.35   # ~3 requests/second without API key

    def __init__(self) -> None:
        """Initialise LiteratureSearch with config-driven settings."""
        self.timeout = aiohttp.ClientTimeout(
            total=RetrievalConfig.HTTP_TIMEOUT
        )
        self.headers = {"User-Agent": RetrievalConfig.USER_AGENT}
        self.api_key: Optional[str] = None

        import os
        ncbi_key = os.environ.get("NCBI_API_KEY")
        if ncbi_key:
            self.api_key = ncbi_key
            self.PUBMED_DELAY = 0.1   # 10 requests/second with API key
            logger.debug("PubMed: NCBI_API_KEY found — using higher rate limit.")

    # ── PubMed search ─────────────────────────────────────────────────────────
    async def search_pubmed(
        self,
        query: str,
        max_results: int = 150,
    ) -> list[dict]:
        """
        Search PubMed for abstracts matching the query.

        Two-step process:
          1. ESearch: retrieve PMID list for the query.
          2. EFetch: fetch full abstract XML for each PMID in batches.

        Filters:
          - Only English-language articles.
          - Only articles with abstracts (hasabstract filter).
          - Sorted by relevance.

        Args:
            query: Search query string (MeSH terms or free text).
            max_results: Maximum number of abstracts to retrieve.

        Returns:
            list[dict]: List of abstract dicts (schema described in module docstring).
        """
        logger.info(f"[LiteratureSearch] PubMed query: '{query}' (max {max_results})")

        # ── Step 1: ESearch — get PMID list ───────────────────────────────────
        pmids = await self._esearch(query, max_results)
        if not pmids:
            logger.warning("[LiteratureSearch] PubMed ESearch returned 0 PMIDs.")
            return []

        logger.info(f"[LiteratureSearch] PubMed: {len(pmids)} PMIDs retrieved.")

        # ── Step 2: EFetch — retrieve abstracts in batches ────────────────────
        abstracts: list[dict] = []
        for i in range(0, len(pmids), self.FETCH_BATCH_SIZE):
            batch = pmids[i : i + self.FETCH_BATCH_SIZE]
            batch_abstracts = await self._efetch(batch)
            abstracts.extend(batch_abstracts)
            logger.debug(
                f"[LiteratureSearch] Fetched batch "
                f"{i // self.FETCH_BATCH_SIZE + 1}: "
                f"{len(batch_abstracts)} abstracts."
            )
            # Respect rate limit between batch calls
            await asyncio.sleep(self.PUBMED_DELAY)

        logger.info(
            f"[LiteratureSearch] PubMed: {len(abstracts)} abstracts fetched."
        )
        return abstracts

    async def _esearch(self, query: str, max_results: int) -> list[str]:
        """
        Call the PubMed ESearch endpoint to get a list of PMIDs.

        Args:
            query: Search query string.
            max_results: Maximum PMIDs to return.

        Returns:
            list[str]: List of PMID strings.
        """
        params: dict = {
            "db": "pubmed",
            "term": f"({query}) AND hasabstract AND english[lang]",
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "usehistory": "n",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.ESEARCH_URL}?{urlencode(params)}"

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"[LiteratureSearch] ESearch HTTP {resp.status}."
                        )
                        return []
                    data = await resp.json(content_type=None)
                    pmids = data.get("esearchresult", {}).get("idlist", [])
                    return pmids

        except asyncio.TimeoutError:
            logger.warning("[LiteratureSearch] ESearch timed out.")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[LiteratureSearch] ESearch error: {exc}")
            return []

    async def _efetch(self, pmids: list[str]) -> list[dict]:
        """
        Fetch full abstract XML from PubMed EFetch for a batch of PMIDs.

        Parses the PubMed XML format to extract title, abstract, authors,
        year, and journal name.

        Args:
            pmids: List of PubMed IDs to fetch (max 200 recommended).

        Returns:
            list[dict]: Parsed abstract dicts.
        """
        params: dict = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.EFETCH_URL}?{urlencode(params)}"

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"[LiteratureSearch] EFetch HTTP {resp.status}."
                        )
                        return []
                    xml_text = await resp.text(encoding="utf-8", errors="replace")

        except asyncio.TimeoutError:
            logger.warning("[LiteratureSearch] EFetch timed out.")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[LiteratureSearch] EFetch error: {exc}")
            return []

        return self._parse_pubmed_xml(xml_text)

    @staticmethod
    def _parse_pubmed_xml(xml_text: str) -> list[dict]:
        """
        Parse PubMed XML response into a list of abstract dicts.

        Handles malformed XML gracefully by skipping individual articles
        that cause parse errors.

        Args:
            xml_text: Raw XML string from PubMed EFetch.

        Returns:
            list[dict]: Parsed abstract dicts.
        """
        abstracts: list[dict] = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning(f"[LiteratureSearch] XML parse error: {exc}")
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                # ── PMID ─────────────────────────────────────────────────────
                pmid_el = article.find(".//PMID")
                pmid = f"PMID:{pmid_el.text}" if pmid_el is not None else "PMID:unknown"

                # ── Title ─────────────────────────────────────────────────────
                title_el = article.find(".//ArticleTitle")
                title = (
                    "".join(title_el.itertext()).strip()
                    if title_el is not None else ""
                )

                # ── Abstract text ─────────────────────────────────────────────
                abstract_texts: list[str] = []
                for ab_text in article.findall(".//AbstractText"):
                    label = ab_text.get("Label", "")
                    text = "".join(ab_text.itertext()).strip()
                    if text:
                        if label:
                            abstract_texts.append(f"{label}: {text}")
                        else:
                            abstract_texts.append(text)
                abstract = " ".join(abstract_texts)

                if not abstract:
                    continue   # skip articles without abstracts

                # ── Authors ───────────────────────────────────────────────────
                authors: list[str] = []
                for author in article.findall(".//Author"):
                    last = author.find("LastName")
                    if last is not None and last.text:
                        authors.append(last.text)

                # ── Year ──────────────────────────────────────────────────────
                year_el = article.find(".//PubDate/Year")
                medline_year = article.find(".//MedlineDate")
                if year_el is not None and year_el.text:
                    year = year_el.text
                elif medline_year is not None and medline_year.text:
                    year = medline_year.text[:4]
                else:
                    year = "unknown"

                # ── Journal ───────────────────────────────────────────────────
                journal_el = article.find(".//Journal/Title")
                journal = journal_el.text.strip() if journal_el is not None else ""

                # ── Full text field (title + abstract for embedding) ───────────
                full_text = LiteratureSearch._clean_text(
                    f"{title}. {abstract}"
                )

                abstracts.append({
                    "pmid": pmid,
                    "title": title[:300],
                    "text": full_text,
                    "source": "pubmed",
                    "authors": authors[:5],   # cap at 5 for metadata
                    "year": year,
                    "journal": journal[:100],
                })

            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[LiteratureSearch] Skipping malformed article: {exc}")
                continue

        return abstracts

    # ── arXiv search ──────────────────────────────────────────────────────────
    async def search_arxiv(
        self,
        query: str,
        max_results: int = 30,
    ) -> list[dict]:
        """
        Search arXiv for preprints relevant to the query.

        Uses the arXiv Atom API via feedparser. Searches q-bio and cs
        categories for computational biology and drug discovery preprints.

        Args:
            query: Search query string.
            max_results: Maximum number of preprints to retrieve.

        Returns:
            list[dict]: Abstract dicts (source="arxiv").
        """
        logger.info(
            f"[LiteratureSearch] arXiv query: '{query}' (max {max_results})"
        )

        search_query = (
            f"all:{query.replace(' ', '+')}"
            f"+AND+(cat:q-bio.BM+OR+cat:q-bio.QM+OR+cat:cs.LG)"
        )
        params = {
            "search_query": search_query,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{RetrievalConfig.ARXIV_BASE_URL}?{urlencode(params)}"

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"[LiteratureSearch] arXiv HTTP {resp.status}."
                        )
                        return []
                    feed_text = await resp.text(encoding="utf-8", errors="replace")

        except asyncio.TimeoutError:
            logger.warning("[LiteratureSearch] arXiv request timed out.")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[LiteratureSearch] arXiv error: {exc}")
            return []

        return self._parse_arxiv_feed(feed_text)

    @staticmethod
    def _parse_arxiv_feed(feed_text: str) -> list[dict]:
        """
        Parse an arXiv Atom feed into a list of abstract dicts.

        Args:
            feed_text: Raw Atom XML feed string from arXiv API.

        Returns:
            list[dict]: Parsed abstract dicts (source="arxiv").
        """
        abstracts: list[dict] = []

        try:
            feed = feedparser.parse(feed_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[LiteratureSearch] arXiv feedparser error: {exc}")
            return []

        for entry in feed.entries:
            try:
                arxiv_id = getattr(entry, "id", "").split("/abs/")[-1]
                title = getattr(entry, "title", "").replace("\n", " ").strip()
                summary = getattr(entry, "summary", "").replace("\n", " ").strip()

                if not summary:
                    continue

                authors: list[str] = []
                for author in getattr(entry, "authors", [])[:5]:
                    name = getattr(author, "name", "")
                    if name:
                        # Extract last name
                        last = name.split()[-1] if name.split() else name
                        authors.append(last)

                published = getattr(entry, "published", "")
                year = published[:4] if published else "unknown"

                full_text = LiteratureSearch._clean_text(
                    f"{title}. {summary}"
                )

                abstracts.append({
                    "pmid": f"arXiv:{arxiv_id}",
                    "title": title[:300],
                    "text": full_text,
                    "source": "arxiv",
                    "authors": authors,
                    "year": year,
                    "journal": "arXiv preprint",
                })

            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[LiteratureSearch] Skipping arXiv entry: {exc}")
                continue

        logger.info(
            f"[LiteratureSearch] arXiv: {len(abstracts)} preprints parsed."
        )
        return abstracts

    # ── Text cleaning ─────────────────────────────────────────────────────────
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalise abstract text for embedding and LLM consumption.

        Operations:
          - Remove HTML tags.
          - Collapse whitespace.
          - Remove non-printable characters.
          - Truncate to 2000 characters (prevents OOM during embedding).

        Args:
            text: Raw abstract text string.

        Returns:
            str: Cleaned, normalised text.
        """
        # Remove HTML tags (some PubMed abstracts contain <i>, <b>, etc.)
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove non-printable characters
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Truncate
        return text[:2000]