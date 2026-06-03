"""
tools/patent_search.py — PatentsView API client for drug patent retrieval.

Searches the PatentsView REST API (https://api.patentsview.org) for
patents related to a disease indication or drug target.

PatentsView API:
  - No authentication required for basic queries.
  - Rate limit: ~45 requests/minute.
  - Returns patents granted by the USPTO.

Each returned patent dict has:
    {
        "patent_id":  str,    # Patent number (e.g. "9,012,345")
        "title":      str,    # Patent title
        "assignee":   str,    # First assignee (company/institution)
        "year":       str,    # Grant year
        "abstract":   str,    # Patent abstract (may be None)
        "ipc_codes":  list,   # IPC classification codes
    }
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import aiohttp
from loguru import logger

from config import RetrievalConfig


class PatentSearch:
    """
    Queries the PatentsView API for drug-related patents.

    Constructs keyword queries from the indication/target name and
    retrieves patent counts and summaries for the competitive landscape
    section of the project proposal.

    Usage:
        search = PatentSearch()
        patents = await search.search("EGFR lung cancer", max_results=50)
        count   = await search.count_patents("EGFR inhibitor")
    """

    # PatentsView query fields to retrieve
    RETURN_FIELDS = [
        "patent_id",
        "patent_title",
        "patent_abstract",
        "patent_date",
        "assignee_organization",
        "ipc_subgroup_id",
    ]

    def __init__(self) -> None:
        """Initialise PatentSearch with configured timeout and headers."""
        self.base_url = RetrievalConfig.PATENTSVIEW_URL
        self.timeout = aiohttp.ClientTimeout(
            total=RetrievalConfig.HTTP_TIMEOUT
        )
        self.headers = {
            "User-Agent": RetrievalConfig.USER_AGENT,
            "Content-Type": "application/json",
        }

    async def search(
        self,
        query: str,
        max_results: int = 50,
    ) -> list[dict]:
        """
        Search PatentsView for patents matching the query.

        Builds a PatentsView JSON query using _text_contains on patent_title
        and patent_abstract. Falls back to an empty list on API error to
        avoid blocking the pipeline.

        Args:
            query: Search query string (disease name or target gene).
            max_results: Maximum number of patents to retrieve.

        Returns:
            list[dict]: Patent dicts (schema described in module docstring).
        """
        logger.info(
            f"[PatentSearch] PatentsView query: '{query}' (max {max_results})"
        )

        # Build keywords from the query
        keywords = self._extract_keywords(query)
        pv_query = self._build_query(keywords)

        payload = {
            "q": pv_query,
            "f": self.RETURN_FIELDS,
            "o": {"per_page": min(max_results, 100), "page": 1},
            "s": [{"patent_date": "desc"}],
        }

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
            ) as session:
                async with session.post(
                    self.base_url,
                    json=payload,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        patents = self._parse_response(data)
                        logger.info(
                            f"[PatentSearch] {len(patents)} patents retrieved."
                        )
                        return patents
                    elif resp.status == 429:
                        logger.warning(
                            "[PatentSearch] Rate limited (429). "
                            "Waiting 5s and retrying once."
                        )
                        await asyncio.sleep(5)
                        return await self.search(query, max_results)
                    else:
                        logger.warning(
                            f"[PatentSearch] HTTP {resp.status} from PatentsView. "
                            "Continuing without patent data."
                        )
                        return []

        except asyncio.TimeoutError:
            logger.warning("[PatentSearch] PatentsView request timed out.")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[PatentSearch] PatentsView error: {exc}. "
                "Continuing without patent data."
            )
            return []

    async def count_patents(self, query: str) -> int:
        """
        Return the total count of patents matching a query without fetching details.

        Args:
            query: Search query string.

        Returns:
            int: Total patent count (0 on error).
        """
        keywords = self._extract_keywords(query)
        pv_query = self._build_query(keywords)

        payload = {
            "q": pv_query,
            "f": ["patent_id"],
            "o": {"per_page": 1, "page": 1},
        }

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
            ) as session:
                async with session.post(self.base_url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        return data.get("total_patent_count", 0)
                    return 0
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """
        Extract the most relevant keywords from a query string.

        Removes common stop words and returns the 3 most specific terms.
        The PatentsView API performs best with 2-4 focused keywords.

        Args:
            query: Raw query string (e.g. "non-small cell lung cancer EGFR").

        Returns:
            list[str]: Filtered keyword list (max 4 terms).
        """
        stop_words = {
            "the", "a", "an", "and", "or", "for", "of", "in", "with",
            "non", "small", "cell", "type", "disease", "disorder",
            "treatment", "therapy", "inhibitor", "drug", "compound",
        }
        words = [
            w.strip("()[].,;:")
            for w in query.lower().split()
            if len(w) > 3 and w.lower() not in stop_words
        ]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = [w for w in words if not (w in seen or seen.add(w))]  # type: ignore
        return unique[:4]

    @staticmethod
    def _build_query(keywords: list[str]) -> dict:
        """
        Build a PatentsView JSON query dict for the given keywords.

        Uses _or to match patents containing ANY of the keywords in
        either the title or abstract.

        Args:
            keywords: List of keyword strings.

        Returns:
            dict: PatentsView-compatible query dict.
        """
        if not keywords:
            return {"_text_any": {"patent_title": "drug cancer"}}

        # Build OR conditions across title and abstract for each keyword
        conditions: list[dict] = []
        for kw in keywords:
            conditions.append({"_text_contains": {"patent_title": kw}})
            conditions.append({"_text_contains": {"patent_abstract": kw}})

        return {"_or": conditions}

    @staticmethod
    def _parse_response(data: dict) -> list[dict]:
        """
        Parse PatentsView API response into a list of patent dicts.

        Args:
            data: Parsed JSON response from PatentsView.

        Returns:
            list[dict]: Normalised patent dicts.
        """
        patents: list[dict] = []
        raw_patents = data.get("patents") or []

        for patent in raw_patents:
            if not isinstance(patent, dict):
                continue

            # Extract first assignee from list
            assignees = patent.get("assignees") or []
            assignee = ""
            if assignees and isinstance(assignees[0], dict):
                assignee = assignees[0].get("assignee_organization", "")

            # Extract IPC codes from list
            ipc_list = patent.get("IPCs") or []
            ipc_codes: list[str] = []
            for ipc in ipc_list[:5]:
                if isinstance(ipc, dict):
                    code = ipc.get("ipc_subgroup_id", "")
                    if code:
                        ipc_codes.append(code)

            # Extract year from patent_date (format: "YYYY-MM-DD")
            patent_date = patent.get("patent_date", "")
            year = patent_date[:4] if patent_date else "unknown"

            abstract = patent.get("patent_abstract") or ""

            patents.append({
                "patent_id": patent.get("patent_id", ""),
                "title": (patent.get("patent_title") or "")[:200],
                "assignee": assignee[:100],
                "year": year,
                "abstract": abstract[:500],
                "ipc_codes": ipc_codes,
            })

        return patents