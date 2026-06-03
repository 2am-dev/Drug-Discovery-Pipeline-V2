"""
tools/target_lookup.py — UniProt REST API client for protein target data.

Provides async methods for:
  1. Gene symbol → UniProt accession lookup.
  2. UniProt entry detail retrieval (binding sites, active sites, PDB IDs).
  3. PDB ID validation via RCSB PDB REST API.

UniProt REST API v2:
  - No authentication required.
  - Base: https://rest.uniprot.org/uniprotkb
  - Returns JSON by default.

RCSB PDB API:
  - Validation endpoint: https://data.rcsb.org/rest/v1/core/entry/{pdb_id}
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

import aiohttp
from loguru import logger

from config import RetrievalConfig, DockingConfig


class TargetLookup:
    """
    Retrieves protein structure and annotation data for drug targets.

    Provides UniProt entry details (binding sites, active site residues,
    associated PDB structures) and RCSB PDB ID verification.

    Usage:
        lookup = TargetLookup()
        info = await lookup.get_uniprot_info("EGFR")    # gene → UniProt
        details = await lookup.get_uniprot_details("P00533")  # accession → details
        valid = await lookup.verify_pdb_id("1M17")      # check PDB exists
    """

    UNIPROT_BASE = RetrievalConfig.UNIPROT_BASE_URL
    RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    RCSB_DOWNLOAD_URL = DockingConfig.PDB_DOWNLOAD_URL

    def __init__(self) -> None:
        """Initialise TargetLookup with configured timeout and headers."""
        self.timeout = aiohttp.ClientTimeout(total=RetrievalConfig.HTTP_TIMEOUT)
        self.headers = {
            "User-Agent": RetrievalConfig.USER_AGENT,
            "Accept": "application/json",
        }

    # ── Gene → UniProt ID lookup ──────────────────────────────────────────────
    async def get_uniprot_info(self, gene_name: str) -> Optional[dict]:
        """
        Look up a human protein by gene symbol and return basic UniProt info.

        Searches UniProt for reviewed (Swiss-Prot) human entries with
        the given gene name. Returns the top hit.

        Args:
            gene_name: HGNC gene symbol (e.g. "EGFR", "KRAS").

        Returns:
            dict | None: Basic protein info dict, or None if not found.
            Keys: uniprot_id, gene_name, protein_name, pdb_ids.
        """
        query = (
            f"gene_exact:{gene_name} AND organism_id:9606 AND reviewed:true"
        )
        url = (
            f"{self.UNIPROT_BASE}/search"
            f"?query={query}"
            f"&fields=accession,id,gene_names,protein_name,xref_pdb"
            f"&format=json&size=1"
        )

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.debug(
                            f"[TargetLookup] UniProt search HTTP {resp.status} "
                            f"for gene '{gene_name}'."
                        )
                        return None
                    data = await resp.json(content_type=None)

        except asyncio.TimeoutError:
            logger.warning(
                f"[TargetLookup] UniProt search timed out for '{gene_name}'."
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[TargetLookup] UniProt search error: {exc}")
            return None

        results = data.get("results", [])
        if not results:
            logger.debug(
                f"[TargetLookup] No UniProt entry found for gene '{gene_name}'."
            )
            return None

        entry = results[0]
        uniprot_id = entry.get("primaryAccession", "")
        protein_name = self._extract_protein_name(entry)
        pdb_ids = self._extract_pdb_ids(entry)

        logger.debug(
            f"[TargetLookup] {gene_name} → UniProt: {uniprot_id}, "
            f"PDB IDs: {pdb_ids[:5]}"
        )

        return {
            "uniprot_id": uniprot_id,
            "gene_name": gene_name,
            "protein_name": protein_name,
            "pdb_ids": pdb_ids,
        }

    # ── UniProt entry details ─────────────────────────────────────────────────
    async def get_uniprot_details(self, uniprot_id: str) -> Optional[dict]:
        """
        Retrieve detailed UniProt entry for a protein accession.

        Fetches binding site annotations, active site residues,
        and associated PDB structure IDs.

        Args:
            uniprot_id: UniProt accession number (e.g. "P00533").

        Returns:
            dict | None: Detailed protein info, or None on failure.
            Keys: uniprot_id, active_site, binding_sites, pdb_ids.
        """
        url = (
            f"{self.UNIPROT_BASE}/{uniprot_id}"
            f"?fields=accession,ft_act_site,ft_binding,xref_pdb,sequence"
            f"&format=json"
        )

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
            ) as session:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        logger.warning(
                            f"[TargetLookup] UniProt ID '{uniprot_id}' not found."
                        )
                        return None
                    if resp.status != 200:
                        logger.debug(
                            f"[TargetLookup] UniProt detail HTTP {resp.status} "
                            f"for '{uniprot_id}'."
                        )
                        return None
                    data = await resp.json(content_type=None)

        except asyncio.TimeoutError:
            logger.warning(
                f"[TargetLookup] UniProt details timed out for '{uniprot_id}'."
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[TargetLookup] UniProt details error: {exc}")
            return None

        # ── Parse feature annotations ─────────────────────────────────────────
        features = data.get("features", [])
        active_site = self._parse_active_site(features)
        binding_sites = self._parse_binding_sites(features)
        pdb_ids = self._extract_pdb_ids(data)

        logger.debug(
            f"[TargetLookup] {uniprot_id}: "
            f"active_site={active_site}, "
            f"binding_sites={binding_sites[:3]}, "
            f"pdb_ids={pdb_ids[:5]}"
        )

        return {
            "uniprot_id": uniprot_id,
            "active_site": active_site,
            "binding_sites": binding_sites,
            "pdb_ids": pdb_ids,
        }

    # ── PDB ID verification ───────────────────────────────────────────────────
    async def verify_pdb_id(self, pdb_id: str) -> bool:
        """
        Check whether a PDB ID exists and is accessible from RCSB.

        Args:
            pdb_id: 4-character PDB identifier (e.g. "1M17").

        Returns:
            bool: True if the PDB entry is accessible (HTTP 200).
        """
        url = self.RCSB_ENTRY_URL.format(pdb_id=pdb_id.upper())

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    is_valid = resp.status == 200
                    logger.debug(
                        f"[TargetLookup] PDB {pdb_id}: "
                        f"{'valid' if is_valid else 'invalid'} (HTTP {resp.status})"
                    )
                    return is_valid
        except Exception:  # noqa: BLE001
            return False

    # ── PDB file download ─────────────────────────────────────────────────────
    async def download_pdb_file(
        self,
        pdb_id: str,
        output_dir: str = "data/pdb",
    ) -> Optional[str]:
        """
        Download a PDB file from RCSB and save it locally.

        Skips download if the file already exists (caching).

        Args:
            pdb_id: 4-character PDB identifier.
            output_dir: Directory to save the PDB file.

        Returns:
            str | None: Path to the downloaded PDB file, or None on failure.
        """
        import os
        from pathlib import Path

        pdb_id = pdb_id.upper()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = out_dir / f"{pdb_id}.pdb"

        # Return cached file if it exists
        if pdb_path.exists() and pdb_path.stat().st_size > 0:
            logger.debug(f"[TargetLookup] PDB {pdb_id} cached at {pdb_path}.")
            return str(pdb_path)

        url = self.RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id)

        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"[TargetLookup] PDB download failed: "
                            f"HTTP {resp.status} for {pdb_id}."
                        )
                        return None
                    content = await resp.read()
                    pdb_path.write_bytes(content)
                    logger.info(
                        f"[TargetLookup] PDB {pdb_id} downloaded → {pdb_path} "
                        f"({len(content) // 1024} KB)"
                    )
                    return str(pdb_path)

        except asyncio.TimeoutError:
            logger.warning(f"[TargetLookup] PDB {pdb_id} download timed out.")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[TargetLookup] PDB {pdb_id} download error: {exc}")
            return None

    # ── Private parsing helpers ───────────────────────────────────────────────
    @staticmethod
    def _extract_protein_name(entry: dict) -> str:
        """
        Extract the recommended protein name from a UniProt entry dict.

        Args:
            entry: UniProt entry JSON dict.

        Returns:
            str: Protein name string, or empty string if not found.
        """
        try:
            prot_desc = entry.get("proteinDescription", {})
            rec_name = prot_desc.get("recommendedName", {})
            full_name = rec_name.get("fullName", {})
            return full_name.get("value", "")
        except (AttributeError, TypeError):
            return ""

    @staticmethod
    def _extract_pdb_ids(entry: dict) -> list[str]:
        """
        Extract PDB cross-reference IDs from a UniProt entry dict.

        Looks in both 'uniProtKBCrossReferences' (full entry) and
        'xref_pdb' fields (search result format).

        Args:
            entry: UniProt entry JSON dict.

        Returns:
            list[str]: List of uppercase PDB ID strings.
        """
        pdb_ids: list[str] = []

        # Full entry format: uniProtKBCrossReferences
        xrefs = entry.get("uniProtKBCrossReferences", [])
        for xref in xrefs:
            if isinstance(xref, dict) and xref.get("database") == "PDB":
                pdb_id = xref.get("id", "")
                if pdb_id:
                    pdb_ids.append(pdb_id.upper())

        # Search result format: may be in a nested structure
        if not pdb_ids:
            xref_pdb = entry.get("xref_pdb", "")
            if xref_pdb:
                ids = [
                    x.strip().upper()
                    for x in re.split(r"[,;\s]+", xref_pdb)
                    if len(x.strip()) == 4
                ]
                pdb_ids.extend(ids)

        return pdb_ids

    @staticmethod
    def _parse_active_site(features: list[dict]) -> str:
        """
        Extract active site annotation from UniProt feature list.

        Args:
            features: List of feature dicts from UniProt entry.

        Returns:
            str: Active site description, or "Not annotated".
        """
        active_sites: list[str] = []
        for feature in features:
            if feature.get("type") == "Active site":
                loc = feature.get("location", {})
                pos = loc.get("start", {}).get("value", "?")
                desc = feature.get("description", "")
                active_sites.append(f"Position {pos}: {desc}")

        return "; ".join(active_sites) if active_sites else "Not annotated"

    @staticmethod
    def _parse_binding_sites(features: list[dict]) -> list[str]:
        """
        Extract binding site residue annotations from UniProt feature list.

        Args:
            features: List of feature dicts from UniProt entry.

        Returns:
            list[str]: Binding site descriptions (max 10).
        """
        binding_sites: list[str] = []
        for feature in features:
            if feature.get("type") in {"Binding site", "Mutagenesis"}:
                loc = feature.get("location", {})
                start = loc.get("start", {}).get("value", "?")
                end = loc.get("end", {}).get("value", "?")
                desc = feature.get("description", "")
                ligand = ""
                if feature.get("ligand"):
                    ligand = feature["ligand"].get("name", "")

                site_str = f"Pos {start}"
                if start != end:
                    site_str += f"-{end}"
                if ligand:
                    site_str += f" (ligand: {ligand})"
                if desc:
                    site_str += f": {desc}"

                binding_sites.append(site_str)

        return binding_sites[:10]