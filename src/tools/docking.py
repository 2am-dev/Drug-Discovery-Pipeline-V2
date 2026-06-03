"""
tools/docking.py — AutoDock Vina subprocess wrapper and receptor preparation.

Provides:
  1. DockingTool.download_pdb()   — Fetch PDB from RCSB (delegates to TargetLookup).
  2. DockingTool.estimate_binding_box() — Calculate docking search box from residues.
  3. DockingTool.prepare_receptor() — Convert PDB → PDBQT with pdbfixer/meeko.
  4. DockingTool.prepare_ligand()  — Convert SMILES → PDBQT with RDKit/meeko.
  5. DockingTool.dock_molecule_sync() — Run Vina subprocess (blocking, for ThreadPool).
  6. DockingTool.parse_vina_output() — Extract best affinity from Vina log.

Tool dependencies (graceful degradation if missing):
  - AutoDock Vina binary (required for real docking).
  - meeko >= 0.5 (ligand/receptor PDBQT preparation).
  - pdbfixer (receptor cleaning, optional — raw PDB used as fallback).
  - RDKit (SMILES → 3D conformer generation).

If any dependency is missing, dock_molecule_sync() returns None and the
DockingEvaluatorAgent falls back to mock scoring.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from config import DockingConfig
from schemas import DockingResult


class DockingTool:
    """
    Wraps AutoDock Vina for molecular docking of SMILES against a PDB receptor.

    All I/O-bound operations (PDB download, PDBQT preparation) are async.
    The Vina subprocess call is synchronous and designed to run inside
    a ThreadPoolExecutor in the DockingEvaluatorAgent.

    Usage:
        tool = DockingTool()
        pdb_path = await tool.download_pdb("1M17")
        centre = await tool.estimate_binding_box(pdb_path, ["M793", "L718"])
        result = tool.dock_molecule_sync(smiles, pdb_path, centre, index=1)
    """

    def __init__(self) -> None:
        """Initialise DockingTool with configured paths."""
        self.vina_binary = DockingConfig.VINA_BINARY
        self.output_dir = Path("outputs/poses")
        self.pdb_cache_dir = Path("data/pdb")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── PDB download ──────────────────────────────────────────────────────────
    async def download_pdb(self, pdb_id: str) -> Optional[str]:
        """
        Download a PDB structure file from RCSB.

        Delegates to TargetLookup.download_pdb_file() which handles
        caching and async HTTP retrieval.

        Args:
            pdb_id: 4-character PDB identifier (case-insensitive).

        Returns:
            str | None: Local file path to the downloaded PDB file,
                        or None if download failed.
        """
        from tools.target_lookup import TargetLookup
        lookup = TargetLookup()
        return await lookup.download_pdb_file(
            pdb_id=pdb_id,
            output_dir=str(self.pdb_cache_dir),
        )

    # ── Binding box estimation ────────────────────────────────────────────────
    async def estimate_binding_box(
        self,
        pdb_path: str,
        residues: list[str],
    ) -> dict:
        """
        Estimate the Vina docking search box centred on binding site residues.

        Parses the PDB ATOM records to find Cα (CA) coordinates for the
        specified residues and calculates the centroid. Falls back to the
        protein's geometric centre if residues are not found.

        Args:
            pdb_path: Path to the receptor PDB file.
            residues: List of residue strings (e.g. ["M793", "L718", "V726"]).
                      Format: single-letter AA + residue number.

        Returns:
            dict: Docking box parameters:
                  {"x": float, "y": float, "z": float,
                   "size_x": int, "size_y": int, "size_z": int}
        """
        # Parse residue numbers from the strings (e.g. "M793" → 793)
        residue_numbers: set[str] = set()
        for res in residues:
            match = re.search(r"\d+", res)
            if match:
                residue_numbers.add(match.group())

        ca_coords: list[tuple[float, float, float]] = []

        try:
            with open(pdb_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.startswith(("ATOM", "HETATM")):
                        continue
                    atom_name = line[12:16].strip()
                    res_seq = line[22:26].strip()

                    if atom_name == "CA" and res_seq in residue_numbers:
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            ca_coords.append((x, y, z))
                        except ValueError:
                            continue
        except OSError as exc:
            logger.warning(f"[DockingTool] Cannot read PDB {pdb_path}: {exc}")

        if ca_coords:
            # Centroid of binding site Cα atoms
            cx = sum(c[0] for c in ca_coords) / len(ca_coords)
            cy = sum(c[1] for c in ca_coords) / len(ca_coords)
            cz = sum(c[2] for c in ca_coords) / len(ca_coords)
            logger.debug(
                f"[DockingTool] Binding box centred from "
                f"{len(ca_coords)} residues: ({cx:.2f}, {cy:.2f}, {cz:.2f})"
            )
        else:
            # Fallback: use protein geometric centre
            logger.warning(
                "[DockingTool] Binding site residues not found in PDB. "
                "Using protein geometric centre."
            )
            cx, cy, cz = await self._protein_centre(pdb_path)

        size = DockingConfig.SEARCH_BOX_SIZE
        return {
            "x": round(cx, 3),
            "y": round(cy, 3),
            "z": round(cz, 3),
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
        }

    async def _protein_centre(self, pdb_path: str) -> tuple[float, float, float]:
        """
        Calculate the geometric centre of all Cα atoms in a PDB file.

        Args:
            pdb_path: Path to the PDB file.

        Returns:
            tuple[float, float, float]: (x, y, z) centroid coordinates.
        """
        all_ca: list[tuple[float, float, float]] = []
        try:
            with open(pdb_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            all_ca.append((x, y, z))
                        except ValueError:
                            continue
        except OSError:
            pass

        if all_ca:
            return (
                sum(c[0] for c in all_ca) / len(all_ca),
                sum(c[1] for c in all_ca) / len(all_ca),
                sum(c[2] for c in all_ca) / len(all_ca),
            )
        return (0.0, 0.0, 0.0)

    # ── Receptor preparation ──────────────────────────────────────────────────
    def prepare_receptor_pdbqt(self, pdb_path: str) -> Optional[str]:
        """
        Convert a PDB receptor file to PDBQT format for Vina.

        Attempts meeko (preferred) then falls back to a minimal
        hydrogen-stripping approach if meeko is not installed.

        Args:
            pdb_path: Path to the input PDB file.

        Returns:
            str | None: Path to the output PDBQT file, or None on failure.
        """
        pdbqt_path = Path(pdb_path).with_suffix(".pdbqt")

        # Return cached file if it exists
        if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
            return str(pdbqt_path)

        # Try MGLTools prepare_receptor4.py (if available)
        try:
            result = subprocess.run(
                [
                    "prepare_receptor4.py",
                    "-r", pdb_path,
                    "-o", str(pdbqt_path),
                    "-A", "hydrogens",
                    "-U", "nphs_lps_waters_nonstdres",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and pdbqt_path.exists():
                logger.debug(
                    f"[DockingTool] Receptor PDBQT prepared: {pdbqt_path}"
                )
                return str(pdbqt_path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try Open Babel as fallback
        try:
            result = subprocess.run(
                ["obabel", pdb_path, "-O", str(pdbqt_path), "-xr"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and pdbqt_path.exists():
                logger.debug(
                    f"[DockingTool] Receptor PDBQT prepared via obabel: {pdbqt_path}"
                )
                return str(pdbqt_path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        logger.warning(
            "[DockingTool] Neither prepare_receptor4.py nor obabel found. "
            "Attempting to use raw PDB as PDBQT (may fail in Vina)."
        )
        # Minimal fallback: copy PDB to PDBQT (works for simple cases)
        import shutil
        shutil.copy2(pdb_path, str(pdbqt_path))
        return str(pdbqt_path)

    # ── Ligand preparation ────────────────────────────────────────────────────
    def prepare_ligand_pdbqt(
        self,
        smiles: str,
        ligand_index: int,
    ) -> Optional[str]:
        """
        Convert a SMILES string to a Vina-ready PDBQT file.

        Steps:
          1. Generate a 3D conformer with RDKit (ETKDG).
          2. Minimise with MMFF94 force field.
          3. Convert to PDBQT with meeko (preferred) or obabel (fallback).

        Args:
            smiles: Valid SMILES string.
            ligand_index: Integer index used for output filename uniqueness.

        Returns:
            str | None: Path to the ligand PDBQT file, or None on failure.
        """
        ligand_sdf = self.output_dir / f"ligand_{ligand_index:03d}.sdf"
        ligand_pdbqt = self.output_dir / f"ligand_{ligand_index:03d}.pdbqt"

        if ligand_pdbqt.exists() and ligand_pdbqt.stat().st_size > 0:
            return str(ligand_pdbqt)

        # ── Step 1: Generate 3D conformer with RDKit ──────────────────────────
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(
                    f"[DockingTool] Invalid SMILES for ligand {ligand_index}: "
                    f"{smiles[:50]}"
                )
                return None

            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            ret = AllChem.EmbedMolecule(mol, params)

            if ret == -1:
                logger.warning(
                    f"[DockingTool] 3D embedding failed for ligand {ligand_index}."
                )
                return None

            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

            writer = Chem.SDWriter(str(ligand_sdf))
            writer.write(mol)
            writer.close()

        except ImportError:
            logger.error(
                "[DockingTool] RDKit not available for 3D conformer generation."
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[DockingTool] RDKit conformer error for ligand {ligand_index}: {exc}"
            )
            return None

        # ── Step 2: Convert SDF → PDBQT with meeko ───────────────────────────
        try:
            import meeko
            mk_prep = meeko.MoleculePreparation()
            mol_setups = mk_prep.prepare(
                Chem.MolFromMolFile(str(ligand_sdf), removeHs=False)
            )
            for setup in mol_setups:
                pdbqt_string = meeko.PDBQTWriterLegacy.write_string(setup)
                ligand_pdbqt.write_text(pdbqt_string)
            logger.debug(
                f"[DockingTool] Ligand PDBQT prepared with meeko: {ligand_pdbqt}"
            )
            return str(ligand_pdbqt)
        except ImportError:
            pass   # Try obabel fallback
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[DockingTool] meeko failed: {exc}")

        # ── Step 3: Fallback to Open Babel ────────────────────────────────────
        try:
            result = subprocess.run(
                ["obabel", str(ligand_sdf), "-O", str(ligand_pdbqt)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and ligand_pdbqt.exists():
                logger.debug(
                    f"[DockingTool] Ligand PDBQT prepared via obabel: {ligand_pdbqt}"
                )
                return str(ligand_pdbqt)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        logger.warning(
            f"[DockingTool] Could not prepare PDBQT for ligand {ligand_index}. "
            "Neither meeko nor obabel available."
        )
        return None

    # ── Vina subprocess (synchronous — runs in ThreadPool) ────────────────────
    def dock_molecule_sync(
        self,
        smiles: str,
        pdb_path: str,
        box_centre: dict,
        ligand_index: int,
    ) -> Optional[DockingResult]:
        """
        Run AutoDock Vina for a single ligand–receptor pair.

        Designed to run inside a ThreadPoolExecutor (blocking subprocess).
        Prepares receptor and ligand PDBQT files, runs Vina, parses output.

        Args:
            smiles: Ligand SMILES string.
            pdb_path: Path to the receptor PDB file.
            box_centre: Dict with keys x, y, z, size_x, size_y, size_z.
            ligand_index: Integer index for unique output file naming.

        Returns:
            DockingResult | None: Parsed result, or None if docking failed.
        """
        logger.debug(
            f"[DockingTool] Docking ligand {ligand_index}: "
            f"{smiles[:40]} …"
        )

        # ── Prepare receptor PDBQT ────────────────────────────────────────────
        receptor_pdbqt = self.prepare_receptor_pdbqt(pdb_path)
        if not receptor_pdbqt:
            logger.warning(
                f"[DockingTool] Receptor prep failed for ligand {ligand_index}."
            )
            return None

        # ── Prepare ligand PDBQT ──────────────────────────────────────────────
        ligand_pdbqt = self.prepare_ligand_pdbqt(smiles, ligand_index)
        if not ligand_pdbqt:
            logger.warning(
                f"[DockingTool] Ligand prep failed for index {ligand_index}."
            )
            return None

        # ── Output files ──────────────────────────────────────────────────────
        out_pdbqt = self.output_dir / f"pose_{ligand_index:03d}.pdbqt"
        log_file  = self.output_dir / f"vina_{ligand_index:03d}.log"

        # ── Build Vina command ────────────────────────────────────────────────
        cmd = [
            self.vina_binary,
            "--receptor", receptor_pdbqt,
            "--ligand",   ligand_pdbqt,
            "--center_x", str(box_centre["x"]),
            "--center_y", str(box_centre["y"]),
            "--center_z", str(box_centre["z"]),
            "--size_x",   str(box_centre["size_x"]),
            "--size_y",   str(box_centre["size_y"]),
            "--size_z",   str(box_centre["size_z"]),
            "--out",      str(out_pdbqt),
            "--log",      str(log_file),
            "--exhaustiveness", str(DockingConfig.EXHAUSTIVENESS),
            "--num_modes",      str(DockingConfig.NUM_POSES),
            "--cpu", "2",
        ]

        # ── Run Vina ──────────────────────────────────────────────────────────
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,   # 5 minutes per ligand
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"[DockingTool] Vina timed out for ligand {ligand_index}."
            )
            return None
        except FileNotFoundError:
            logger.error(
                f"[DockingTool] Vina binary '{self.vina_binary}' not found."
            )
            return None

        if result.returncode != 0:
            logger.warning(
                f"[DockingTool] Vina returned non-zero exit code for "
                f"ligand {ligand_index}: {result.stderr[:200]}"
            )
            return None

        # ── Parse affinity from log ───────────────────────────────────────────
        log_path = str(log_file)
        affinity = self.parse_vina_log(
            log_file=log_path if Path(log_path).exists() else None,
            stdout=result.stdout,
        )

        if affinity is None:
            logger.warning(
                f"[DockingTool] Could not parse affinity from Vina output "
                f"for ligand {ligand_index}."
            )
            return None

        logger.debug(
            f"[DockingTool] Ligand {ligand_index}: affinity = {affinity} kcal/mol"
        )

        return DockingResult(
            smiles=smiles,
            binding_affinity_kcal_mol=affinity,
            ligand_efficiency=0.0,   # calculated by caller (needs heavy atom count)
            pose_file=str(out_pdbqt) if out_pdbqt.exists() else None,
            key_interactions=[],     # populated by LLM interaction analysis
            binding_mode_summary=None,
            rank=ligand_index,
            is_mock=False,
        )

    # ── Vina output parsing ───────────────────────────────────────────────────
    @staticmethod
    def parse_vina_log(
        log_file: Optional[str],
        stdout: str = "",
    ) -> Optional[float]:
        """
        Extract the best (most negative) binding affinity from Vina output.

        Vina output table format:
            mode |   affinity | dist from best mode
                 | (kcal/mol) | rmsd l.b.| rmsd u.b.
            -----+------------+----------+----------
               1 |       -8.4 |    0.000 |    0.000

        Args:
            log_file: Path to the Vina log file (may be None).
            stdout: Vina stdout string (fallback if log file unavailable).

        Returns:
            float | None: Best binding affinity (most negative), or None.
        """
        # Pattern: line starting with whitespace, then mode number, then affinity
        pattern = re.compile(
            r"^\s+\d+\s+(-?\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+",
            re.MULTILINE,
        )

        # Try log file first
        if log_file:
            try:
                text = Path(log_file).read_text(encoding="utf-8", errors="replace")
                matches = pattern.findall(text)
                if matches:
                    return float(matches[0])   # first match = best mode
            except OSError:
                pass

        # Try stdout
        matches = pattern.findall(stdout)
        if matches:
            return float(matches[0])

        # Alternative pattern: "Affinity: -8.4 kcal/mol"
        alt_pattern = re.compile(r"Affinity:\s*(-?\d+\.\d+)\s*kcal/mol")
        text_to_search = stdout
        if log_file:
            try:
                text_to_search += Path(log_file).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                pass
        alt_match = alt_pattern.search(text_to_search)
        if alt_match:
            return float(alt_match.group(1))

        return None