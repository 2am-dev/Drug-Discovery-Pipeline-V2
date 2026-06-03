"""
utils/prompts.py — JSON-enforced prompt templates for all pipeline agents.

IMPORTANT: These prompts are injected INTO the Python code as strings that
are sent to the LLM running inside Ollama. They are NOT instructions to the
LLM answering this meta-prompt. They are templates used at runtime.

Design principles
─────────────────
1. Every prompt ends with JSON_ENFORCEMENT — a strict JSON-only instruction
   block that suppresses markdown, prose, and code fences.
2. Prompts use Python f-string placeholders (e.g. {disease}) that are filled
   in at runtime by the calling agent.
3. All expected output structures are documented inline with field descriptions
   so the LLM understands exactly what each field should contain.
4. Prompts are intentionally concise to conserve the token budget.

Usage
─────
    from utils.prompts import build_agent_prompt, HYPOTHESIS_PROMPT

    filled_prompt = HYPOTHESIS_PROMPT.format(
        target_candidates=json.dumps(candidates),
        indication="non-small cell lung cancer",
    )
    messages = build_agent_prompt(system=SYSTEM_DRUG_EXPERT, user=filled_prompt)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# JSON enforcement block — appended to EVERY agent prompt
# ─────────────────────────────────────────────────────────────────────────────
JSON_ENFORCEMENT = """
CRITICAL INSTRUCTIONS — READ BEFORE RESPONDING:
1. Output ONLY valid JSON. Nothing else.
2. Do NOT wrap your response in markdown code blocks (no ```json or ```).
3. Do NOT add any explanatory text before or after the JSON object.
4. Do NOT use single quotes — use double quotes for all strings.
5. Do NOT include trailing commas in objects or arrays.
6. Ensure all string values have properly escaped special characters.
7. Your response must begin with { and end with }.
8. Validate the JSON structure in your reasoning before outputting it.
"""


# ─────────────────────────────────────────────────────────────────────────────
# System persona
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_DRUG_EXPERT = (
    "You are an expert computational drug discovery scientist with deep knowledge "
    "of medicinal chemistry, structural biology, pharmacology, and clinical "
    "development. You analyse scientific literature and data to generate "
    "actionable drug discovery hypotheses. You always respond with structured "
    "JSON as instructed."
)

SYSTEM_ANALYST = (
    "You are a senior biomedical data analyst specialising in literature mining, "
    "target identification, and evidence synthesis for early-stage drug discovery. "
    "You always respond with structured JSON as instructed."
)

SYSTEM_CHEMIST = (
    "You are an expert medicinal chemist with expertise in rational drug design, "
    "structure-activity relationships (SAR), and ADMET property optimisation. "
    "You always respond with structured JSON as instructed."
)

SYSTEM_REPORTER = (
    "You are a scientific writer specialising in drug discovery project proposals. "
    "You synthesise complex multi-agent pipeline outputs into clear, compelling "
    "proposals for scientific and business audiences. "
    "You always respond with structured JSON as instructed."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build OpenAI-format message list
# ─────────────────────────────────────────────────────────────────────────────
def build_agent_prompt(
    system: str,
    user: str,
    assistant_primer: str = "",
) -> list[dict[str, str]]:
    """
    Construct an OpenAI-format chat message list for an agent call.

    Args:
        system: System prompt string (persona + task framing).
        user: User message string (data + specific instructions).
        assistant_primer: Optional string to prime the assistant's response,
                          e.g. "{" to nudge the model to start with JSON.

    Returns:
        list[dict[str, str]]: Ready-to-send message list.

    Example:
        >>> msgs = build_agent_prompt(SYSTEM_DRUG_EXPERT, filled_hypothesis_prompt)
        >>> response = await client.chat(messages=msgs, schema=HypothesisResponse)
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if assistant_primer:
        messages.append({"role": "assistant", "content": assistant_primer})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Planner prompt
# ─────────────────────────────────────────────────────────────────────────────
PLANNER_PROMPT = """\
You are planning a drug discovery pipeline run.

INPUT:
  Disease / Target: {indication_or_target}
  Input type: {input_type}
  Enable synthesis: {enable_synthesis}
  Enable docking: {enable_docking}
  Enable patents: {enable_patents}

TASK:
Create a structured execution plan for the pipeline. Each phase must list
its dependencies. Set synthesis phase status to "skipped" if enable_synthesis
is false. Set docking phase status to "skipped" if enable_docking is false.

Estimate the total pipeline duration in minutes based on the enabled phases:
- retrieval: ~10 min
- hypothesis: ~5 min
- molecule_design: ~8 min
- docking: ~15 min (if enabled)
- synthesis: ~7 min (if enabled)
- report: ~5 min

OUTPUT: Return a JSON object matching this exact structure:
{{
  "task_id": "{task_id}",
  "disease_or_target": "<the indication or target>",
  "phases": [
    {{"phase": "retrieval", "status": "pending", "dependencies": []}},
    {{"phase": "hypothesis", "status": "pending", "dependencies": ["retrieval"]}},
    {{"phase": "molecule_design", "status": "pending", "dependencies": ["hypothesis"]}},
    {{"phase": "docking", "status": "pending or skipped", "dependencies": ["molecule_design"]}},
    {{"phase": "synthesis", "status": "pending or skipped", "dependencies": ["molecule_design"]}},
    {{"phase": "report", "status": "pending", "dependencies": ["docking"]}}
  ],
  "estimated_duration_minutes": <integer>,
  "pipeline_notes": "<any relevant planning notes>"
}}
{json_enforcement}"""


# ─────────────────────────────────────────────────────────────────────────────
# Retriever: batch summarisation prompt
# ─────────────────────────────────────────────────────────────────────────────
RETRIEVER_SUMMARISE_PROMPT = """\
You are summarising a batch of scientific abstracts to identify drug targets.

DISEASE / TARGET: {indication_or_target}
BATCH NUMBER: {batch_number} of {total_batches}

ABSTRACTS:
{abstracts_text}

TASK:
Analyse these abstracts and extract:
1. Any druggable protein targets mentioned.
2. Key mechanistic findings relevant to {indication_or_target}.
3. Supporting evidence quality (clinical / preclinical / in vitro).
4. Any PMID citations mentioned.

OUTPUT: Return a JSON object with this structure:
{{
  "batch_number": {batch_number},
  "targets_found": [
    {{
      "gene_name": "<HGNC gene symbol>",
      "evidence_summary": "<1-2 sentence mechanistic summary>",
      "evidence_type": "clinical|preclinical|in_vitro|review",
      "citations": ["PMID:XXXXX"]
    }}
  ],
  "key_findings": ["<finding 1>", "<finding 2>"],
  "papers_processed": <integer>
}}
{json_enforcement}"""


# ─────────────────────────────────────────────────────────────────────────────
# Retriever: target ranking prompt
# ─────────────────────────────────────────────────────────────────────────────
RETRIEVER_RANK_PROMPT = """\
You are a drug target identification expert.

DISEASE / TARGET: {indication_or_target}

COMPILED EVIDENCE FROM LITERATURE ({total_papers} papers, {total_patents} patents):
{evidence_summary}

TASK:
Rank the identified drug targets by their druggability and therapeutic relevance.
For each target, provide UniProt ID if known and relevant PDB structure IDs.

OUTPUT: Return a JSON object with this structure:
{{
  "target_candidates": [
    {{
      "gene_name": "<HGNC symbol>",
      "uniprot_id": "<UniProt accession or null>",
      "pdb_ids": ["<PDB ID>"],
      "evidence_summary": "<comprehensive 2-3 sentence summary>",
      "literature_citations": ["PMID:XXXXX"],
      "patent_count": <integer>,
      "druggability_score": <float 0.0-1.0>,
      "novelty_score": <float 0.0-1.0>
    }}
  ],
  "total_papers_reviewed": {total_papers},
  "total_patents_reviewed": {total_patents},
  "retrieval_timestamp": "{timestamp}"
}}
{json_enforcement}"""


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis prompt
# ─────────────────────────────────────────────────────────────────────────────
HYPOTHESIS_PROMPT = """\
You are formulating a drug discovery hypothesis.

DISEASE / TARGET: {indication_or_target}

TOP TARGET CANDIDATES (from literature mining):
{target_candidates_json}

UNIPROT DATA (if available):
{uniprot_data}

TASK:
1. Select the single BEST druggable target from the candidates.
2. Formulate a mechanistic hypothesis for how inhibiting/modulating this target
   could treat {indication_or_target}.
3. Identify the most relevant binding site residues for drug design.
4. Provide 2-3 alternative targets as backups.
5. Assign a confidence score (0.0-1.0) based on evidence strength.

OUTPUT: Return a JSON object with this structure:
{{
  "selected_target": {{
    "gene_name": "<HGNC symbol>",
    "uniprot_id": "<accession>",
    "pdb_id": "<best structure PDB ID>",
    "binding_site_residues": ["<residue e.g. Met793>"],
    "target_class": "<kinase|GPCR|protease|nuclear_receptor|other>",
    "disease_relevance": "<1 sentence>"
  }},
  "hypothesis": {{
    "mechanism": "<detailed mechanistic hypothesis, 2-4 sentences>",
    "rationale": "<evidence-based rationale citing key studies>",
    "therapeutic_modality": "<small_molecule|antibody|PROTAC|other>",
    "confidence_score": <float 0.0-1.0>
  }},
  "alternative_targets": [
    {{
      "gene_name": "<symbol>",
      "uniprot_id": "<accession or null>",
      "rationale": "<why this is a backup>"
    }}
  ]
}}
{json_enforcement}"""


# ─────────────────────────────────────────────────────────────────────────────
# Molecule design prompt
# ─────────────────────────────────────────────────────────────────────────────
MOLECULE_DESIGN_PROMPT = """\
You are a medicinal chemist designing novel drug candidates.

TARGET: {gene_name} ({uniprot_id})
BINDING SITE RESIDUES: {binding_site_residues}
DISEASE: {indication_or_target}
MECHANISM: {mechanism}
THERAPEUTIC MODALITY: small molecule

KNOWN REFERENCE COMPOUNDS (from literature):
{reference_compounds}

TASK:
Design {generation_count} novel small-molecule drug candidates as SMILES strings.
These should be chemically diverse, drug-like, and designed to interact with
the target's binding site. Use scaffold decoration, bioisosteric replacement,
and fragment merging strategies.

DRUG-LIKENESS FILTERS TO APPLY:
- Molecular weight: ≤ 500 Da
- LogP: ≤ 5.0
- H-bond donors: ≤ 5
- H-bond acceptors: ≤ 10
- Lipinski violations: ≤ 1
- QED score: ≥ 0.5

OUTPUT: Return a JSON object with this structure:
{{
  "generated_molecules": [
    {{
      "smiles": "<valid SMILES string>",
      "name": "<systematic or provisional name>",
      "generation_method": "scaffold_decoration|bioisostere|fragment_merge|de_novo",
      "design_rationale": "<1 sentence explaining the design choice>",
      "predicted_interactions": ["<interaction with binding site residue>"]
    }}
  ],
  "design_strategy": "<overall design strategy description>",
  "reference_scaffold": "<core scaffold SMILES if applicable or null>"
}}
{json_enforcement}"""


# ─────────────────────────────────────────────────────────────────────────────
# Docking analysis prompt (LLM interprets mock or real docking results)
# ─────────────────────────────────────────────────────────────────────────────
DOCKING_ANALYSIS_PROMPT = """\
You are a structural biologist analysing molecular docking results.

TARGET: {gene_name} ({pdb_id})
BINDING SITE RESIDUES: {binding_site_residues}

DOCKING RESULTS:
{docking_data_json}

TASK:
For each docked molecule, analyse the binding mode and predict key molecular
interactions (hydrogen bonds, hydrophobic contacts, pi-stacking, salt bridges).
Rank molecules by binding affinity and interaction quality.
Identify the lead compound with the best balance of affinity and drug-likeness.

OUTPUT: Return a JSON object with this structure:
{{
  "docking_results": [
    {{
      "smiles": "<SMILES>",
      "binding_affinity_kcal_mol": <float, negative = better>,
      "ligand_efficiency": <float, affinity / heavy_atom_count>,
      "key_interactions": ["<interaction description>"],
      "binding_mode_summary": "<1-2 sentence description>",
      "rank": <integer, 1 = best>
    }}
  ],
  "lead_compound_smiles": "<SMILES of best compound>",
  "lead_compound_rationale": "<why this compound was selected>",
  "receptor_pdb": "{pdb_id}",
  "docking_software": "{docking_software}"
}}
{json_enforcement}"""


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis evaluation prompt (OPTIONAL module)
# ─────────────────────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT = """\
You are a synthetic organic chemist evaluating drug candidate synthesisability.

MOLECULES TO EVALUATE:
{molecules_json}

TASK:
For each molecule, propose a practical synthetic route from commercially
available starting materials. Consider:
1. SA score (provided): lower = easier to synthesise.
2. Retrosynthetic disconnection points.
3. Common reagents and reaction conditions.
4. Estimated number of synthetic steps.

SA SCORE INTERPRETATION:
- 1.0-2.0: Very easy (commercially available or 1-2 steps)
- 2.0-3.0: Easy (3-5 steps, standard chemistry)
- 3.0-4.0: Moderate (5-8 steps, some specialist steps)
- 4.0-6.0: Difficult (>8 steps or specialist reagents)
- >6.0: Very difficult (not recommended)

OUTPUT: Return a JSON object with this structure:
{{
  "synthesis_routes": [
    {{
      "smiles": "<SMILES>",
      "sa_score": <float>,
      "estimated_steps": <integer>,
      "starting_materials": ["<material 1>", "<material 2>"],
      "key_reactions": ["<reaction type>"],
      "route_description": "<step-by-step synthesis outline>",
      "feasibility": "high|medium|low",
      "estimated_yield_percent": <integer or null>
    }}
  ],
  "synthesis_enabled": true,
  "recommended_candidate": "<SMILES of most synthetically accessible candidate>"
}}
{json_enforcement}"""


# ─────────────────────────────────────────────────────────────────────────────
# Report compilation prompts
# ─────────────────────────────────────────────────────────────────────────────
REPORT_SECTION_PROMPT = """\
You are writing a section of a drug discovery project proposal.

SECTION: {section_name}
DISEASE / TARGET: {indication_or_target}
PIPELINE DATA:
{section_data_json}

TASK:
Write a professional, scientific {section_name} section for a drug discovery
project proposal. The section should be:
- Factual and grounded in the provided data.
- Written for a mixed scientific and business audience.
- Approximately {target_word_count} words.
- Free of speculation beyond what the data supports.

OUTPUT: Return a JSON object with this structure:
{{
  "section_name": "{section_name}",
  "content": "<full section text in Markdown format>",
  "key_points": ["<bullet point 1>", "<bullet point 2>"],
  "word_count": <integer>
}}
{json_enforcement}"""

REPORT_EXECUTIVE_SUMMARY_PROMPT = """\
You are writing the executive summary of a drug discovery project proposal.

DISEASE: {indication_or_target}
SELECTED TARGET: {gene_name} ({uniprot_id})
LEAD COMPOUND SMILES: {lead_smiles}
LEAD COMPOUND BINDING AFFINITY: {binding_affinity} kcal/mol
PIPELINE COMPLETION TIME: {execution_time_minutes} minutes

KEY FINDINGS:
{key_findings_json}

TASK:
Write a compelling 200-300 word executive summary that:
1. States the unmet medical need.
2. Identifies the selected target and mechanistic hypothesis.
3. Summarises the lead compound's key properties.
4. Outlines the next development steps.

OUTPUT: Return a JSON object with this structure:
{{
  "executive_summary": "<200-300 word summary in Markdown>",
  "headline": "<one-line project title>",
  "development_stage": "Hit Identification",
  "target_indication": "{indication_or_target}"
}}
{json_enforcement}"""