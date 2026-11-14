# Undergraduate Thesis Condensation & Rigor Hardening Strategy

* **Author:** NVKQ2022 (Nguyen Viet Ky Quan -- 23521267)
* **Date:** 2026-11-14
* **Academic Context:** Undergraduate Thesis Defense / Technical Report (UIT, VNU-HCM)
* **Target Artifact:** `thesis_report/main.pdf` (reducing from 34 pages to ~20–21 pages)

---

## Overview

This strategy condenses the undergraduate thesis report from 34 pages down to a dense, publication-grade ~20-page monograph. It eliminates unverified algorithmic proposals from the Abstract and Chapter 3, relocates Decoupled Federated Learning (FedPer/FedRep) strictly into Chapter 6 (Future Work) as an empirically grounded hypothesis, and centers 100% of the narrative on the empirical 23-scenario benchmark matrix, the minority extinction cliff, and the multi-regime Pareto frontier.

---

## Multi-Lens Analysis & Key Findings

### 1. Structural & Float Layout Inefficiencies (`egm-design`, `egm-codebase-details`)
* **Frontmatter Bloat**: Frontmatter currently spans 7 pages due to isolated page breaks for Abstract, Acknowledgments, Table of Contents, List of Figures, List of Tables, and Acronyms. Over 60% of this space is whitespace.
* **Float Rigidness**: Figures using `[H]` from the `float` package create massive page breaks and empty vertical gaps when they fail to fit on a single page alongside preceding text.
* **Redundant Pages**:
  * List of Figures and List of Tables consume 2 full pages and can be omitted or combined for a 20-page monograph.
  * Appendices A and B consume 2 full pages while simply repeating dataset distribution and hyperparameter tables already present in Chapters 2, 3, and 5.
  * Explicit `\clearpage` in Chapter 5 forces an unnecessary page break after a 14-line section.

### 2. Academic Credibility & Defense Vulnerability (`egm-zoom-out`, `egm-comprehension`)
* **The "Unverified Proposal" Trap**: Proposing Decoupled FL in the Abstract and Chapter 3 without empirical ablation or implementation invites severe committee pushback ("Why propose what was not tested?").
* **The Negative Finding High Ground**: Presenting the work strictly as an **Empirical Counter-Evidence and Benchmarking Study** backed by 23 scenarios, 5.1M flows, and 1.53M test samples provides bulletproof defense credibility.
* **Narrative Arc**:
  1. *Problem*: IoT high-volume vs. rare attacks under non-IID data.
  2. *Empirical Setup*: 23 controlled scenarios across 4 Dirichlet regimes (IID, $\alpha=1.0$, $\alpha=0.5$, $\alpha=0.1$).
  3. *Core Discovery*: FedAvg dilutes rare attack gradients ($n_k/N$ weighting); FedProx induces a "Minority Extinction Cliff" at $\mu \ge 0.01$.
  4. *Pareto Solution*: $\mu = 0.001$ suppresses drift by 78.7%–90.8% while preserving minority plasticity.
  5. *System Positioning*: FL-IoT-IDS operates at Layer 3/4; Layer 7 belongs to WAF.
  6. *Future Direction*: Decoupled FL is mathematically required to personalize classification heads.

### 3. Page Budget Allocation (`egm-implementation-readiness`)

| Section | Current Pages | Target Budget | Key Actions |
| :--- | :---: | :---: | :--- |
| **Frontmatter** (Cover, Abstract, ToC, Acronyms) | 7.0 | **2.5** | Omit LoF/LoT; merge Acronyms; clean Abstract text. |
| **Chapter 1: Introduction** | 3.0 | **2.0** | Tighten engineering disclaimer; focus on 3 empirical contributions. |
| **Chapter 2: Literature Review** | 4.5 | **3.0** | Condense dataset description; shrink Related Works matrix Table 2.1 to 0.5 page. |
| **Chapter 3: Methodology** | 3.5 | **2.5** | Remove `proposed_solutions.tex`; focus on 4-step controlled benchmark pipeline. |
| **Chapter 4: Implementation** | 2.5 | **1.5** | Remove redundant disclaimer; condense gRPC testbed and training loop prose. |
| **Chapter 5: Results & Discussions** | 8.5 | **5.5** | Keep Master Table 5.1 (23 scenarios); retain hero Pareto Figure 5.5; consolidate Figs 5.1 & 5.2. |
| **Chapter 6: Conclusions & Future Work** | 1.5 | **1.5** | Expand Section 6.2 with Decoupled FL theoretical formulation. |
| **References** | 1.5 | **1.5** | Retain complete IEEE references. |
| **Appendices A & B** | 2.0 | **0.0** | Omit; absorb essential data into Ch 2 and Ch 5. |
| **Total Page Count** | **34.0** | **~20.0–21.0** | **Net Reduction: -13 to -14 pages (~40%)** |

---

## Goals vs. Non-Goals

### In-Scope Goals
* **Purity of Empirical Positioning**: Eliminate all speculative algorithms from Abstract and Chapter 3; retain Decoupled FL strictly as a theoretically motivated future-work hypothesis in Chapter 6.
* **Document Compression to ~20 Pages**: Strip frontmatter bloat (LoF/LoT), remove redundant appendices, eliminate forced page breaks, and tighten float geometry.
* **Preservation of Scientific Substance**: Keep the complete 23-scenario benchmark matrix, all 4 empirical findings, and the multi-regime Pareto figure intact.

### Out-of-Scope Non-Goals
* **No New Model Training**: Do not run additional seeds or hyperparameter points.
* **No Code Refactoring**: Do not modify Flower runner scripts or pipeline code; modifications are strictly confined to the LaTeX report repository.

---

## Key Tactical Decisions

1. **Purge Decoupled FL from Frontmatter and Chapter 3**: Delete lines mentioning Decoupled FL from `abstract.tex`, `introduction.tex`, and `methodology.tex`. Delete `proposed_solutions.tex`.
2. **Expand Chapter 6.2 with Mathematical Hypothesis**: Formally articulate why localizing the classification head $\mathbf{w}_{\text{head}}^{(k)}$ resolves the minority extinction cliff identified in Finding 1.
3. **Condense LaTeX Geometry**: Use `titlesec` to reduce chapter top margins, omit LoF/LoT in `main.tex`, and switch figure placements from rigid `[H]` to `[htbp]`.
