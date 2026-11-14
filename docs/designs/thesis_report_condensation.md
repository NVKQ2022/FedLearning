# RFC: Undergraduate Thesis Condensation & Rigor Hardening Strategy

* **Author:** NVKQ2022 (Nguyen Viet Ky Quan -- 23521267)
* **Date:** 2026-11-14
* **Academic Context:** Undergraduate Thesis Defense / Technical Report (UIT, VNU-HCM)
* **Target Artifact:** `thesis_report/main.pdf` (reducing from 34 pages to ~20–21 pages)
* **Status:** Published for Team & Committee Review

---

## 1. Overview & Problem Statement

The undergraduate thesis report (`thesis_report/main.pdf`) currently spans 34 pages. While experimentally comprehensive across 23 controlled scenarios on the CICIoT2023 dataset, the manuscript suffers from three structural flaws:
1. **Unverified Algorithmic Proposals**: The Abstract and Chapter 3 mention algorithmic proposals (Decoupled Federated Learning / FedPer / FedRep) without corresponding empirical validation in Chapter 5. In an academic thesis defense, this invites severe criticism regarding unproven claims.
2. **Structural & Frontmatter Bloat**: Frontmatter consumes 7 full pages due to isolated page breaks for Abstract, Acknowledgments, Table of Contents, List of Figures, List of Tables, and Acronyms. Figures with rigid `[H]` anchors force artificial vertical voids and push text across page breaks.
3. **Data Redundancies**: Appendices A and B consume 2 full pages simply repeating dataset statistics and hyperparameter tables already present in Chapters 2, 3, and 5.

This RFC outlines the strategy to condense the manuscript into a dense, publication-grade **~20–21 page monograph**, establishing a pure **Empirical Counter-Evidence and Benchmarking Study**.

---

## 2. Multi-Lens Analysis & Strategic Decisions

### 2.1 Academic Positioning (Negative Finding & Empirical Exposé)
* Re-anchor the manuscript strictly as an **Empirical Benchmarking Study**:
  * 23 rigorously controlled scenarios on 5,116,391 network flows of CICIoT2023.
  * Zero-data-leakage global holdout evaluation (1,534,918 test samples).
  * Discovery of the **"Minority Extinction Cliff"** at $\mu \ge 0.01$: proving mathematically and empirically that proximal regularization annihilates minority attack detection (Brute-force drops to 0.00%) even under homogeneous IID conditions.
  * Discovery of the **Universal Pareto Sweet Spot** at $\mu = 0.001$: achieving 78.7%–90.8% parameter drift reduction across all 4 Dirichlet regimes (IID, $\alpha=1.0$, $\alpha=0.5$, $\alpha=0.1$) while preserving minority plasticity.
* **Decoupled FL Relocation**: Completely purge Decoupled FL from the Abstract, Chapter 1 contributions, and Chapter 3. Relocate it strictly to Chapter 6 (Section 6.2, Future Work) as a mathematically derived hypothesis born directly from the empirical failure of global classifier aggregation.

### 2.2 Page Budget Allocation

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

## 3. Goals vs. Non-Goals

### In-Scope Goals
* **Purity of Empirical Positioning**: Eliminate speculative algorithms from Abstract and Chapter 3; retain Decoupled FL strictly as an empirically motivated future-work hypothesis in Chapter 6.
* **Document Compression to ~20 Pages**: Strip frontmatter bloat (LoF/LoT), remove redundant appendices, eliminate forced page breaks, and tighten float geometry.
* **Preservation of Scientific Substance**: Keep the complete 23-scenario benchmark matrix, all 4 empirical findings, and the multi-regime Pareto figure intact.

### Out-of-Scope Non-Goals
* **No New Model Training**: No additional seeds or hyperparameter evaluations are required.
* **No Code Refactoring**: Modifications are strictly confined to the LaTeX report repository.

---

## 4. Execution Plan

1. **Frontmatter Cleanup (`00_frontmatter/abstract.tex`)**:
   - Update scenario count from 10 to 23.
   - Remove references to Decoupled FL from the abstract text and keyword list.
   - Mention the universal Pareto sweet spot at $\mu = 0.001$ ($57.42\%$ Macro-F1 at $\alpha=0.1$).
2. **Introduction Alignment (`01_introduction/introduction.tex`)**:
   - Reiterate the 23-scenario benchmark matrix.
   - Streamline the engineering disclaimer.
3. **Methodology Solidification (`03_methodology/methodology.tex`)**:
   - Delete all references to `proposed_solutions.tex`.
4. **Results & Float Geometry (`05_results_discussions/results_discussions.tex`)**:
   - Remove unnatural `\clearpage`.
   - Ensure Master Table 5.1 is compactly formatted with `\renewcommand{\arraystretch}{0.95}`.
   - Replace rigid `[H]` float specifiers with `[htbp]`.
5. **Conclusions & Future Architecture (`06_conclusions/conclusions.tex`)**:
   - In Section 6.2, present the formal mathematical formulation of Decoupled Federated Learning (FedPer/FedRep): localizing $\mathbf{w}_{\text{head}}^{(k)}$ while sharing $\mathbf{w}_{\text{base}}$ to resolve the minority extinction cliff.
6. **Main Document Streamlining (`main.tex`)**:
   - Comment out `\listoffigures`, `\listoftables`, and appendices `appendix_a.tex`, `appendix_b.tex`.
   - Compile via `./build_report.sh` and verify page count reaches ~20–21 pages.
