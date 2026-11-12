# Empirical Benchmark of the Stability-Plasticity Dilemma in FL-IoT-IDS

* **Author:** Nguyễn Việt Kỳ Quân (NVKQ2022)
* **Date:** 2026-11-12

## Overview

This empirical study investigates the performance collapse of Federated Learning (FL) algorithms on minority attack classes under statistical heterogeneity in IoT Network Intrusion Detection Systems (IoT-NIDS). 

Using the CICIoT2023 benchmark (5.1 million network flows), the study deconstructs the consensus in recent literature (2022–2026) that Federated Proximal (FedProx) is strictly superior to Federated Averaging (FedAvg) under non-IID conditions. The core finding demonstrates that this reported superiority is an artifact of aggregate classification accuracy ($>98\%$), which is dominated by benign and DDoS traffic ($>90\%$). Under severe Dirichlet label skew ($\alpha=0.1$), FedProx suppresses parameter drift but over-regularizes local model updates. This proximal anchoring extinguishes the gradient signals of rare, high-consequence attacks (`Web-based` at $0.49\%$ and `Brute-force` at $0.26\%$), reducing their detection recall to $0.00\%$.

By completing the missing baseline (E5 FedAvg at $\alpha=0.1$), executing proximal sensitivity sweeps ($\mu \in \{0.001, 0.01, 0.05, 0.1\}$), and conducting multi-seed verification ($N=3$), this project maps the empirical Pareto frontier between Client Drift Stability ($\sigma_{loss}, \Delta w$) and Minority Detection Plasticity ($\text{Recall}_{minority}$).

## Analysis

### 1. Mathematical Mechanism of Minority Collapse
The local objective of FedProx on client $k$ is formulated as:
$$\min_{w} h_k(w; w_t) = \mathcal{L}_{task}(w) + \frac{\mu}{2} \|w - w_t\|_2^2$$
where $w_t$ is the global model received from the central server at round $t$, and $\mu$ is the proximal regularization parameter.

In an extreme non-IID partition ($\alpha=0.1$), a client that holds rare attack samples must take large gradient steps to adapt its decision boundary. However, the proximal penalty $\frac{\mu}{2}\|w - w_t\|_2^2$ acts as an elastic anchor pulling local weights back to $w_t$. Because minority samples generate weak gradient magnitudes relative to the proximal loss penalty, the optimization updates are dominated by the distance constraint. As $\mu$ increases, parameter drift decreases by $>95\%$, but the output layer linear weights for minority classes remain pinned near zero. The global model achieves convergence stability by sacrificing the plasticity required to identify stealthy intrusions.

### 2. Multi-Lens Critique & Systemic Risks
* **Confounding Variables (`egm-design`):** To prove that minority collapse is caused by the proximal operator rather than poor optimization, all training hyperparameters must be locked: identical architecture (`TabularIoTMLP`, $39 \to 128 \to 64 \to 8$, 14,024 parameters), learning rate ($\eta = 10^{-3}$), weight decay ($10^{-4}$), batch size ($128$), and local epochs ($E=2$).
* **Optimizer State Bleed (`egm-codebase-details`):** In PyTorch implementations of federated learning, reusing local AdamW optimizer instances across communication rounds carries stale momentum buffers ($m_t, v_t$). This creates trajectory momentum that distorts the proximal gradient. Local optimizers must be re-instantiated fresh at the start of each round when updated weights are loaded.
* **The Asymmetric Baseline Gap (`egm-codebase-details`):** The current experiment suite contains E5 FedProx at $\alpha=0.1$ ($\mu=0.05$), but lacks the counterpart E5 FedAvg run at $\alpha=0.1$ ($\mu=0.0$). This missing cell prevents claiming that proximal regularization is worse than standard averaging under extreme skew.
* **Cybersecurity Impact (`egm-zoom-out`):** In IoT networks, attackers use web application vulnerabilities and credential brute-forcing for initial perimeter access, using volumetric DDoS only for disruption. Deploying an IDS that achieves $99\%$ accuracy by classifying DDoS correctly while maintaining $0.00\%$ recall on initial-access vectors leaves the internal network completely unprotected.
* **Statistical Validity (`egm-implementation-readiness`):** Single-seed results ($seed=42$) are insufficient to eliminate the possibility of partition artifacts. Evaluating three distinct random seeds ($seed \in \{42, 101, 2024\}$) across the $\mu$ grid produces standard deviations that convert anecdotal observations into statistical laws.

### 3. Contradiction Resolution (The Inventive Perspective)
The fundamental technical contradiction in FedProx is:
* *Improving Parameter:* Global consensus convergence and drift suppression on majority classes.
* *Worsening Parameter:* Local feature representation plasticity on rare minority classes.

The Ideal Final Result (IFR) is an algorithm that bounds parameter drift on common flow representations while permitting unrestricted gradient movement on classification decision boundaries. While the present study focuses on diagnostic benchmarking, it outlines three concrete resolution blueprints for future work:
1. **Layer-Selective Proximal Regularization:** Restricting $\mu > 0$ to feature extractor layers ($W_1, W_2$) while setting $\mu = 0$ on the classification head.
2. **Entropy-Adaptive Proximal Penalty:** Scaling $\mu_k = \mu_0 \cdot \left(\frac{\mathcal{H}(P_k)}{\log_2 C}\right)^\gamma$ based on local client label entropy.
3. **Annealed Proximal Schedule:** Initializing with high $\mu$ for consensus formation, followed by exponential decay to enable minority feature acquisition.

## Goals vs. Non-Goals

### In-Scope Goals
* **Complete E5 Head-to-Head Comparison:** Execute FedAvg ($\mu=0.0$) under Dirichlet $\alpha=0.1$ using the exact partition and seed as E5 FedProx.
* **Map the Stability-Plasticity Pareto Frontier:** Run a 5-point parameter sweep across $\mu \in \{0.0, 0.001, 0.01, 0.05, 0.1\}$ under $\alpha=0.1$.
* **Multi-Seed Statistical Verification:** Execute 3 random seeds ($42, 101, 2024$) for all 5 configurations ($15$ simulations total), logging mean and standard deviation for Accuracy, Macro-F1, Minority Recall (`Web-based`, `Brute-force`), and Parameter Drift ($L_2$ norm).
* **Guaranteed Leakage-Free Evaluation:** Evaluate all configurations exclusively on the stationary $30\%$ holdout test set ($1,534,918$ samples), using `RobustScaler` fitted strictly on training data.
* **Document Chapter Upgrades:** Integrate findings and Pareto frontier plots directly into Chapters 4 and 5 of the thesis report (`thesis_report/`).

### Out-of-Scope Non-Goals
* **Developing a Brand-New Aggregation Algorithm:** This study is strictly an empirical diagnostic benchmark; it does not deploy a new custom optimizer or theorem in the primary test matrix.
* **Multi-Dataset Ingestion (Edge-IIoTset / TON_IoT):** Cross-dataset evaluation is deferred to future work due to schema mismatches (39 flow features vs 61 features) and compute constraints.
* **Generative Data Resampling (GANs / SMOTE):** Data synthesis is excluded due to memory and compute infeasibility on edge IoT microcontrollers.

## Alternatives Weighed

* **Direction B: Developing a Novel Algorithmic Fix (Deferred by Author):**
  * *Reason:* Requires implementing, debugging, and proving a new algorithm (e.g., Dual-Head Decoupled FL or Adaptive $\mu$) on Colab within a tight graduation deadline. If the algorithm yields noisy results, the entire thesis defense is compromised.
* **Adding a Second Dataset (Edge-IIoTset) (Rejected by Author):**
  * *Reason:* Edge-IIoTset uses different feature extractors (61 features vs. 39 features in CICIoT2023). Running parallel benchmarks doubles data preprocessing and compute overhead without resolving the core mathematical question of proximal regularization.
* **Synthetic Oversampling via SMOTE-ENN / GANs (Rejected):**
  * *Reason:* Synthetic sample generation requires large memory buffers and intensive compute, violating the hardware invariants of edge IoT devices (e.g., microcontrollers, low-power routers).

## Key Decisions & Staging for Next Steps

### Key Technical Decisions
1. **Benchmark Matrix:** 15 total simulations (5 parameter settings $\times$ 3 random seeds).
2. **Computational Budget:** 10 communication rounds $\times$ 2 local epochs per simulation; estimated total runtime is $\approx 40$ minutes on Google Colab T4 GPU.
3. **Artifact Output:** Results will be written to `reports/pareto_frontier_e5_ciciot2023.csv` and visualized as Pareto trade-off curves.

### Staging for Next Steps
1. **Runner Deployment:** Embed the automated sweep runner script into `notebooks/colab_experiment_runner.ipynb`.
2. **Execution & Metric Logging:** Run the 15-scenario sweep on Google Colab and push raw CSV results back to the repository.
3. **Report Integration:** Update Table V.1 in `thesis_report/05_results_discussions/results_discussions.tex` with mean $\pm$ standard deviation and insert the Pareto curve figure.
