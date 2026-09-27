# 📝 Paper Audit Sheets & Literature Extraction Cards

This document contains extraction records, baseline metrics, and audit findings for the literature repository in `papers/`.

---

## 1. Local PDF Repository Audit Log

| Status | File Name | Stated Citation | Actual Paper Title Extracted | Recommendation |
| :---: | :--- | :--- | :--- | :--- |
| ✅ | `Belarbi2023_Federated_Deep_Learning_for_Intrusion_Detection_in_IoT.pdf` | Belarbi et al. (2023) | *Federated Deep Learning for Intrusion Detection in IoT Networks* | Validated |
| ✅ | `Beutel2020_Flower.pdf` | Beutel et al. (2020) | *FLOWER: A Friendly Federated Learning Framework* | Validated |
| ✅ | `Dadkhah2024_CICIoMT2024.pdf` | Dadkhah et al. (2024) | *CICIoMT2024: Attack Vectors in Healthcare Devices...* | Validated |
| ✅ | `Kairouz2021_Advances_and_Open_Problems_in_Federated_Learning.pdf` | Kairouz et al. (2021) | *Advances and Open Problems in Federated Learning* | Validated |
| ✅ | `Karimireddy2020_SCAFFOLD.pdf` | Karimireddy et al. (2020) | *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning* | Validated |
| ✅ | `Li2020_FedProx.pdf` | Li et al. (2020) | *Federated Optimization in Heterogeneous Networks* (FedProx) | Validated |
| ✅ | `McMahan2017_FedAvg.pdf` | McMahan et al. (2017) | *Communication-Efficient Learning of Deep Networks from Decentralized Data* | Validated |
| ✅ | `Neto2023_CICIoT2023.pdf` | Neto et al. (2023) | *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT* | Validated |
| ✅ | `VARS_FL_Client_Selection_Non_IID_IoT.pdf` | Lakas & Ferrag (2024) | *VARS-FL: Validation-Aligned Client Selection for Non-IID FL in IoT* | Validated |
| ✅ | `Zhang2024_FedMADE_Robust_FL_IoT_Intrusion_Detection.pdf` | Sun, Sharma, Nwodo, Stavrou, Wang (2024) | *FedMADE: Robust Federated Learning for Intrusion Detection in IoT Networks* | Validated |
| ⚠️ | `Rey2021_Federated_Learning_IoT_Intrusion_Detection.pdf` | Rey et al. (2021) | *Federated Learning for Internet of Things: A FL Framework for On-Device Anomaly Data Detection* (Tuo Zhang, Chaoyang He et al. - FedML) | Author is Tuo Zhang et al., not Rey |
| ❌ | `Ferrag2020_Federated_Deep_Learning_IoT_CyberSecurity.pdf` | Ferrag et al. (2021) | *CASTELO: Clustered Atom Subtypes Aided Lead Optimization* (IBM Watson) | **Incorrect download: Chemistry paper. Redownload Ferrag et al., IEEE Access 2021** |
| ❌ | `RuzafaAlcazar2021_Evaluating_FL_for_IoT_Intrusion_Detection.pdf` | Ruzafa-Alcázar et al. (2021) | *Misalignment vs Topology in Axion-Like Models* (JHEP) | **Incorrect download: Physics paper. Redownload Ruzafa-Alcázar et al., IEEE TII 2021** |

---

## 2. Deep Extraction Cards

### Card 1: McMahan et al. (2017) — FedAvg
* **Venue:** AISTATS 2017
* **Core Formula:**
  $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_{t+1}^k, \quad w_{t+1}^k \leftarrow w_t - \eta \sum_{\tau=1}^E \nabla F_k(w_{t,\tau}^k)$$
* **Assumptions:** Synchronous rounds, fraction $C$ of clients sampled per round, non-convex global objective.
* **Key Finding:** Adding local SGD epochs ($E > 1$) reduces required communication rounds by 1-2 orders of magnitude on IID and mild non-IID data.
* **Vulnerability:** Under severe non-IID distributions, local models drift toward local stationary points, causing instability and degraded accuracy upon server averaging.

---

### Card 2: Li et al. (2020) — FedProx
* **Venue:** MLSys 2020
* **Core Objective:**
  $$\min_w h_k(w; w_t) = F_k(w) + \frac{\mu}{2}\|w - w_t\|_2^2$$
* **Key Innovations:**
  1. **Proximal Regularizer ($\mu$):** Shrinks local updates towards current global model $w_t$, controlling drift radius.
  2. **Tolerating Inexact Solutions ($\gamma$-inexact):** Clients can perform variable work (variable local epochs $E_k$) based on system battery/compute constraints without breaking theoretical convergence.
* **Optimal Hyperparameter Range:** Typically $\mu \in [0.001, 1.0]$. Too large $\mu$ freezes model updates; too small $\mu \to 0$ behaves identically to FedAvg.
* **Communication Overhead:** Identical to FedAvg ($1\times$ payload, no auxiliary state transmitted).

---

### Card 3: Neto et al. (2023) — CICIoT2023 Benchmark
* **Venue:** MDPI Sensors 2023, 23(13), 5941
* **Environment:** 105 devices, 33 attack types executed across 7 attack classes + 1 Benign class.
* **Extracted Centralized Baselines (Multi-Class 8-Class):**
  * **Deep Neural Network (DNN):** Accuracy: $99.11\%$, Precision: $67.94\%$, Recall: $90.66\%$, F1-score: $69.73\%$
  * **Random Forest (RF):** Accuracy: $99.44\%$, Precision: $70.54\%$, Recall: $91.00\%$, F1-score: $71.93\%$
  * **Logistic Regression (LR):** Accuracy: $83.17\%$, Precision: $51.24\%$, Recall: $69.61\%$, F1-score: $53.94\%$
* **Extracted Centralized Baselines (Binary 2-Class):**
  * **DNN:** Accuracy: $99.44\%$, Precision: $94.76\%$, Recall: $93.33\%$, F1-score: $94.03\%$
  * **RF:** Accuracy: $99.68\%$, Precision: $96.54\%$, Recall: $96.52\%$, F1-score: $96.53\%$
* **Key Takeaway:** The centralized MLP (DNN) achieves $69.73\%$ F1 on 8 classes due to severe minority attack degradation. This sets the target upper bound for our FL prototype.

---

### Card 4: Sun et al. (2024) — FedMADE (Virginia Tech)
* **Venue:** arXiv:2408.07152 (cs.CR), Aug 2024
* **Dataset:** CICIoT2023 mapped to 63 IoT device clients.
* **Critical Baseline Results on CICIoT2023:**
  * Binary Attack Detection: FedAvg ($99.65\%$ F1), FedProx ($99.62\%$ F1).
  * **Multi-Class Per-Class Accuracy (Minority Attacks):**
    * Web-based Attacks: FedAvg achieves **0% to 13.8%**; FedProx achieves **~10%**.
    * Brute-Force Attacks: FedAvg achieves **< 15%**; FedProx achieves **~12%**.
  * **Cause:** Web-based and Brute-force represent only $0.74\%$ of total traffic and exist on only 3 of 67 devices. FedAvg averaging completely dilutes their parameters.
* **Relevance to Thesis:** Confirms that reporting Accuracy is misleading and proves that measuring minority class detection in CICIoT2023 is a cutting-edge research concern.

---

### Card 5: Belarbi et al. (2023) — Federated Deep Learning for IoT IDS
* **Venue:** IEEE NetSoft / Globecom 2023 (Cardiff University & Toshiba Europe)
* **Tech Stack:** Flower (`flwr`) + PyTorch
* **Dataset:** TON-IoT (10 classes), split among 10 clients by IP address.
* **Architectures:** DBN and DNN (MLP with layers 38 -> 128 -> 128 -> 64 -> 10, ReLU activations).
* **Algorithms:** FedAvg, FedProx, FedYogi across 50 communication rounds, 2 local epochs/round.
* **Hardware:** 8-core CPU, 16GB RAM.
* **Key Finding:** FedProx stabilized loss reduction over FedAvg under skewed IP-based distributions, but server-side pre-training provided the biggest acceleration.

---

### Card 6: Karimireddy et al. (2020) — SCAFFOLD
* **Venue:** ICML 2020
* **Core Idea:** Adds client control variate $c_i$ and global control variate $c$ to estimate client drift and modify local SGD update direction:
  $$g_i(w) - c_i + c$$
* **Pros:** Theoretically eliminates client drift even under arbitrary non-IID distributions without requiring bounded gradient dissimilarity.
* **Cons for IoT:** Doubled communication cost (clients must upload/download weights $w$ AND control variates $c$), making it prohibitive for bandwidth-constrained IoT edge devices.

---

## 3. BibTeX Citation Repository for Thesis

```bibtex
@inproceedings{mcmahan2017communication,
  author    = {H. Brendan McMahan and Eider Moore and Daniel Ramage and Seth Hampson and Blaise Ag{\"u}era y Arcas},
  title     = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
  booktitle = {Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  pages     = {1273--1282},
  year      = {2017}
}

@inproceedings{li2020federated,
  author    = {Tian Li and Anit Kumar Sahu and Manzil Zaheer and Maziar Sanjabi and Ameet Talwalkar and Virginia Smith},
  title     = {Federated Optimization in Heterogeneous Networks},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)},
  volume    = {2},
  pages     = {429--450},
  year      = {2020}
}

@article{neto2023ciciot2023,
  author    = {Euclides Carlos Pinto Neto and Sajjad Dadkhah and Raphael Ferreira and Amirhossein Zohourian and Rongxing Lu and Ali A. Ghorbani},
  title     = {{CICIoT2023}: A Real-Time Dataset and Benchmark for Large-Scale Attacks in {IoT} Environment},
  journal   = {Sensors},
  volume    = {23},
  number    = {13},
  pages     = {5941},
  year      = {2023},
  publisher = {MDPI}
}

@article{sun2024fedmade,
  author    = {Shihua Sun and Pragya Sharma and Kenechukwu Nwodo and Angelos Stavrou and Haining Wang},
  title     = {{FedMADE}: Robust Federated Learning for Intrusion Detection in {IoT} Networks Using a Dynamic Aggregation Method},
  journal   = {arXiv preprint arXiv:2408.07152},
  year      = {2024}
}

@inproceedings{belarbi2023federated,
  author    = {Othmane Belarbi and Theodoros Spyridopoulos and Eirini Anthi and Ioannis Mavromatis and Pietro Carnelli and Aftab Khan},
  title     = {Federated Deep Learning for Intrusion Detection in {IoT} Networks},
  booktitle = {IEEE International Conference on Communications (ICC) / NetSoft Workshops},
  year      = {2023}
}

@article{lakas2024varsfl,
  author    = {Mohamed Lakas and Mohamed Amine Ferrag},
  title     = {{VARS-FL}: Validation-Aligned Client Selection for Non-{IID} Federated Learning in {IoT} Systems},
  journal   = {IEEE Internet of Things Journal},
  year      = {2024}
}

@article{dadkhah2024ciciomt2024,
  author    = {Sajjad Dadkhah and Euclides Carlos Pinto Neto and Raphael Ferreira and Reginald Chukwuka Molokwu and Somayeh Sadeghi and Ali A. Ghorbani},
  title     = {{CICIoMT2024}: Attack Vectors in Healthcare Devices---A Multi-Protocol Dataset for Assessing {IoMT} Device Security},
  journal   = {Internet of Things},
  year      = {2024},
  publisher = {Elsevier}
}

@article{beutel2020flower,
  author    = {Daniel J. Beutel and Taner Topal and Akhil Mathur and Xinchi Qiu and Javier Fernandez-Marques and Yan Gao and Lorenzo Sani and Kwing Hei Li and Titouan Parcollet and Pedro Porto Buarque de Gusm{\~a}o and Nicholas D. Lane},
  title     = {Flower: A Friendly Federated Learning Research Framework},
  journal   = {arXiv preprint arXiv:2007.14390},
  year      = {2020}
}

@inproceedings{karimireddy2020scaffold,
  author    = {Sai Praneeth Karimireddy and Satyen Kale and Mehryar Mohri and Sashank J. Reddi and Sebastian U. Stich and Ananda Theertha Suresh},
  title     = {{SCAFFOLD}: Stochastic Controlled Averaging for Federated Learning},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  pages     = {5132--5143},
  year      = {2020}
}

@article{kairouz2021advances,
  author    = {Peter Kairouz and H. Brendan McMahan and Brendan Avent and Aur{\'e}lien Bellet and Mehdi Bennis and others},
  title     = {Advances and Open Problems in Federated Learning},
  journal   = {Foundations and Trends{\textregistered} in Machine Learning},
  volume    = {14},
  number    = {1--2},
  pages     = {1--210},
  year      = {2021}
}
```
