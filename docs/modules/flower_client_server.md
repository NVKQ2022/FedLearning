# 📦 Module Documentation: `src/federated/flower_client.py` & `src/federated/flower_server.py`

Publication-grade Flower framework implementation for decentralized IoT Network Intrusion Detection (FL-IoT-IDS) under non-IID conditions.

* **Client File:** [`src/federated/flower_client.py`](file:///home/quan/projects/FedLearning/src/federated/flower_client.py)
* **Server File:** [`src/federated/flower_server.py`](file:///home/quan/projects/FedLearning/src/federated/flower_server.py)
* **Thesis Mapping:** Directly realizes Sections 5, 6, 7, 8, and 9 of the thesis proposal:
  *"Xây dựng và đánh giá prototype hệ thống phát hiện xâm nhập IoT sử dụng Federated Learning trong môi trường dữ liệu non-IID"* (Nguyễn Việt Kỳ Quân, 2026).

---

## 🗺️ Architectural Context & Flower gRPC Pipeline

```mermaid
flowchart TD
    subgraph FederatedServerLayer["Federated Server Layer (flower_server.py)"]
        ServerStrategy["FlowerIoTServerStrategy (FedAvg / FedProx)"]
        CentralEval["build_flower_server_eval_fn()<br>• Macro-F1 (imbalance-safe)<br>• Minority Recall (Web, Brute-force)<br>• Checkpoint best weights"]
        CommTracker["Communication Cost Tracker<br>2 * sum(m_t * Model_Size) bytes"]
    end

    subgraph Network["gRPC Transport (127.0.0.1:8080 or LAN)"]
        ProtoDown["Broadcast: Global Parameters w_t + Config (mu, local_epochs)"]
        ProtoUp["Client Report: Updated Weights w_k + Samples n_k + Metrics (drift, time)"]
    end

    subgraph FederatedClientLayer["Federated Client Layer (flower_client.py)"]
        ClientNode["FlowerIoTClient (flwr.client.NumPyClient)"]
        LocalData["Private Client DataLoader<br>(Dirichlet Non-IID or IID)"]
        Trainer["FederatedClientTrainer<br>• Local epochs<br>• Proximal penalty: (mu/2)*||w - w_t||^2<br>• Gradient clipping"]
        Diagnostics["Local Diagnostics<br>• Wall-clock training time<br>• Parameter drift: ||w_loc - w_glob||_2"]
    end

    ServerStrategy --> ProtoDown
    ProtoDown --> ClientNode
    ClientNode --> LocalData --> Trainer --> Diagnostics
    Diagnostics --> ProtoUp
    ProtoUp --> ServerStrategy
    ServerStrategy --> CentralEval
    ServerStrategy --> CommTracker
```

---

## 1. What Do They Do?

### 1.1 `FlowerIoTClient` ([`flower_client.py`](file:///home/quan/projects/FedLearning/src/federated/flower_client.py))
* **Framework Interface:** Subclasses `flwr.client.NumPyClient`.
* **Zero Protobuf Friction:** Exchanging weights as universal `List[np.ndarray]` via Flower's high-level API.
* **FedProx Proximal Loss Support:** Receives `mu` parameter dynamically from server config, constraining parameter drift:
  $$\mathcal{L}_{\text{prox}}(w) = \mathcal{L}_{\text{task}}(w) + \frac{\mu}{2} \|w - w_t\|_2^2$$
* **Client Diagnostics:** Measures client parameter drift $\|w_{\text{local}} - w_{\text{global}}\|_2$ and local wall-clock training time per round.
* **Standalone CLI Execution:** Can be launched as an independent OS process:
  ```bash
  python -m src.federated.flower_client --client-id 0 --server-address 127.0.0.1:8080
  ```

### 1.2 `FlowerIoTServerStrategy` ([`flower_server.py`](file:///home/quan/projects/FedLearning/src/federated/flower_server.py))
* **Framework Interface:** Subclasses `flwr.server.strategy.FedAvg` / `FedProx`.
* **Centralized Holdout Evaluation:** Evaluates global model on the unseen test set after every round, tracking:
  - **Macro-F1 Score:** Prevents majority DDoS/DoS traffic from dominating evaluation.
  - **Minority Class Recall:** Tracks stealth attack isolation (`Web-based`, `Brute-force`).
  - **Best Checkpoint Preservation:** Automatically archives top weights (`flower_fedprox_best_weights.npz`).
* **Communication Cost Tracking:** Accurately estimates cumulative payload in megabytes:
  $$\text{Cumulative Communication} = 2 \times \sum_{t=1}^T m_t \times B$$
  where $m_t$ is participating clients and $B \approx 54.28\text{ KB}$ for `TabularIoTMLPModel`.
* **Standalone Server Process:**
  ```bash
  python -m src.federated.flower_server --server-address 0.0.0.0:8080 --rounds 10 --strategy fedprox --mu 0.05
  ```

---

## 2. Why Do You Need It?

| Thesis Problem / Constraint | Naive Approach Failure Mode | How Our Flower Implementation Solves It |
| :--- | :--- | :--- |
| **Non-IID Client Drift** | Standard FedAvg oscillates or diverges under severe label skew ($\alpha = 0.1$). | `FlowerIoTClient` applies proximal regularization with $\mu$ and tracks parameter drift $\|w - w_t\|_2$. |
| **Evaluation Skew in FL** | Averaging client validation metrics gives biased scores because clients have skewed local classes. | Server executes **centralized holdout evaluation** (`build_flower_server_eval_fn`) using a balanced test set with zero client data leakage. |
| **System Profiling Requirement** | Most FL repos only measure accuracy, ignoring thesis Section 9 (communication & time). | `FlowerIoTServerStrategy` automatically logs total wall-clock times, round durations, and communication megabytes. |
| **Single-Machine Reality** | Buying 10 physical Raspberry Pis is impractical for rapid development. | Clients and Server can run as separate OS processes over `127.0.0.1:8080` or via Flower simulation. |

---

## 3. How To Use It?

### 3.1 Localhost Multi-Process Execution (Real gRPC over `127.0.0.1:8080`)

```python
# 1. Start Server in terminal 1:
python -m src.federated.flower_server --server-address 127.0.0.1:8080 --rounds 10 --strategy fedprox --mu 0.05

# 2. Start Client 0 in terminal 2:
python -m src.federated.flower_client --client-id 0 --server-address 127.0.0.1:8080

# 3. Start Client 1 in terminal 3:
python -m src.federated.flower_client --client-id 1 --server-address 127.0.0.1:8080
```

### 3.2 Programmatic Integration in Python / Notebooks

```python
from src.federated.flower_client import FlowerIoTClient
from src.federated.flower_server import FlowerIoTServerStrategy, build_flower_server_eval_fn

# 1. Server Centralized Evaluation
eval_fn = build_flower_server_eval_fn(
    model=global_model,
    val_loader=data["val_loader"],
    class_names=class_names,
    minority_classes=["Web-based", "Brute-force"],
    device=device
)

# 2. Strategy
strategy = FlowerIoTServerStrategy(
    strategy_name="fedprox",
    mu=0.05,
    min_fit_clients=5,
    evaluate_fn=eval_fn
)
```
