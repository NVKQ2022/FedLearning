# 🎓 KHUNG CẤU TRÚC CHI TIẾT KHÓA LUẬN TỐT NGHIỆP (UIT)

> **Tên đề tài (Tiếng Việt):** Xây dựng và đánh giá prototype hệ thống phát hiện xâm nhập IoT sử dụng Federated Learning trong môi trường dữ liệu non-IID  
> **Tên đề tài (English):** Development and Evaluation of an IoT Intrusion Detection System Prototype using Federated Learning in Non-IID Data Environments  
> **Chuyên ngành:** Mạng máy tính & An toàn thông tin - Khoa Mạng máy tính và Truyền thông, Trường ĐH Công nghệ Thông tin (UIT), ĐHQG-HCM  
> **Cán bộ hướng dẫn:** PGS. TS. Lê Trung Quân  
> **Sinh viên thực hiện:** Nguyễn Việt Kỳ Quân (MSSV: 23521267)  

---

## 📑 PHẦN ĐẦU (FRONT MATTER)
* **Trang phụ bìa (Title Page)**
* **Lời cam đoan (Declaration of Authorship)**
* **Lời cảm ơn (Acknowledgments)**
* **Tóm tắt khóa luận (Abstract):** Bản tiếng Việt và tiếng Anh (nêu rõ bối cảnh IoT-IDS, lý do dùng FL, bài toán Non-IID, kết quả thực nghiệm đạt được).
* **Mục lục (Table of Contents)**
* **Danh mục các hình vẽ (List of Figures)**
* **Danh mục các bảng biểu (List of Tables)**
* **Danh mục các ký hiệu & từ viết tắt (List of Acronyms / Abbreviations):** FL, IDS, NIDS, IID, Non-IID, FedAvg, FedProx, MLP, IoT, DoS, DDoS, gRPC, CE, FL-IoT-IDS.

---

## 📖 NỘI DUNG CHÍNH (MAIN BODY)

### 📌 CHƯƠNG 1: GIỚI THIỆU (INTRODUCTION)
* **1.1. Tổng quan & Bối cảnh nghiên cứu (Overview):**
  * Sự bùng nổ của mạng vạn vật (IoT) và nguy cơ mất an toàn an ninh mạng trên các thiết bị biên.
  * Vai trò của Hệ thống phát hiện xâm nhập mạng (NIDS) dựa trên Trí tuệ nhân tạo (AI/ML).
* **1.2. Đặt vấn đề & Các thách thức cốt lõi (Problem Statement):**
  * *Thách thức 1 (Quyền riêng tư & Băng thông):* Mô hình học máy tập trung (Centralized ML) đòi hỏi thu thập dữ liệu thô về máy chủ trung tâm, gây nghẽn mạng và vi phạm chính sách bảo mật dữ liệu IoT.
  * *Thách thức 2 (Dữ liệu phi đồng nhất - Non-IID):* Dữ liệu phân tán giữa các trạm IoT biên có phân phối nhãn rất khác biệt, gây ra hiện tượng trôi dạt trọng số cục bộ (*Client Drift*), làm mô hình liên đoàn khó hội tụ.
  * *Thách thức 3 (Mất cân bằng lớp cực đoan):* Các cuộc tấn công nguy hiểm, tinh vi (như `Web-based`, `Brute-force`) có tần suất xuất hiện rất thấp (< 1%), dễ bị mô hình bỏ sót.
* **1.3. Mục tiêu và Nhiệm vụ nghiên cứu (Research Objectives & Scope):**
  * Mục tiêu tổng quát: Thiết kế, hiện thực prototype FL-IoT-IDS hoàn chỉnh và đánh giá thực nghiệm đa chiều.
  * Các nhiệm vụ cụ thể: Tiền xử lý CICIoT2023, phân chia Dirichlet Non-IID, triển khai kiến trúc 4 lớp với Flower gRPC, so sánh thực nghiệm FedAvg và FedProx.
* **1.4. Phương pháp luận & Tóm tắt đóng góp của đề tài (Methodology & Contributions):**
  * Xây dựng prototype mạng phân tán thực thụ sử dụng Flower framework và giao thức TCP gRPC.
  * Đánh giá toàn diện trên 6 kịch bản chuẩn (E1 đến E6) với các chỉ số kép (Model Quality vs System Overhead).
* **1.5. Phạm vi và Giới hạn của đề tài (Scopes & Limitations):**
  * Tập trung vào dữ liệu dạng bảng network flow; mô phỏng phân tán đa tiến trình trên máy chủ tính toán.
* **1.6. Bố cục của Khóa luận tốt nghiệp (Thesis Outline):** Tóm tắt nội dung các chương tiếp theo.

---

### 📌 CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH LIÊN QUAN (LITERATURE REVIEW)
* **2.1. An ninh mạng IoT và Hệ thống Phát hiện Xâm nhập (IoT-NIDS):**
  * 2.1.1. Đặc trưng lưu lượng và bề mặt tấn công trong hạ tầng IoT.
  * 2.1.2. Tiếp cận phát hiện xâm nhập bằng học máy dạng bảng (Tabular Deep Learning).
  * 2.1.3. Khảo sát tập dữ liệu chuẩn CICIoT2023 (39 đặc trưng số, 8 nhóm nhãn tấn công).
* **2.2. Học liên đoàn (Federated Learning) trong An toàn thông tin:**
  * 2.2.1. Nguyên lý cơ bản: Huấn luyện cục bộ tại biên, chỉ trao đổi trọng số mô hình.
  * 2.2.2. Thuật toán tổng hợp cơ bản Federated Averaging (FedAvg - McMahan et al., 2017).
  * 2.2.3. Lợi ích giảm thiểu chi phí truyền thông và bảo vệ dữ liệu cục bộ.
* **2.3. Thách thức Dữ liệu Non-IID và Thuật toán FedProx:**
  * 2.3.1. Phân loại dị biệt dữ liệu và hiện tượng trôi dạt tham số (*Client Drift*).
  * 2.3.2. Mô hình hóa tính không đồng nhất bằng Phân phối Dirichlet ($Dir(\alpha)$).
  * 2.3.3. Thuật toán Federated Proximal (FedProx - Li et al., 2020) và thành phần phạt điều hòa $\frac{\mu}{2}\|\mathbf{w} - \mathbf{w}_t\|^2$.
* **2.4. Tổng quan các nghiên cứu liên quan & Khoảng trống học thuật (Research Gap):**
  * Bảng so sánh tổng hợp các công trình FL-IDS tiêu biểu gần đây (2020 - 2025).
  * Định vị đóng góp của đề tài trong việc giải quyết đồng thời Non-IID, mất cân bằng lớp và hạn chế phần cứng IoT.

---

### 📌 CHƯƠNG 3: THIẾT KẾ HỆ THỐNG PROTOTYPE FL-IOT-IDS (METHODOLOGY & SYSTEM DESIGN)
* **3.1. Thiết kế tổng thể Kiến trúc 4 Lớp (The 4-Layer Architecture):**
  * Sơ đồ khối tổng thể từ dữ liệu thô, phân vùng, trạm biên, máy chủ tổng hợp đến hệ thống đánh giá.
* **3.2. Lớp 1: Tiền xử lý Dữ liệu và Phân vùng (Data & Partitioning Layer):**
  * 3.2.1. Quy trình tiền xử lý chuẩn không rò rỉ thông tin (No-Leakage Preprocessing với RobustScaler fit trên tập train).
  * 3.2.2. Cơ chế phân vùng dữ liệu: Stratified IID và Dirichlet Non-IID ($\alpha = 1.0, 0.5, 0.1$).
  * 3.2.3. Phân tích độ dị biệt bằng Shannon Entropy và đồ thị phân phối nhãn.
  * 3.2.4. Kỹ thuật cân bằng lớp: Class-weighted Loss và Multi-Class Focal Loss cho các lớp thiểu số.
* **3.3. Lớp 2: Trạm Giám sát IoT Biên (Federated Edge Client Layer):**
  * 3.3.1. Thiết kế mô hình mạng nơ-ron nhỏ gọn TabularIoTMLP (~14,000 tham số, ~54.28 KB) phù hợp phần cứng IoT hạn chế.
  * 3.3.2. Bộ huấn luyện cục bộ (Federated Client Trainer) tích hợp ràng buộc Proximal Regularization.
  * 3.3.3. Đánh giá tính cá nhân hóa (Personalization) trên tập validation riêng nội bộ của từng client.
* **3.4. Lớp 3: Máy chủ Tổng hợp và Điều phối (Federated Server / Aggregation Layer):**
  * 3.4.1. Kiến trúc giao tiếp mạng phân tán gRPC qua Flower framework.
  * 3.4.2. Chiến lược tổng hợp: FedAvgStrategy và TrackingStrategyWrapper (đo độ lệch $\Delta \mathbf{w}$, chi phí truyền thông).
  * 3.4.3. Cơ chế Checkpoint trọng số toàn cục tối ưu (`best_weights.npz`) dựa trên Macro-F1.
* **3.5. Lớp 4: Khung Đánh giá và Đo lường Đa chiều (Evaluation & Monitoring Layer):**
  * 3.5.1. Bộ chỉ số chất lượng mô hình: Macro-F1 (trọng tâm chống thiên lệch), Accuracy, Minority Attack Recall (`Web-based`, `Brute-force`), Confusion Matrix $8 \times 8$.
  * 3.5.2. Bộ chỉ số hiệu năng hệ thống: Chi phí truyền thông tích lũy ($2 \sum m_t B$), Độ trôi tham số ($\|\mathbf{w}_{\text{local}} - \mathbf{w}_{\text{global}}\|_2$), Thời gian hoàn thành vòng giao tiếp (Wall-clock Round Latency).

---

### 📌 CHƯƠNG 4: HIỆN THỰC VÀ KẾT QUẢ THỰC NGHIỆM (IMPLEMENTATION & RESULTS)
* **4.1. Môi trường Hiện thực và Triển khai:**
  * Phần cứng, hệ điều hành và ngăn xếp công nghệ: Python 3.12, PyTorch, Flower, gRPC sockets, Docker/Process simulation.
  * Cấu trúc thư mục Scenario và cơ chế lưu trữ Artifacts tự động (`summary.md`, `confusion_matrix.png`, `convergence.png`, `per_class_recall.png`).
* **4.2. Thiết lập Ma trận Kịch bản Thực nghiệm (Experimental Benchmark Matrix):**
  * **E1:** Mô hình MLP Tập trung làm chuẩn đối sánh (Centralized Baseline Upper-bound).
  * **E2:** Học liên đoàn trong điều kiện lý tưởng IID ($K=5$, FedAvg vs FedProx).
  * **E3:** Đánh giá ảnh hưởng của dữ liệu Non-IID nhẹ (Dirichlet $\alpha=1.0$).
  * **E4:** Đánh giá ảnh hưởng của dữ liệu Non-IID trung bình (Dirichlet $\alpha=0.5$).
  * **E5:** Đánh giá sự suy thoái cực hạn và khả năng ổn định của FedProx với Non-IID nặng (Dirichlet $\alpha=0.1$).
  * **E6:** Khảo sát tính mở rộng quy mô trạm biên (Client Scalability Benchmark với $K \in \{5, 7, 10\}$ clients).
* **4.3. Phân tích và Thảo luận Kết quả (Results Analysis & Discussion):**
  * 4.3.1. Đối sánh Centralized Baseline (E1) vs Federated Learning (E2): Đo lường sự đánh đổi giữa bảo mật dữ liệu và độ chính xác mô hình.
  * 4.3.2. Tác động của mức độ Non-IID (E2 $\to$ E3 $\to$ E4 $\to$ E5): Phân tích hiện tượng suy thoái Macro-F1 và khả năng chống trôi dạt của FedProx với tham số $\mu=0.05$.
  * 4.3.3. Đánh giá khả năng phát hiện các lớp tấn công thiểu số (`Web-based` và `Brute-force`) qua biểu đồ Per-Class Recall và Ma trận nhầm lẫn $8 \times 8$.
  * 4.3.4. Đánh giá quỹ đạo hội tụ (Convergence) và Phân tích chi phí truyền thông (Communication Overhead): Chứng minh tính khả thi trên mạng IoT băng thông hẹp.
  * 4.3.5. Đánh giá tính mở rộng hệ thống (Scalability Benchmark - E6): Tác động của việc tăng số lượng client ($K=5 \to 7 \to 10$) lên thời gian, chi phí mạng và độ ổn định mô hình.
* **4.4. Khuyến nghị Thiết kế Kỹ thuật (Engineering Guidelines):**
  * Hướng dẫn lựa chọn tham số $\mu$, số vòng giao tiếp và kích thước batch cho mạng IoT thực tế.

---

### 📌 CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSIONS & FUTURE WORK)
* **5.1. Các kết quả và đóng góp đạt được:**
  * Xây dựng thành công prototype FL-IoT-IDS chạy thực nghiệm phân tán gRPC thực thụ.
  * Chứng minh thực nghiệm vai trò của FedProx trong việc giảm thiểu suy thoái hiệu năng do Non-IID.
  * Cung cấp bộ benchmark đầy đủ 6 kịch bản với độ tái lập 100%.
* **5.2. Các hạn chế của đề tài:**
  * Mô phỏng multi-process trên một máy chủ tính toán thay vì phân tán vật lý trên các thiết bị nhúng thực.
  * Chưa tích hợp cơ chế bảo mật bổ trợ nâng cao (Differential Privacy, Secure Multi-party Computation) hoặc chống tấn công đầu độc mô hình (Byzantine-robust aggregation).
* **5.3. Hướng phát triển trong tương lai:**
  * Triển khai prototype trực tiếp trên mạng vật lý gồm các bo mạch nhúng (Raspberry Pi, Jetson Nano).
  * Tích hợp kỹ thuật nén gradient/lượng tử hóa để giảm sâu hơn nữa chi phí truyền thông.

---

## 📚 PHẦN CUỐI (BACK MATTER)
* **Tài liệu tham khảo (References):** Trích dẫn chuẩn IEEE (McMahan 2017, Li 2020 FedProx, CICIoT2023, Flower, v.v.).
* **Phụ lục (Appendices):**
  * Phụ lục A: Bảng phân tích phân phối dữ liệu chi tiết theo từng Client (Dirichlet Partition Tables).
  * Phụ lục B: Trích đoạn mã nguồn cốt lõi (Core Implementation Snippets: `flower_client.py`, `flower_server.py`, `grpc_runner.py`).
  * Phụ lục C: Hướng dẫn cài đặt và tái lập môi trường thực nghiệm qua Docker / Google Colab.
