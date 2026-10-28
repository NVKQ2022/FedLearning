# 📊 Báo cáo Thực nghiệm Kịch bản: `flower_e6_fedavg_iid_client=10`

## 1. Thông số Cấu hình Thực nghiệm
| Tham số | Giá trị | Ý nghĩa thực nghiệm |
| :--- | :--- | :--- |
| **Kịch bản (Scenario)** | `flower_e6_fedavg_iid_client=10` | Tên định danh kịch bản thực nghiệm |
| **Chiến lược FL (Strategy)** | `FEDAVG` (mu=0.0) | Thuật toán tổng hợp tham số tại Server |
| **Phân vùng dữ liệu (Partition)** | `iid` (alpha=0.5) | Mức độ dị biệt Non-IID giữa các Client |
| **Số lượng Edge Clients (K)** | `10` clients | Số trạm mạng IoT biên tham gia huấn luyện |
| **Số vòng giao tiếp (Rounds)** | `10` rounds (Epochs cục bộ: `2`) | Tổng chu kỳ huấn luyện liên đoàn |
| **Chi phí truyền thông (Comm Cost)** | **10.60 MB** | Tổng lưu lượng mạng tải lên/tải xuống |
| **Độ trôi dạt tham số (Client Drift)** | **7.25** | Độ lệch chuẩn Euclidean ||w_local - w_global||_2 |

## 2. Kết quả Đánh giá Tổng thể trên Global Test Set (30% Upcoming Data)
| Chỉ số Đánh giá (Metric) | Kết quả (%) | Ý nghĩa theo Đề cương Luận văn KLTN |
| :--- | :---: | :--- |
| **Test Accuracy** | **84.41%** | Độ chính xác tổng thể trên toàn bộ lưu lượng kiểm định |
| **Macro-F1 Score** | **67.28%** | **Chỉ số cốt lõi** đánh giá cân bằng (không thiên lệch lớp đa số) |
| **Weighted-F1 Score** | **83.73%** | F1 có tính trọng số theo tần suất xuất hiện thực tế |
| **Macro-Precision** | **83.69%** | Độ chuẩn xác trung bình trên 8 lớp tấn công |
| **Macro-Recall** | **64.81%** | Tỷ lệ nhận diện trung bình trên 8 lớp tấn công |

## 3. Khả năng Phát hiện các Lớp Tấn công Thiểu số (Minority Attack Recall)
> [!IMPORTANT]
> Trong môi trường mạng IoT, tấn công thiểu số (Web-based, Brute-force) có tần suất thấp nhưng mức độ đe dọa cao, là trọng tâm đánh giá của Luận văn.

| Lớp tấn công thiểu số | Tần suất xuất hiện | Tỷ lệ nhận diện (Recall %) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Web-based** | 0.49% (Rất hiếm) | **0.39%** | Thách thức do mất cân bằng cực đoan |
| **Brute-force** | 0.26% (Rất hiếm) | **19.21%** | Phát hiện tốt |

## 4. Chi tiết Hiệu năng theo từng Lớp Tấn công (Per-Class Metrics)
| Lớp tấn công (Class) | Phân loại | Precision (%) | Recall (%) | F1-Score (%) | Số mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | Đa số / Chuẩn | 79.46% | 87.84% | 83.44% | 329,458 |
| **Brute-force** | Thiểu số (Minority) | 95.32% | 19.21% | 31.98% | 3,919 |
| **DDoS** | Đa số / Chuẩn | 86.84% | 96.03% | 91.21% | 576,601 |
| **DoS** | Đa số / Chuẩn | 85.61% | 61.78% | 71.77% | 219,103 |
| **Mirai** | Đa số / Chuẩn | 99.79% | 99.56% | 99.68% | 63,247 |
| **Recon** | Đa số / Chuẩn | 77.07% | 71.63% | 74.25% | 207,160 |
| **Spoof** | Đa số / Chuẩn | 88.55% | 82.00% | 85.15% | 127,981 |
| **Web-based** | Thiểu số (Minority) | 56.86% | 0.39% | 0.77% | 7,449 |

## 5. Trực quan hóa Đồ thị Thực nghiệm
### 📈 Quỹ đạo Hội tụ và Chi phí Mạng (Convergence & Communication)
![Convergence Trajectory](convergence.png)

### 🎯 Ma trận Nhầm lẫn Toàn diện (Normalized Confusion Matrix 8x8)
![Confusion Matrix](confusion_matrix.png)

### 📊 Tỷ lệ Nhận diện theo Lớp & Tấn công Thiểu số (Per-Class Recall)
![Per-Class Recall](per_class_recall.png)
