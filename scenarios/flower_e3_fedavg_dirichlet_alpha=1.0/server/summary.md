# 📊 Báo cáo Thực nghiệm Kịch bản: `flower_e3_fedavg_dirichlet_alpha=1.0`

## 1. Thông số Cấu hình Thực nghiệm
| Tham số | Giá trị | Ý nghĩa thực nghiệm |
| :--- | :--- | :--- |
| **Kịch bản (Scenario)** | `flower_e3_fedavg_dirichlet_alpha=1.0` | Tên định danh kịch bản thực nghiệm |
| **Chiến lược FL (Strategy)** | `FEDAVG` (mu=0.0) | Thuật toán tổng hợp tham số tại Server |
| **Phân vùng dữ liệu (Partition)** | `dirichlet` (alpha=1.0) | Mức độ dị biệt Non-IID giữa các Client |
| **Số lượng Edge Clients (K)** | `5` clients | Số trạm mạng IoT biên tham gia huấn luyện |
| **Số vòng giao tiếp (Rounds)** | `10` rounds (Epochs cục bộ: `2`) | Tổng chu kỳ huấn luyện liên đoàn |
| **Chi phí truyền thông (Comm Cost)** | **5.30 MB** | Tổng lưu lượng mạng tải lên/tải xuống |
| **Độ trôi dạt tham số (Client Drift)** | **10.75** | Độ lệch chuẩn Euclidean ||w_local - w_global||_2 |

## 2. Kết quả Đánh giá Tổng thể trên Global Test Set (30% Upcoming Data)
| Chỉ số Đánh giá (Metric) | Kết quả (%) | Ý nghĩa theo Đề cương Luận văn KLTN |
| :--- | :---: | :--- |
| **Test Accuracy** | **84.42%** | Độ chính xác tổng thể trên toàn bộ lưu lượng kiểm định |
| **Macro-F1 Score** | **67.62%** | **Chỉ số cốt lõi** đánh giá cân bằng (không thiên lệch lớp đa số) |
| **Weighted-F1 Score** | **83.73%** | F1 có tính trọng số theo tần suất xuất hiện thực tế |
| **Macro-Precision** | **83.08%** | Độ chuẩn xác trung bình trên 8 lớp tấn công |
| **Macro-Recall** | **64.89%** | Tỷ lệ nhận diện trung bình trên 8 lớp tấn công |

## 3. Khả năng Phát hiện các Lớp Tấn công Thiểu số (Minority Attack Recall)
> [!IMPORTANT]
> Trong môi trường mạng IoT, tấn công thiểu số (Web-based, Brute-force) có tần suất thấp nhưng mức độ đe dọa cao, là trọng tâm đánh giá của Luận văn.

| Lớp tấn công thiểu số | Tần suất xuất hiện | Tỷ lệ nhận diện (Recall %) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Web-based** | 0.49% (Rất hiếm) | **0.39%** | Thách thức do mất cân bằng cực đoan |
| **Brute-force** | 0.26% (Rất hiếm) | **21.54%** | Phát hiện tốt |

## 4. Chi tiết Hiệu năng theo từng Lớp Tấn công (Per-Class Metrics)
| Lớp tấn công (Class) | Phân loại | Precision (%) | Recall (%) | F1-Score (%) | Số mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | Đa số / Chuẩn | 78.76% | 88.44% | 83.32% | 329,458 |
| **Brute-force** | Thiểu số (Minority) | 92.34% | 21.54% | 34.93% | 3,919 |
| **DDoS** | Đa số / Chuẩn | 86.75% | 96.35% | 91.30% | 576,601 |
| **DoS** | Đa số / Chuẩn | 86.51% | 61.39% | 71.82% | 219,103 |
| **Mirai** | Đa số / Chuẩn | 99.86% | 99.60% | 99.73% | 63,247 |
| **Recon** | Đa số / Chuẩn | 77.14% | 71.58% | 74.26% | 207,160 |
| **Spoof** | Đa số / Chuẩn | 90.57% | 79.83% | 84.86% | 127,981 |
| **Web-based** | Thiểu số (Minority) | 52.73% | 0.39% | 0.77% | 7,449 |

## 5. Trực quan hóa Đồ thị Thực nghiệm
### 📈 Quỹ đạo Hội tụ và Chi phí Mạng (Convergence & Communication)
![Convergence Trajectory](convergence.png)

### 🎯 Ma trận Nhầm lẫn & Tỷ lệ Nhận diện theo Lớp (Confusion Matrix & Recall)
![Confusion Matrix](confusion_matrix.png)
