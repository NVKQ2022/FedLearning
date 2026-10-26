# 📊 Báo cáo Thực nghiệm Kịch bản: `flower_e6_fedavg_iid_client=7`

## 1. Thông số Cấu hình Thực nghiệm
| Tham số | Giá trị | Ý nghĩa thực nghiệm |
| :--- | :--- | :--- |
| **Kịch bản (Scenario)** | `flower_e6_fedavg_iid_client=7` | Tên định danh kịch bản thực nghiệm |
| **Chiến lược FL (Strategy)** | `FEDAVG` (mu=0.0) | Thuật toán tổng hợp tham số tại Server |
| **Phân vùng dữ liệu (Partition)** | `iid` (alpha=0.5) | Mức độ dị biệt Non-IID giữa các Client |
| **Số lượng Edge Clients (K)** | `7` clients | Số trạm mạng IoT biên tham gia huấn luyện |
| **Số vòng giao tiếp (Rounds)** | `10` rounds (Epochs cục bộ: `2`) | Tổng chu kỳ huấn luyện liên đoàn |
| **Chi phí truyền thông (Comm Cost)** | **7.42 MB** | Tổng lưu lượng mạng tải lên/tải xuống |
| **Độ trôi dạt tham số (Client Drift)** | **8.42** | Độ lệch chuẩn Euclidean ||w_local - w_global||_2 |

## 2. Kết quả Đánh giá Tổng thể trên Global Test Set (30% Upcoming Data)
| Chỉ số Đánh giá (Metric) | Kết quả (%) | Ý nghĩa theo Đề cương Luận văn KLTN |
| :--- | :---: | :--- |
| **Test Accuracy** | **84.54%** | Độ chính xác tổng thể trên toàn bộ lưu lượng kiểm định |
| **Macro-F1 Score** | **67.82%** | **Chỉ số cốt lõi** đánh giá cân bằng (không thiên lệch lớp đa số) |
| **Weighted-F1 Score** | **83.94%** | F1 có tính trọng số theo tần suất xuất hiện thực tế |
| **Macro-Precision** | **82.97%** | Độ chuẩn xác trung bình trên 8 lớp tấn công |
| **Macro-Recall** | **65.25%** | Tỷ lệ nhận diện trung bình trên 8 lớp tấn công |

## 3. Khả năng Phát hiện các Lớp Tấn công Thiểu số (Minority Attack Recall)
> [!IMPORTANT]
> Trong môi trường mạng IoT, tấn công thiểu số (Web-based, Brute-force) có tần suất thấp nhưng mức độ đe dọa cao, là trọng tâm đánh giá của Luận văn.

| Lớp tấn công thiểu số | Tần suất xuất hiện | Tỷ lệ nhận diện (Recall %) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Web-based** | 0.49% (Rất hiếm) | **0.63%** | Thách thức do mất cân bằng cực đoan |
| **Brute-force** | 0.26% (Rất hiếm) | **21.10%** | Phát hiện tốt |

## 4. Chi tiết Hiệu năng theo từng Lớp Tấn công (Per-Class Metrics)
| Lớp tấn công (Class) | Phân loại | Precision (%) | Recall (%) | F1-Score (%) | Số mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | Đa số / Chuẩn | 79.56% | 87.95% | 83.55% | 329,458 |
| **Brute-force** | Thiểu số (Minority) | 93.24% | 21.10% | 34.42% | 3,919 |
| **DDoS** | Đa số / Chuẩn | 87.35% | 95.49% | 91.24% | 576,601 |
| **DoS** | Đa số / Chuẩn | 84.33% | 63.70% | 72.58% | 219,103 |
| **Mirai** | Đa số / Chuẩn | 99.84% | 99.53% | 99.68% | 63,247 |
| **Recon** | Đa số / Chuẩn | 76.81% | 72.32% | 74.50% | 207,160 |
| **Spoof** | Đa số / Chuẩn | 89.83% | 81.26% | 85.33% | 127,981 |
| **Web-based** | Thiểu số (Minority) | 52.81% | 0.63% | 1.25% | 7,449 |

## 5. Trực quan hóa Đồ thị Thực nghiệm
### 📈 Quỹ đạo Hội tụ và Chi phí Mạng (Convergence & Communication)
![Convergence Trajectory](convergence.png)

### 🎯 Ma trận Nhầm lẫn Toàn diện (Normalized Confusion Matrix 8x8)
![Confusion Matrix](confusion_matrix.png)

### 📊 Tỷ lệ Nhận diện theo Lớp & Tấn công Thiểu số (Per-Class Recall)
![Per-Class Recall](per_class_recall.png)
