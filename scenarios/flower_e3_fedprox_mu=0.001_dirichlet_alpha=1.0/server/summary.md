# 📊 Báo cáo Thực nghiệm Kịch bản: `flower_e3_fedprox_mu=0.001_dirichlet_alpha=1.0`

## 1. Thông số Cấu hình Thực nghiệm
| Tham số | Giá trị | Ý nghĩa thực nghiệm |
| :--- | :--- | :--- |
| **Kịch bản (Scenario)** | `flower_e3_fedprox_mu=0.001_dirichlet_alpha=1.0` | Tên định danh kịch bản thực nghiệm |
| **Chiến lược FL (Strategy)** | `FEDPROX` (mu=0.001) | Thuật toán tổng hợp tham số tại Server |
| **Phân vùng dữ liệu (Partition)** | `dirichlet` (alpha=1.0) | Mức độ dị biệt Non-IID giữa các Client |
| **Số lượng Edge Clients (K)** | `5` clients | Số trạm mạng IoT biên tham gia huấn luyện |
| **Số vòng giao tiếp (Rounds)** | `10` rounds (Epochs cục bộ: `2`) | Tổng chu kỳ huấn luyện liên đoàn |
| **Chi phí truyền thông (Comm Cost)** | **5.30 MB** | Tổng lưu lượng mạng tải lên/tải xuống |
| **Độ trôi dạt tham số (Client Drift)** | **2.15** | Độ lệch chuẩn Euclidean ||w_local - w_global||_2 |

## 2. Kết quả Đánh giá Tổng thể trên Global Test Set (30% Upcoming Data)
| Chỉ số Đánh giá (Metric) | Kết quả (%) | Ý nghĩa theo Đề cương Luận văn KLTN |
| :--- | :---: | :--- |
| **Test Accuracy** | **83.83%** | Độ chính xác tổng thể trên toàn bộ lưu lượng kiểm định |
| **Macro-F1 Score** | **66.28%** | **Chỉ số cốt lõi** đánh giá cân bằng (không thiên lệch lớp đa số) |
| **Weighted-F1 Score** | **83.12%** | F1 có tính trọng số theo tần suất xuất hiện thực tế |
| **Macro-Precision** | **79.27%** | Độ chuẩn xác trung bình trên 8 lớp tấn công |
| **Macro-Recall** | **63.74%** | Tỷ lệ nhận diện trung bình trên 8 lớp tấn công |

## 3. Khả năng Phát hiện các Lớp Tấn công Thiểu số (Minority Attack Recall)
> [!IMPORTANT]
> Trong môi trường mạng IoT, tấn công thiểu số (Web-based, Brute-force) có tần suất thấp nhưng mức độ đe dọa cao, là trọng tâm đánh giá của Luận văn.

| Lớp tấn công thiểu số | Tần suất xuất hiện | Tỷ lệ nhận diện (Recall %) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Web-based** | 0.49% (Rất hiếm) | **0.01%** | Thách thức do mất cân bằng cực đoan |
| **Brute-force** | 0.26% (Rất hiếm) | **16.97%** | Phát hiện tốt |

## 4. Chi tiết Hiệu năng theo từng Lớp Tấn công (Per-Class Metrics)
| Lớp tấn công (Class) | Phân loại | Precision (%) | Recall (%) | F1-Score (%) | Số mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | Đa số / Chuẩn | 77.95% | 87.98% | 82.66% | 329,458 |
| **Brute-force** | Thiểu số (Minority) | 98.37% | 16.97% | 28.94% | 3,919 |
| **DDoS** | Đa số / Chuẩn | 86.55% | 96.05% | 91.06% | 576,601 |
| **DoS** | Đa số / Chuẩn | 85.43% | 60.82% | 71.05% | 219,103 |
| **Mirai** | Đa số / Chuẩn | 99.88% | 99.36% | 99.62% | 63,247 |
| **Recon** | Đa số / Chuẩn | 76.00% | 70.64% | 73.22% | 207,160 |
| **Spoof** | Đa số / Chuẩn | 90.01% | 78.10% | 83.63% | 127,981 |
| **Web-based** | Thiểu số (Minority) | 20.00% | 0.01% | 0.03% | 7,449 |

## 5. Trực quan hóa Đồ thị Thực nghiệm
### 📈 Quỹ đạo Hội tụ và Chi phí Mạng (Convergence & Communication)
![Convergence Trajectory](convergence.png)

### 🎯 Ma trận Nhầm lẫn Toàn diện (Normalized Confusion Matrix 8x8)
![Confusion Matrix](confusion_matrix.png)

### 📊 Tỷ lệ Nhận diện theo Lớp & Tấn công Thiểu số (Per-Class Recall)
![Per-Class Recall](per_class_recall.png)
