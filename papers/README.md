# Federated Learning Canonical Research Papers

Thư mục này chứa các bài báo khoa học nền tảng làm cơ sở lý thuyết (methodology) cho phương pháp đánh giá mô hình trong dự án.

## 1. Flower: A Friendly Federated Learning Research Framework
- **Tác giả:** Beutel et al., 2020 (arXiv:2007.14390)
- **Vai trò:** Bài báo giới thiệu kiến trúc framework Flower được sử dụng trực tiếp trong mã nguồn.
- **Trích dẫn chính:** Định nghĩa kiến trúc đánh giá hai tầng: *Centralized Evaluation* (đánh giá tập trung trên Server) và *Federated Evaluation* (đánh giá phân tán trên các Client).

## 2. Advances and Open Problems in Federated Learning
- **Tác giả:** Kairouz et al., 2019/2021 (arXiv:1912.04977)
- **Vai trò:** Bản survey toàn diện và sâu sắc nhất về Federated Learning (được viết bởi các chuyên gia từ Google, Stanford, MIT, v.v.).
- **Trích dẫn chính:** Phân tích các giới hạn của phương pháp đánh giá phân tán (Decentralized Evaluation) khi đối mặt với dữ liệu Non-IID và Data Poisoning; làm nổi bật sự cần thiết của một tập kiểm thử toàn cục (Global Test Set) trên máy chủ trong các thiết lập mô phỏng.

## 3. LEAF: A Benchmark for Federated Settings
- **Tác giả:** Caldas et al., 2018 (arXiv:1812.01097)
- **Vai trò:** Chuẩn mực cho việc benchmarking (đánh giá và so sánh) các thuật toán Federated Learning.
- **Trích dẫn chính:** Định hình phương pháp chuẩn để chia (split) dữ liệu Train/Validation/Test, chứng minh việc đo lường hiệu năng tổng thể của thuật toán phải được tiến hành độc lập để tránh bị sai lệch bởi phân phối dữ liệu không đồng nhất cục bộ.

## 4. Federated Evaluation of On-device Personalization
- **Tác giả:** Wang et al., 2019 (arXiv:1910.10252) - Nhóm nghiên cứu từ Google.
- **Vai trò:** Bổ sung góc nhìn chuyên sâu về cách đánh giá mô hình trong Federated Learning, đặc biệt là khi mô hình được cá nhân hóa (personalized) trên từng thiết bị.
- **Trích dẫn chính:** Bài báo thảo luận chi tiết về sự cần thiết của *Federated Evaluation* (đánh giá phân tán) khi dữ liệu hoàn toàn phi tập trung và cách đo lường chính xác tác động của Global Model đối với trải nghiệm cá nhân hóa của từng Client. Nó cung cấp cơ sở vững chắc cho việc sử dụng `fraction_evaluate` để các Client tự đánh giá trên tập Local Test của mình.
