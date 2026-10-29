# 🛡️ Ghi chú 04: Chiến lược Bảo vệ Khóa luận & Bộ Câu hỏi - Trả lời Phản biện (Defense Playbook)

> **Tài liệu tham chiếu nội bộ cho Khóa luận Tốt nghiệp:**  
> *Xây dựng và đánh giá Prototype hệ thống phát hiện xâm nhập IoT sử dụng Federated Learning trong môi trường dữ liệu Non-IID*  
> **Sinh viên thực hiện:** Nguyễn Việt Kỳ Quân — 23521267 | **CBHD:** PGS. TS. Lê Trung Quân

---

## 1. Bản Án Khoa Học Cuối Cùng (Reviewer's Final Verdict)

Trước khi bước ra bảo vệ, sinh viên cần thấu hiểu góc nhìn khách quan của các chuyên gia đánh giá độc lập:

### ⚖️ Nếu chấm như một Bài báo Khoa học Quốc tế (IEEE / Elsevier / Springer):
- **Phán quyết:** **Weak Accept (Chấp nhận ở mức trung bình khá)** cho các hội nghị hoặc tạp chí chuyên ngành Q2/Q3 (dạng bài *Empirical & Benchmark Study*).
- **Lý do:** Công trình chưa có sự đột phá về lý thuyết thuật toán thuần túy (không đủ để vào các hội nghị top-tier như ACM CCS, IEEE S&P, hay NeurIPS), nhưng sở hữu phương pháp luận đo kiểm cực kỳ chặt chẽ, tính trung thực khoa học cao, và cung cấp bằng chứng định lượng có giá trị tham chiếu thực tiễn lớn cho lĩnh vực IoT Security.

### 🎓 Nếu chấm như một Khóa luận Tốt nghiệp Đại học tại UIT (ĐHQG-HCM):
- **Phán quyết:** **Xuất sắc (Thuộc Top 10% – 15% của khóa)**.
- **Lý do:**
  1. Mặt bằng chung khóa luận thường dừng lại ở: làm web/app CRUD, hoặc lấy một notebook AI mẫu có sẵn chạy trên Jupyter một cách hời hợt, báo cáo các con số ảo 99% không thể tái lập.
  2. Khóa luận này vượt trội ở 4 điểm:
     - Tự tay làm chủ toàn bộ hệ thống phân tán đa tiến trình trên socket gRPC thật.
     - Hiểu sâu bản chất toán học của phân phối Non-IID Dirichlet và hiện tượng Client Drift.
     - Thiết lập hệ thống đo kiểm đa chiều (Utility, Cost, Scalability) trên tập dữ liệu lớn chuẩn quốc tế CICIoT2023.
     - **Dũng cảm báo cáo số liệu thật và phân tích điểm thất bại (Negative Findings)** thay vì giấu khuyết điểm để lấy số đẹp.

---

## 2. Tuyên ngôn Bảo vệ Vàng (The Golden Defense Statement)

> [!IMPORTANT]
> **Lời khuyên đắt giá nhất cho bạn khi đứng trước Hội đồng:**  
> *"Đừng cố gắng phòng thủ rằng mô hình của em hoàn hảo hay không có tì vết. Hãy tự tin tuyên bố rằng: **Giá trị lớn nhất của khóa luận này là chỉ ra chính xác các rào cản kỹ thuật định lượng khi triển khai Federated Learning trên mạng IoT thực tế: từ chi phí băng thông, ngưỡng chịu đựng Non-IID, đến sự bất khả thi của việc phát hiện tấn công cực hiếm nếu chỉ dựa vào thuật toán tổng hợp thông thường.** Đó mới là phong thái của một kỹ sư và một nhà nghiên cứu thực thụ!"*

---

## 3. Bộ Câu hỏi "Hóc búa" Thường Gặp & Kịch bản Trả lời Mẫu (Q&A Script)

Dưới đây là 5 câu hỏi kinh điển mà các Giảng viên phản biện khó tính tại UIT và các hội đồng chuyên ngành chắc chắn sẽ chất vấn:

---

### ❓ Câu hỏi 1: "Tại sao đề tài không đề xuất một thuật toán tối ưu mới mà chỉ sử dụng FedAvg và FedProx có sẵn?"

* **Gợi ý Kịch bản Trả lời:**
  > *"Kính thưa Thầy/Cô, mục tiêu cốt lõi của khóa luận được định vị là một **Nghiên cứu Thực nghiệm và Đo kiểm Hệ thống (Empirical Benchmarking & Prototype Study)**. 
  > 
  > Trong nghiên cứu khoa học, trước khi đề xuất một giải thuật mới, cộng đồng cần một hệ thống đo kiểm cơ sở (baseline) thật sự chuẩn mực và trung thực để hiểu rõ các giới hạn vật lý của các thuật toán tiêu chuẩn. Rất nhiều công trình trước đây vội vã đưa ra thuật toán mới nhưng lại đánh giá trên dữ liệu mô phỏng không thực tế hoặc che giấu các lớp thiểu số. 
  > 
  > Đề tài này giải quyết khoảng trống đó: xây dựng một prototype gRPC hoàn chỉnh và đo kiểm toàn diện FedAvg/FedProx dưới nhiều mức độ Dirichlet khác nhau, tạo ra một bộ chỉ số tham chiếu (benchmark baseline) đáng tin cậy cho các nghiên cứu cải tiến giải thuật tiếp theo của khoa."*

---

### ❓ Câu hỏi 2: "F1-score chỉ đạt 68% và Recall của lớp Web-based rớt thảm hại xuống 0.39% - 1.66%, hệ thống này làm sao triển khai thực tế được?"

* **Gợi ý Kịch bản Trả lời:**
  > *"Kính thưa Thầy/Cô, con số Recall 0.39% của lớp Web-based chính là **phát hiện phản nghiệm (Negative Finding) quan trọng nhất** của khóa luận. 
  > 
  > Về mặt toán học, em đã chứng minh rằng cơ chế lấy trung bình gia quyền theo số lượng mẫu $n_k/N$ của FedAvg và số hạng phạt $\mu$ của FedProx hoàn toàn triệt tiêu tín hiệu gradient của các lớp cực hiếm ($<0.5\%$), dù cục bộ có dùng Focal Loss hay Class Weights.
  > 
  > Về mặt ứng dụng thực tế, đề tài khẳng định hệ thống FL-NIDS không hoạt động đơn lẻ, mà là một thành phần trong kiến trúc **Phòng thủ Chiều sâu (Defense-in-Depth)**. Trong đó:
  > - Tầng ứng dụng L7 cần WAF (như ModSecurity) để bắt các cuộc tấn công Web-based chuyên sâu (SQLi, XSS).
  > - Tầng mạng L3/L4 chính là nơi mô hình FL của em phát huy sức mạnh tối đa, đạt Recall và Accuracy $>90\% - 99\%$ đối với các cuộc tấn công lưu lượng lớn như DDoS, DoS, Reconnaissance, Spoofing, và Mirai Botnet."*

---

### ❓ Câu hỏi 3: "Tại sao lại chạy đa tiến trình gRPC trên cùng một máy tính mà không nạp vào các bo mạch Raspberry Pi vật lý ngoài thực tế?"

* **Gợi ý Kịch bản Trả lời:**
  > *"Kính thưa Thầy/Cô, việc triển khai trên đa tiến trình (`multiprocessing`) thông qua giao thức mạng gRPC socket thật giải quyết hai mục tiêu khoa học quan trọng:
  > 1. **Bảo toàn tính phân tán thực chất:** Các client hoạt động hoàn toàn độc lập về không gian địa chỉ bộ nhớ, chu kỳ sống và tiến trình tính toán, giao tiếp qua nhị phân gRPC y hệt như mạng vật lý (khác biệt hoàn toàn với các mô phỏng gộp mảng in-memory trong Jupyter Notebook).
  > 2. **Cô lập biến số thực nghiệm (Controlled Environment):** Để đo lường chính xác tác động thuần túy của hệ số Non-IID Dirichlet $\alpha$ và số lượng client $K$, môi trường cần loại trừ các yếu tố nhiễu loạn ngẫu nhiên của đường truyền vô tuyến (như rớt gói đột ngột, pin yếu).
  > 
  > Nhờ thiết kế mô hình MLP cực nhẹ (~14k tham số, footprint chỉ 54 KB, tiêu tốn 10.60 MB cho 10 rounds), đề tài đã chứng minh tính khả thi toán học và băng thông. Việc đưa lên cụm Raspberry Pi vật lý hoặc mô phỏng mạng vô tuyến qua Mininet-WiFi là bước chuyển giao tự nhiên được em quy hoạch trong Hướng phát triển của đề tài ạ."*

---

### ❓ Câu hỏi 4: "Tại sao trong thực nghiệm của em, FedProx với $\mu=0.05$ lại cho Macro-F1 thấp hơn FedAvg (59% vs 68% ở IID, 53% vs 64% ở $\alpha=0.5$), có mâu thuẫn với lý thuyết của Li et al. (2020) không?"

* **Gợi ý Kịch bản Trả lời:**
  > *"Kính thưa Thầy/Cô, kết quả này hoàn toàn không mâu thuẫn mà còn làm sâu sắc hơn lý thuyết của Li et al. khi áp dụng vào bài toán an ninh mạng:
  > - Li et al. đề xuất FedProx để hạn chế Client Drift trong bài toán Computer Vision với các lớp dữ liệu cân bằng tương đối. Trong thực nghiệm của em, FedProx đã hoàn thành xuất sắc nhiệm vụ này: **chuẩn khoảng cách trôi mô hình (Drift L2 Norm) giảm hơn 95%** (từ 10.21 xuống 0.52).
  > - Tuy nhiên, trong an ninh mạng IoT, dữ liệu bị mất cân bằng lớp cực đoan. Khi ta áp đặt số hạng phạt $\frac{\mu}{2}\|w - w_t\|^2$, ta đã vô tình 'ghìm chặt' các mô hình client xung quanh mô hình toàn cục, dập tắt hoàn toàn khả năng dịch chuyển gradient của client để học các lớp thiểu số (`Loss of Local Plasticity`). Kết quả là Recall của lớp Web và Brute Force bị kéo thẳng về 0.00%, làm sụt giảm Macro-F1 toàn cục.
  > 
  > Đây là một đánh đổi (trade-off) quan trọng mà các nghiên cứu trước đây chưa chỉ ra rõ."*

---

### ❓ Câu hỏi 5: "Khi dữ liệu Non-IID Dirichlet $\alpha=0.1$, Macro-F1 bị rơi xuống 43%, giải pháp tiếp theo của em là gì?"

* **Gợi ý Kịch bản Trả lời:**
  > *"Kính thưa Thầy/Cô, kết quả $\alpha=0.1$ cho thấy ngưỡng giới hạn vật lý của các phương pháp FL tổng hợp trọng số phẳng (Flat Aggregation). Khi dữ liệu phân cực quá lớn, các client có các bài toán tối ưu đối nghịch nhau, dẫn đến việc lấy trung bình biến thành nhiễu loạn triệt tiêu.
  > 
  > Để khắc phục hiện tượng này, trong tương lai hệ thống cần chuyển dịch sang các hướng tiếp cận tiên tiến hơn:
  > 1. **Personalized Federated Learning (pFL):** Duy trì một phần mạng toàn cục (feature extractor) và cá nhân hóa phần đầu phân loại (classifier head) cho từng client.
  > 2. **Clustered FL (CFL):** Phân cụm các client có phân phối tương đồng để tổng hợp riêng thay vì gộp chung toàn mạng.
  > 3. **Dynamic Loss Rescaling tại Server:** Điều chỉnh hàm mất mát toàn cục dựa trên độ lệch ma trận Hessian của các client để bảo vệ các lớp thiểu số."*

---

## 4. Tóm tắt Kỹ năng Thuyết trình & Trả lời

1. **Luôn lắng nghe hết câu hỏi của Thầy/Cô**, ghi chép từ khóa vào sổ tay, không ngắt lời.
2. **Luôn cảm ơn câu hỏi** và xác nhận câu hỏi chạm đúng vào bản chất cốt lõi của đề tài.
3. **Dẫn chứng bằng số liệu định lượng cụ thể** (ví dụ: *"Dạ ở kịch bản E4 với $\alpha=0.5$, Macro-F1 của FedAvg đạt 64.73% trong khi FedProx đạt 53.01%..."*), tuyệt đối không trả lời chung chung mang tính cảm tính.
4. **Trung thực khoa học:** Nếu Thầy/Cô chỉ ra một khuyết điểm đúng, hãy dũng cảm thừa nhận và trình bày phương án khắc phục thay vì tranh cãi bảo thủ.
