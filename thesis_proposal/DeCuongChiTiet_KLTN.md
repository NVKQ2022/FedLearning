CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT

ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG

NAM

Độc Lập - Tự Do - Hạnh Phúc

TIN

KHOA MẠNG MÁY TÍNH VÀ TRUYỀN

THÔNG

## ĐỀ CƯƠNG CHI TIẾT KHÓA LUẬN TỐT NGHIỆP

## HỌC KỲ I NĂM HỌC 2026 – 2027

## Tên đề tài:

- \- Tên tiếng Việt: Xây dựng và đánh giá prototype hệ thống phát hiện xâm nhập IoT sử dụng Federated Learning trong môi trường dữ liệu non-IID

- \- Tên tiếng Anh: Development and evaluation of an IoT intrusion detection system prototype using Federated Learning in non-IID data environments

Cán bộ hướng dẫn: PGS.TS Lê Trung Quân

Thời gian thực hiện: Từ ngày 15/09/2026 đến ngày 17/12/2026

Sinh viên thực hiện:

- 1. Nguyễn Việt Kỳ Quân – 23521267 – 0921124475

## Nội dung đề tài:

## 1. Bối cảnh và lý do chọn đề tài

Sự phát triển của Internet of Things (IoT) làm gia tăng nhanh chóng số lượng thiết bị kết nối và lưu lượng mạng, đồng thời mở rộng bề mặt tấn công. Các hệ thống Intrusion Detection System (IDS) dựa trên học máy và học sâu có thể được sử dụng để phát hiện và phân loại lưu lượng bất thường trong môi trường IoT.


Trong mô hình học tập tập trung, dữ liệu từ nhiều nguồn thường được thu thập về một nơi để huấn luyện. Federated Learning (FL) cho phép các client giữ dữ liệu tại chỗ và chỉ trao đổi model updates với server, qua đó phù hợp với các hệ thống có dữ liệu phân tán. Tuy nhiên, dữ liệu thực tế tại các client thường không đồng nhất (Non-IID), làm ảnh hưởng đến quá trình hội tụ và chất lượng mô hình.

Đề tài tập trung xây dựng một prototype FL-IDS có nhiều client giả lập, trong đó dữ liệu được chia từ dataset IoT thành các local datasets có phân phối khác nhau. Các client sử dụng cùng một mô hình MLP cho dữ liệu network-flow dạng bảng; server thực hiện aggregation bằng FedAvg và FedProx. Định hướng của đề tài ưu tiên triển khai và thực nghiệm, thay vì tập trung vào việc đề xuất một thuật toán aggregation hoàn toàn mới.

## 2. Đối tượng nghiên cứu

- \- Phát hiện và phân loại lưu lượng mạng trong môi trường IoT: Nghiên cứu khả năng nhận diện lưu lượng mạng thông thường và các loại lưu lượng/tấn công bất thường trong hệ thống IoT.

- \- Learning với dữ liệu phân tán: Nghiên cứu cơ chế huấn luyện mô hình trên nhiều client có dữ liệu riêng biệt, trong đó dữ liệu không cần tập trung về một máy chủ duy nhất.

- \- Ảnh hưởng của dữ liệu Non-IID: Đánh giá tác động của sự khác biệt về phân bố dữ liệu giữa các client đến quá trình hội tụ và hiệu năng của mô hình Federated Learning.

- \- Mô hình MLP cho dữ liệu Network Flow: Sử dụng mô hình Multi-Layer Perceptron (MLP) để học và phân loại dữ liệu network-flow dạng bảng.

- \- Phương pháp FedAvg và FedProx: Nghiên cứu, triển khai và so sánh hiệu quả của hai phương pháp tổng hợp mô hình FedAvg và FedProx trong môi trường dữ liệu phân tán.

- \- Chia dữ liệu IID và Non-IID: Xây dựng các kịch bản phân chia dữ liệu giữa các client theo IID và Non-IID, sử dụng phân phối Dirichlet để kiểm soát mức độ không đồng nhất của dữ liệu.

- \- Kiến trúc hệ thống prototype: Xây dựng hệ thống gồm bốn lớp chính: Data/Partition Layer, Federated Client Layer, Federated Server/Aggregator Layer và Evaluation/Monitoring Layer, phục vụ quá trình phân chia dữ liệu, huấn luyện, tổng hợp và đánh giá mô hình.


## Mục tiêu của đề tài:

## 1. Mục tiêu tổng quan

Xây dựng một prototype hệ thống FL-IDS có thể triển khai và chạy thực nghiệm trên nhiều client giả lập, trong đó dữ liệu được phân chia theo các điều kiện IID và Non-IID. Đề tài đánh giá hoạt động của FedAvg và FedProx thông qua hiệu năng mô hình, khả năng hội tụ, thời gian huấn luyện, chi phí truyền thông và mức sử dụng tài nguyên.

## 2. Mục tiêu cụ thể

- \- Khảo sát CICIoT2023 và một số dataset IoT/IIoT gần đây từ CIC/UNB.

- \- Xây dựng pipeline tiền xử lý dữ liệu và centralized MLP baseline.

- \- Xây dựng local client gồm dataset, preprocessing, dataloader, MLP, trainer và local evaluation.

- \- Xây dựng federated server gồm client selection, nhận model update, aggregation, broadcast và logging.

- \- Phân vùng dữ liệu theo IID và Dirichlet Non-IID với α = 1.0, 0.5 và 0.1.

- \- Tích hợp FedAvg và FedProx trong cùng một framework triển khai để so sánh.

- \- Xây dựng module evaluation/monitoring theo dõi metrics mô hình và metrics hệ thống.

- \- Đánh giá các kịch bản về mức độ Non-IID, số lượng client và cấu hình huấn luyện.

- \- Đo Accuracy, Precision, Recall, F1, Macro-F1, Minority Recall, Confusion Matrix, convergence, training time, communication cost và CPU/RAM/GPU memory.

- \- Đóng gói hệ thống bằng Docker/process để hỗ trợ tái lập thí nghiệm.

## Phương pháp thực hiện:

## 1. Tổng quan phương pháp

Khóa luận xây dựng prototype hệ thống phát hiện xâm nhập cho mạng IoT dựa trên Federated Learning. Trong hệ thống, dữ liệu được lưu trữ và huấn luyện cục bộ tại các client. Client chỉ gửi tham số mô hình về server để tổng hợp, thay vì chia sẻ dữ liệu gốc.


Quy trình thực hiện gồm các bước:

- 1. Tiền xử lý tập dữ liệu CICIoT2023.

- 2. Xây dựng mô hình MLP tập trung làm baseline.

- 3. Phân chia dữ liệu cho các client theo IID và non-IID.

- 4. Xây dựng kiến trúc Federated Learning.

- 5. Triển khai FedAvg và FedProx.

- 6. Thiết lập các kịch bản thực nghiệm.

- 7. Đánh giá hiệu năng mô hình và tài nguyên hệ thống.

## 2. Tập dữ liệu và tiền xử lý

## 2.1. Tập dữ liệu

Khóa luận sử dụng tập dữ liệu CICIoT2023, gồm lưu lượng mạng được thu thập từ các thiết bị IoT với nhiều nhóm tấn công như DDoS, DoS, Recon, Web-based, Brute Force, Spoofing và Mirai. Dữ liệu dạng CSV được lựa chọn vì phù hợp với mô hình MLP và thuận tiện cho quá trình xử lý.

Khảo sát thêm về các các dataset gần đây trên CIC/UNB. Các dataset này chủ yếu được dùng để cập nhật landscape và chỉ một dataset phụ được đưa vào thực nghiệm nếu còn thời gian. CIC IoT-DIAD 2024 hỗ trợ cả anomaly detection và device identification, với flow-based và packet-based features; CICIoMT2024 tập trung vào IoMT với Wi-Fi, MQTT và Bluetooth; CIC-BCCC-NRC TabularIoTAttack-2024 cung cấp dữ liệu tabular IoT được tổng hợp từ nhiều nguồn [6][7][8].


*Bảng 2.1: Tổng quan các bộ dataset*

| Dataset | Đặc điểm chính | Vai trò trong khóa luận |
| --- | --- | --- |
| CICIoT2023 [1] | 105 thiết bị, 33 attacks, 7 | Dataset chính; toàn bộ |
|   | attack categories, PCAP + | core experiments |
|   | CSV |   |
| CIC IoT-DIAD 2024 [5] | Device identification + | Khảo sát; tùy chọn dataset |
|   | anomaly detection; | phụ |
|   | flow/packet features |   |
| CICIoMT2024 [6] | IoMT; Wi-Fi, MQTT, | Khảo sát; tùy chọn dataset |
|   | Bluetooth; 18 attacks | phụ |
| CIC-BCCC-NRC | Tabular IoT traffic; tổng | Khảo sát và ứng viên kiểm |
| TabularIoTAttack-2024 [7] | hợp từ nhiều nguồn | chứng mở rộng |

## 2.2. Tiền xử lý dữ liệu

Quy trình tiền xử lý gồm:

- \- Loại bỏ dữ liệu trùng lặp và các giá trị không hợp lệ.

- \- Xử lý giá trị thiếu.

- \- Loại bỏ các thuộc tính không cần thiết hoặc có nguy cơ làm rò rỉ nhãn.

- \- Mã hóa các thuộc tính dạng phân loại.

- \- Chuẩn hóa các đặc trưng số.

- \- Chia dữ liệu thành tập train, validation và test.

Các tham số chuẩn hóa chỉ được xác định từ tập train, sau đó áp dụng thống nhất cho validation, test và dữ liệu của các client. Tập test toàn cục được giữ cố định và không tham gia quá trình huấn luyện.


## 3. Xây dựng mô hình MLP cục bộ

Một mô hình MLP tập trung được xây dựng trước để kiểm tra quy trình tiền xử lý và làm baseline so sánh với Federated Learning. Mô hình tập trung và mô hình tại các client sử dụng cùng kiến trúc.

Centralized MLP và local MLP phải dùng cùng feature set, architecture, loss/optimizer protocol và train/test convention để baseline có ý nghĩa.

Các mô hình Logistic Regression, Random Forest hoặc XGBoost chỉ được khảo sát ở centralized baseline nếu cần. Không mở rộng nhiều kiến trúc local DL vì điều đó không phục vụ mục tiêu implementation FL + non-IID.

*Bảng 3 : Kiến trúc MLP tại mỗi client*

| Thành phần | Thiết kế dự kiến |
| --- | --- |
| Input | Số lượng network-flow features sau preprocessing |
| Hidden 1 | Linear(input_dim, 128) + ReLU |
| Regularization | Dropout |
| Hidden 2 | Linear(128, 64) + ReLU |
| Output | Linear(64, number_of_classes) |
| Loss | Cross-Entropy hoặc weighted variant nếu cần xử lý imbalance |
| Optimizer | Adam/SGD; cố định trong các baseline |

Hàm Cross-Entropy được sử dụng cho bài toán phân loại đa lớp. Trong mỗi communication round, client nhận global model từ server, huấn luyện trên dữ liệu cục bộ và gửi các tham số mô hình đã cập nhật về server.


## 4. Phân chia dữ liệu IID và non-IID

## 4.1. Phân chia IID

Trong trường hợp IID, dữ liệu được phân chia sao cho các client có phân phối nhãn gần tương đương nhau. Kịch bản này được sử dụng để kiểm tra hoạt động của hệ thống và làm cơ sở so sánh với dữ liệu non-IID.

## 4.2. Phân chia non-IID

Dữ liệu non-IID được tạo bằng phân phối Dirichlet:

Trong đó, pc biểu diễn tỷ lệ mẫu của lớp ccc được phân bổ cho các client. Khóa luận sử dụng các giá trị:

- \- α=1.0 : mức độ không đồng nhất thấp.

- \- α=0.5 : mức độ không đồng nhất trung bình.

- \- α=0.1 : mức độ không đồng nhất cao.

Giá trị α càng nhỏ thì dữ liệu của mỗi lớp càng tập trung vào một số client. Quá trình phân chia được kiểm tra để bảo đảm mỗi client có đủ số lượng mẫu tối thiểu, không trùng dữ liệu và không sử dụng tập test cho huấn luyện.


*Hình 4.2. Phân phối nhãn tại các client theo IID và Dirichlet non-IID.*

## 5. Kiến trúc hệ thống FL-IDS

Prototype FL-IDS gồm bốn thành phần chính:

- 1. Data and Partition Layer: Tiền xử lý và phân chia dữ liệu.

- 2. Federated Client Layer: Huấn luyện mô hình trên dữ liệu cục bộ.

- 3. Federated Server Layer: Điều phối client và tổng hợp tham số.

- 4. Evaluation and Monitoring Layer: Đánh giá mô hình và theo dõi tài nguyên.

Client không gửi dữ liệu gốc về server mà chỉ gửi tham số mô hình, số lượng mẫu và thông tin huấn luyện. Server chịu trách nhiệm khởi tạo global model, gửi mô hình cho client, tổng hợp kết quả và đánh giá mô hình sau mỗi round.


*Hình 5. Kiến trúc tổng thể của prototype FL-IDS.*

## 6. Phương pháp FedAvg và FedProx

## 6.1. FedAvg

FedAvg tổng hợp các mô hình cục bộ theo số lượng mẫu của từng client:

*Hình 6.1: Công thức FedAvg*

nk : là số lượng mẫu của client k. Client có nhiều dữ liệu hơn sẽ đóng góp trọng số lớn hơn vào global model.

wt+1: là trọng số tiếp theo của model sau khi tổng hợp từ các client

wt+1 k: là trọng số của model của client thứ k

N: là tổng số mẫu dữ liệu của tất cả các client


K: là tổng số client tham gia vào quá trình huấn luyện

FedAvg được sử dụng làm phương pháp baseline. Tuy nhiên, trong môi trường non-IID, mô hình có thể hội tụ chậm hoặc dao động do sự khác biệt giữa dữ liệu của các client.

## 6.2. FedProx

FedProx bổ sung một thành phần proximal vào hàm mất mát cục bộ :

*Hình 6.2 : Công thức promaximal regularization*

Trong đó, μ là hệ số proximal và wt là global model tại đầu round. Thành phần này hạn chế local model thay đổi quá xa global model, từ đó hỗ trợ huấn luyện trong điều kiện dữ liệu không đồng nhất.

Khi so sánh FedAvg và FedProx, các thông số như kiến trúc mô hình, learning rate, batch size, local epoch, số client và dữ liệu phân chia được giữ cố định.

## 7. Triển khai prototype

Prototype được phát triển bằng Python, sử dụng PyTorch để xây dựng mô hình MLP và Flower để triển khai Federated Learning. Các client được giả lập bằng nhiều tiến trình hoặc Docker container trên cùng một máy tính.

Hệ thống gồm các module chính:

- \- Đọc và tiền xử lý dữ liệu.

- \- Phân chia dữ liệu IID và non-IID.

- \- Xây dựng mô hình MLP.


- \- Huấn luyện mô hình cục bộ.

- \- Điều phối server và client.

- \- Tổng hợp mô hình.

- \- Đánh giá và ghi log kết quả.

Quá trình triển khai bắt đầu từ mô hình tập trung, sau đó xây dựng FedAvg với dữ liệu IID. Khi hệ thống hoạt động ổn định, FedProx và các kịch bản non-IID được tích hợp.

Mỗi thí nghiệm lưu lại seed, số client, số round, learning rate, batch size, local epoch, giá trị α, hệ số μ và cấu hình phần cứng để bảo đảm khả năng tái lập.

*Bảng 7. Công nghệ sử dụng trong prototype.*

| Thành phần | Công nghệ dự kiến |
| --- | --- |
| Language | Python |
| Deep Learning | PyTorch [11] |
| Federated Learning | Flower [10] |
| Container/simulation | Docker hoặc process-based |
|   | Python logging dashboard đơn giản |
| Monitoring |   |
|   | nếu cần |
| Versioning | Git |

## 8. Kịch bản thực nghiệm

Các kịch bản thực nghiệm gồm:

- \- E1: MLP tập trung làm baseline.

- \- E2: FedAvg và FedProx với dữ liệu IID.

- \- E3: Dữ liệu non-IID với α=1.0.

- \- E4: Dữ liệu non-IID với α=0.5.

- \- E5: Dữ liệu non-IID với α=0.1.

- \- E6: Thay đổi số lượng client lần lượt là 5, 7 và 10.


Các kịch bản sử dụng cùng tập test, kiến trúc MLP và quy trình tiền xử lý. Nếu tài nguyên cho phép, mỗi cấu hình được chạy nhiều lần với các seed khác nhau. Kết quả được trình bày bằng giá trị trung bình và độ lệch chuẩn.

*Bảng 8: Tổng hợp các kịch bản thực nghiệm.*

| Experiment Thiết lập |   | Mục đích |
| --- | --- | --- |
| E1 | Centralized MLP | Baseline hiệu năng không có FL |
| E2 | IID FL: FedAvg/FedProx Kiểm tra implementation FL trong điều |   |
|   |   | kiện dễ |
| E3 | Dirichlet α=1.0 | Đánh giá bước đầu khi có heterogeneity |
| E4 | Dirichlet α=0.5 | Đánh giá suy giảm/khả năng thích nghi |
| E5 | Dirichlet α=0.1 | Đánh giá trường hợp heterogeneity cao |
| E6 | 5/7/10 clients | Đánh giá tác động của số lượng client |
| E7 | Ablation tùy chọn | Chỉ khi có proposed aggregation hoặc cần |
|   |   | kiểm tra một component |

## 9. Chỉ số đánh giá

Hiệu quả mô hình được đánh giá bằng các chỉ số:


*Bảng 9: Các chỉ số đánh giá mô hình*

| Metric | Ý nghĩa |
| --- | --- |
| Accuracy | Hiệu năng tổng thể |
| Precision / Recall / F1 | Đánh giá theo lớp/overall |
| Macro-F1 | Không để lớp lớn chi phối hoàn toàn |
|   | kết quả |
| Minority Recall | Theo dõi khả năng nhận diện các lớp |
|   | có ít mẫu |
| Confusion Matrix | Phân tích nhầm lẫn giữa các attack |
|   | classes |

Trong đó, Macro-F1 và Minority Recall được ưu tiên vì tập dữ liệu có thể mất cân bằng giữa các lớp. Confusion Matrix được sử dụng để xác định các lớp thường bị mô hình phân loại nhầm.

Ngoài hiệu năng phân loại, hệ thống còn được đánh giá theo:

*Bảng 9: Các chỉ số đánh giá hệ thống*

| Metric | Ý nghĩa |
| --- | --- |
| Convergence rounds | Số round đạt target performance hoặc |
|   | dừng theo criterion |
| Training time | Local, aggregation và total wall-clock |
|   | time |
| Communication cost | Bytes upload/download và tổng bytes |
| CPU/RAM/GPU memory | Tài nguyên sử dụng |
| Scalability | Ảnh hưởng của số clients đến thời |
|   | gian/cost/resource |


Chi phí truyền thông có thể được ước lượng theo công thức:

Hình : Công thức tính chi phí truyền thông

Trong đó, mt là số client tham gia round t, B là kích thước mô hình và T là tổng số communication round.

## 10. Tổng kết phương pháp

Phương pháp thực hiện tập trung vào việc xây dựng prototype FL-IDS sử dụng mô hình MLP và tập dữ liệu CICIoT2023. Dữ liệu được phân chia theo IID và Dirichlet non-IID với các giá trị α=1.0, 0.5và 0.1.

FedAvg được sử dụng làm baseline, trong khi FedProx được sử dụng để đánh giá khả năng cải thiện quá trình huấn luyện trên dữ liệu không đồng nhất. Các thí nghiệm được thực hiện với 5, 7 và 10 client, sau đó đánh giá dựa trên hiệu năng phân loại, tốc độ hội tụ, chi phí truyền thông, thời gian huấn luyện và tài nguyên hệ thống.

## Giới hạn của đề tài:

Đề tài tập trung xây dựng và đánh giá prototype FL-IDS trong môi trường mô phỏng với 5-10 client trên cùng một máy hoặc môi trường thí nghiệm, chưa triển khai trên hệ thống IoT thực tế. Nghiên cứu sử dụng chủ yếu dataset CICIoT2023 và mô hình MLP nhằm tập trung đánh giá ảnh hưởng của dữ liệu non-IID đến Federated Learning. Các vấn đề nâng cao như Differential Privacy, Secure Aggregation, Byzantine Attack hay Model Poisoning chưa được xem xét. Ngoài ra, đề tài không hướng đến tối ưu độ chính xác phát hiện xâm nhập ở mức state-of-the-art mà tập trung vào khả năng triển khai, đánh giá và tái lập hệ thống.


## Hướng phát triển

Trong tương lai, hệ thống có thể được mở rộng theo nhiều hướng. Thứ nhất, triển khai và đánh giá trên các thiết bị IoT hoặc edge device thực tế nhằm kiểm chứng tính khả thi trong môi trường sản xuất. Thứ hai, thử nghiệm trên nhiều bộ dữ liệu IoT/IIoT khác để đánh giá khả năng tổng quát hóa của mô hình. Thứ ba, nghiên cứu các kỹ thuật tăng cường bảo mật và quyền riêng tư như Differential Privacy, Secure Aggregation hoặc cơ chế chống tấn công poisoning. Cuối cùng, có thể mở rộng sang các phương pháp Federated Learning tiên tiến hơn như Personalized Federated Learning, Adaptive Aggregation hoặc sử dụng các kiến trúc học sâu mạnh hơn như CNN, LSTM và Transformer để nâng cao hiệu quả phát hiện xâm nhập trong môi trường dữ liệu non-IID.

## Kế hoạch thực hiện:

| Thời gian | Nội dung |
| --- | --- |
| Tháng 1 – Tuần 1 | Khảo sát FL, IDS, non-IID và dataset CIC/UNB; chốt scope |
| Tháng 1 – Tuần 2 | Phân tích CICIoT2023, preprocessing và data analysis |
| Tháng 1 – Tuần 3 | Xây dựng centralized MLP baseline |
| Tháng 1 – Tuần 4 | Xây dựng local trainer và client interface |
| Tháng 2 – Tuần 1 | Xây dựng Flower server/client communication |
| Tháng 2 – Tuần 2 | Tích hợp FedAvg |
| Tháng 2 – Tuần 3 | Tích hợp FedProx |
| Tháng 2 – Tuần 4 | Xây dựng IID partition và kiểm tra hệ thống |
| Tháng 3 – Tuần 1 | Xây dựng Dirichlet non-IID với α = 1.0, 0.5, 0.1 |
| Tháng 3 – Tuần 2 | Chạy core experiments và repeated runs |
| Tháng 3 – Tuần 3 | Đánh giá convergence, Macro-F1, Minority Recall |
| Tháng 3 – Tuần 4 | Đo training time, communication cost và resource usage |
| Tháng 4 – Tuần 1 | Thực nghiệm 5/7/10 clients |
| Tháng 4 – Tuần 2 | Optional dataset / optional aggregation / monitoring |
| Tháng 4 – Tuần 3 | Phân tích kết quả và viết nội dung chính của khóa luận |
| Tháng 4 – Tuần 4 | Hoàn thiện khóa luận, slide và demo |

## Tài liệu tham khảo:


- [1] E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani, “CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment,” Sensors, 2023.

- [2] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS, 2017.

- [3] T. Li, A. K. Sahu, A. Talwalkar, V. Smith, “Federated Optimization in Heterogeneous Networks,” MLSys, 2020.

- [4] W. Huang, T. Tiropanis, G. Konstantinidis, “Federated learning-based IoT intrusion detection on non-IID data,” Internet of Things. GIoTS 2022, Lecture Notes in Computer Science, Springer, 2022.

- [6] M. Rabbani, J. Gui, F. Nejati, Z. Zhou, A. Kaniyamattam, M. Mirani, G. Piya, I. Opushnyev, R. Lu, A. A. Ghorbani, “Device Identification and Anomaly Detection in IoT Environments,” IEEE Internet of Things Journal, 2024. Dataset: CIC IoT-DIAD 2024.

- [7] S. Dadkhah, E. C. P. Neto, R. Ferreira, R. C. Molokwu, S. Sadeghi, A. A. Ghorbani, “CICIoMT2024: Attack Vectors in Healthcare Devices – A Multi-Protocol Dataset for Assessing IoMT Device Security,” Internet of Things, 2024.

- [8] T. Sasi, A. Habibi Lashkari, R. Lu, P. Xiong, S. Iqbal, “An Efficient Self Attention-Based 1D-CNN-LSTM Network for IoT Attack Detection and Identification Using Network Traffic,” Journal of Information and Intelligence, 2024. Dataset: CIC-BCCC-NRC TabularIoTAttack-2024.

- [5] “A Review of Federated Learning Applications in Intrusion Detection Systems,” Computer Networks, 2024.

- [9] Canadian Institute for Cybersecurity, “CIC Datasets,” University of New Brunswick. https://www.unb.ca/cic/datasets/ [URL 🔗](https://www.unb.ca/cic/datasets/)

- [10] Beutel et al., Flower: A Friendly Federated Learning Research Framework.

TP. HCM, ngày 15 tháng 09 năm 2026

Giảng viên hướng dẫn

Sinh viên

(Ký và ghi rõ họ tên)

(Ký và ghi rõ họ tên)


PGS.TS Lê Trung Quân

Nguyễn Việt Kỳ Quân
