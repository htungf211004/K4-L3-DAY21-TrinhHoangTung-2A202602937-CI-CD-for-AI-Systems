# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

| | |
|---|---|
| Họ và tên | Trịnh Hoàng Tùng |
| MSSV | 2A202602937 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/htungf211004/K4-L3-DAY21-TrinhHoangTung-2A202602937-CI-CD-for-AI-Systems |
| Ngày nộp | Chưa nộp; thực nghiệm Bước 1 ngày 07/10/2026. |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

<!-- Khoảng 120 - 150 từ. Điền kết quả thật từ MLflow UI ở Bước 1, tối thiểu 3 lần chạy. -->

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.710900 | 0.878000 |
| 2 | 50 | 0.05 | 2 | 0.605128 | 0.846000 |
| 3 | 200 | 0.1 | 5 | 0.714932 | 0.874000 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần 3 có F1 cao nhất, vượt ngưỡng 0.65; lần 2 không đạt ngưỡng. Accuracy cao nhất thuộc lần 1, chứng tỏ tối ưu accuracy không đồng nghĩa với tối ưu lớp thu nhập cao. Cấu hình ít cây, learning rate thấp và độ sâu nhỏ đạt F1 thấp hơn; giảm learning rate thường cần tăng số cây để bù lại. Tuy nhiên, ba thí nghiệm thay đổi nhiều tham số cùng lúc nên chưa tách được ảnh hưởng riêng của từng tham số. Cả ba dùng 22.361 mẫu train, 500 mẫu holdout và random_state=42. Số liệu được đối chiếu với MLflow SQLite. Ảnh `01-mlflow-ui.png` do người học chụp hiển thị đủ ba cấu hình, hai độ đo và thứ tự F1 giảm dần.

<!--
Trả lời trong phần Lý do:
  - Vì sao bộ này tốt hơn các bộ còn lại (dựa trên f1_score, không phải accuracy)?
  - Lần chạy có accuracy cao nhất có trùng với lần có f1_score cao nhất không?
    Nếu không, điều đó nói lên điều gì?
  - Bạn quan sát thấy đánh đổi nào giữa n_estimators và learning_rate?
-->

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

<!-- Khoảng 120 - 150 từ. -->

Holdout có 24.8% mẫu thu nhập trên 50K, còn lại là lớp thu nhập thấp. Dự đoán toàn bộ là thu nhập thấp vẫn đạt accuracy 0.752 nhưng F1 lớp dương bằng 0, vì không nhận diện được người thu nhập cao nào. F1 kết hợp precision và recall của lớp dương, phản ánh cả dự đoán nhầm và bỏ sót lớp này. Vì vậy quality gate dùng F1 >= 0.65, còn accuracy chỉ để tham khảo. Mã gọi `f1_score(y_eval, preds)` với mặc định `pos_label=1`. Weighted F1 bị chi phối bởi lớp đa số; macro F1 cho hai lớp trọng số bằng nhau nhưng vẫn đo trung bình hai lớp, khác mục tiêu F1 riêng cho lớp dương của lab.

<!--
Cần nêu được:
  - Phân bố lớp của tập dữ liệu (tỷ lệ lớp thu nhập > 50K) và hệ quả của nó.
  - Accuracy của một mô hình luôn trả lời "thu nhập thấp" là bao nhiêu, vì sao con số
    đó gây hiểu nhầm.
  - F1 của lớp dương đo điều gì mà accuracy không đo được.
  - Vì sao KHÔNG dùng average="weighted" hay average="macro" khi gọi f1_score.
-->

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

<!-- Nêu 2 - 3 khó khăn thật, mỗi ô một câu ngắn. -->

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow không mở được SQLite. | SQLAlchemy 2.1.3 bỏ lớp MLflow 2.13 đang import. | Cài và ghim SQLAlchemy 2.0.30. |
| Tải dependency bị timeout. | Đường truyền tới PyPI chậm. | Thử lại với timeout 120 giây; cài thành công. |
| Chọn cấu hình EC2 tiết kiệm và cấp quyền S3 phù hợp. | Nhãn Free Tier không có nghĩa mọi tài khoản được miễn phí vô hạn. | Kiểm tra Free Plan và credits trên Console; dùng t3.micro CPU Standard, gp3 mã hóa và role chỉ đọc đúng model. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | ___ | ___ |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___
