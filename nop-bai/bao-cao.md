# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Trịnh Hoàng Tùng |
| MSSV | 2A202602937 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/htungf211004/K4-L3-DAY21-TrinhHoangTung-2A202602937-CI-CD-for-AI-Systems |
| Ngày nộp | Chưa nộp; thực nghiệm ngày 07/10/2026. |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.710900 | 0.878000 |
| 2 | 50 | 0.05 | 2 | 0.605128 | 0.846000 |
| 3 | 200 | 0.1 | 5 | 0.714932 | 0.874000 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần 3 có F1 cao nhất, vượt ngưỡng 0.65; lần 2 không đạt ngưỡng. Accuracy cao nhất thuộc lần 1, chứng tỏ tối ưu accuracy không đồng nghĩa với tối ưu lớp thu nhập cao. Cấu hình ít cây, learning rate thấp và độ sâu nhỏ đạt F1 thấp hơn; giảm learning rate thường cần tăng số cây để bù lại. Tuy nhiên, ba thí nghiệm thay đổi nhiều tham số cùng lúc nên chưa tách được ảnh hưởng riêng của từng tham số. Cả ba dùng 22.361 mẫu train, 500 mẫu holdout và random_state=42. Số liệu được đối chiếu với MLflow SQLite và ảnh người học chụp.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Holdout có 24.8% mẫu thu nhập trên 50K. Dự đoán toàn bộ là thu nhập thấp vẫn đạt accuracy 0.752 nhưng F1 lớp dương bằng 0, vì không nhận diện được người thu nhập cao nào. F1 kết hợp precision và recall của lớp dương, phản ánh cả dự đoán nhầm và bỏ sót lớp này. Vì vậy quality gate dùng F1 >= 0.65, còn accuracy chỉ để tham khảo. Mã gọi `f1_score(y_eval, preds)` với mặc định `pos_label=1`. Weighted F1 bị chi phối bởi lớp đa số; macro F1 cho hai lớp trọng số bằng nhau nhưng vẫn đo trung bình hai lớp, khác mục tiêu F1 riêng cho lớp dương của lab.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow không mở được SQLite. | SQLAlchemy 2.1.3 bỏ lớp MLflow 2.13 đang import. | Cài và ghim SQLAlchemy 2.0.30. |
| Tải dependency bị timeout. | Đường truyền tới PyPI chậm. | Thử lại với timeout 120 giây; cài thành công. |
| GitHub từ chối push dữ liệu. | Máy chủ trả Internal Server Error. | Giữ nguyên commit, thử lại thành công; không ghép batch lần nữa. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.714932 | 0.874000 |
| Bước 3 (thêm `train_batch2`) | 0.735426 | 0.882000 |

**Nhận xét:** Số liệu lấy từ `report.json` trong Artifacts của run #1 và #2, làm tròn sáu chữ số; F1 tăng 0.020494 và accuracy tăng 0.008 khi train tăng từ 22.361 lên 44.722 mẫu. Hai batch cùng nguồn và phân phối nên mức cải thiện trên holdout còn hạn chế; kết quả này không chứng minh thêm dữ liệu luôn tốt hơn. Commit dữ liệu `f47b291` tự động chạy thành công cả bốn job, qua ngưỡng 0.65 và triển khai lại; model trên VM đổi mã băm, khớp S3, `/healthz` trả `ok` và `/score` trả nhãn hợp lệ.
