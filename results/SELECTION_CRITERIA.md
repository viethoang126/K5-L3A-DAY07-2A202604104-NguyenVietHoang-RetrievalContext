# Tiêu chí chọn cấu hình agent (chốt trước khi xem kết quả agent trên dev)

**Ngày chốt:** 2026-11-17, lúc các lượt chạy agent trên dev (A0, A1, A2) vừa bắt đầu, chưa đọc trace nào. Chưa có file nào trong `results/agent/test/`.

Chỉ dùng **dev** (15 câu có đáp án + 2 câu thiếu bằng chứng). Xét lần lượt; chỉ sang tiêu chí sau khi hai cấu hình bằng nhau ở tiêu chí trước. Chênh lệch dưới 1 câu được coi là bằng nhau.

1. **Grounded answer** (chấm thủ công 0/1/2, tổng điểm trên 15 câu). Cao hơn thắng.
2. **Abstention**: số câu thiếu bằng chứng từ chối đúng (n/2) trừ số câu có đáp án bị từ chối nhầm. Cao hơn thắng.
3. **Context Evidence Rate** của context cuối (n/15). Cao hơn thắng.
4. **Citation hợp lệ** (citation có trong context và hỗ trợ câu trả lời / tổng citation). Cao hơn thắng.
5. **Chi phí và thời gian**: ít token hơn, rồi median thời gian/câu thấp hơn thắng.

Quy tắc:
- Kết quả test chỉ để báo cáo, không dùng để chọn lại cấu hình hay sửa prompt, code, câu hỏi.
- Sau khi chọn, chạy **baseline A0** và **cấu hình được chọn** trên test với cùng model (`gemini-3.5-flash-lite`), cùng corpus và index.

**Ghi chú 2026-11-17:** lượt chạy đầu với `gemini-3.5-flash` (Google phục vụ bằng `gemini-3.6-flash`) dừng ở câu thứ 7 vì free tier chỉ cho 20 request/ngày. Lượt này được lưu tại `results/pilot/`, không dùng để so sánh. Toàn bộ dev và test chạy lại bằng `gemini-3.5-flash-lite`; tiêu chí ở trên giữ nguyên.

## Áp dụng trên dev (agent v2, `gemini-3.5-flash-lite`) — chốt trước khi chạy test

Thời điểm chốt: 2026-11-17, trước khi chạy bất kỳ câu test nào (chưa có `results/agent/test/`).

| Cấu hình | Grounded (tổng /30) | Abstention đúng − từ chối nhầm | Context Evidence | Citation hợp lệ | Token in+out |
|---|---|---|---|---|---|
| A0_dense_single | 30 | 2 − 0 | 15/15 | 17/17 | 44 936 + 4 487 |
| A1_dense_multi | 29 | 2 − 0 | 15/15 | 19/20 | 55 659 + 4 877 |
| A2_rerank_multi | 30 | 2 − 0 | 15/15 | 19/19 | 59 967 + 5 453 |

- Tiêu chí 1: A0 = A2 (30), A1 thua 1 câu (q10 chấm 1) → loại A1.
- Tiêu chí 2–4: A0 = A2.
- Tiêu chí 5: A0 ít token hơn (49,4k so với 65,4k) → **chọn A0_dense_single**.

Vì cấu hình được chọn trùng với baseline, trên test chạy **A0** (baseline = cấu hình chọn) và thêm **A2** (á quân) **chỉ để báo cáo so sánh**; kết quả test không được dùng để chọn lại.
