# Tiến độ AutoCheckout-CL

File này ghi tiến độ implement theo [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (v1.3). Mỗi khi xong hoặc bắt đầu một task thì cập nhật bảng và thêm một dòng vào nhật ký. Hạn credit GCP: **24/10/2026**.

Trạng thái: **Xong** = đạt tiêu chí nghiệm thu trong plan; **Đang làm**; **Chờ** = bị chặn, ghi rõ chờ gì; **Chưa** = chưa bắt đầu. Cột "Kiểm chứng" ghi test hoặc lệnh đã chạy để xác nhận.

## Việc đang chờ nhóm

| Việc | Ai | Ghi chú |
|---|---|---|
| Tạo Kaggle API token (`kaggle.json`) và cho biết đường dẫn trên Mac | Nhóm | Cần cho T0.5; file chỉ copy thẳng lên VM, không commit |
| Quyết định giữ hay xóa 14 snapshot ổ boot của VM cũ (≈ 30 GB, ≈ 40 nghìn VND/tháng) | Nhóm | VM và 2 ổ đã xóa ngày 28/09 |
| QĐ-5: demo webcam có nằm trong phạm vi không | Nhóm | Cần trước giai đoạn demo |

## Bảng trạng thái

### Giai đoạn 0: nền tảng

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| T0.1 | Xong (28/09) | Git local (QĐ-7). `pdp/` giống hệt blob upstream `7702d91` (so bằng `git hash-object`); bỏ `__pycache__` của upstream |
| T0.2 | Đang làm | `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.venv` trên Mac |
| T0.3 | Chưa | Cần bật VM |
| T0.4 | Chưa | Cần bật VM |
| T0.5 | Chờ | Chờ `kaggle.json` |

### Giai đoạn 1: dữ liệu

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| DL1 | Chưa | |
| DL2 | Chưa | |
| DL3 | Chưa | |
| DL4 | Chưa | |
| DL5 | Chưa | |
| DL6 | Chưa | |

### Giai đoạn 2: sửa code PDP

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| F1 | Chưa | |
| F2 | Chưa | |
| F3 | Chưa | |
| F4 | Chưa | |
| F5 | Chưa | |
| F6 | Chưa | |
| F7 | Chưa | |
| F8 | Chưa | |
| F9 | Chưa | |
| F10 | Chưa | |
| F11 | Chưa | Phần nghiệm thu chạy trên VM |
| F12 | Chưa | |

### Giai đoạn 3–6: hạ tầng chạy, đánh giá, cải tiến, baseline

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| R1 | Chưa | |
| R2 | Chưa | |
| R3 | Chưa | |
| R4 | Chưa | |
| R5 | Chưa | Tùy chọn |
| V1 | Chưa | |
| V2 | Chưa | |
| V3 | Chưa | |
| V4 | Chưa | |
| V5 | Chưa | |
| V6 | Chưa | |
| I1–I5 | Chưa | |
| B1–B3 | Chưa | |

### Mốc và thí nghiệm

| Mốc / thí nghiệm | Trạng thái | Ghi chú |
|---|---|---|
| G0 | Chưa | |
| P1–P3, G1 | Chưa | |
| E0–E5, G2 | Chưa | |
| A1–A9 | Chưa | |
| G3 | Chưa | |

## Nhật ký

- **28/09/2026**
  - Rà soát plan lần cuối trước khi code, cập nhật lên v1.3 (Phụ lục B của plan).
  - Nhóm chốt: QĐ-1 = 224 slot, QĐ-2 = có, QĐ-7 = chưa push GitHub (repo local), xóa VM cũ.
  - Đã xóa VM `anmetarayban` và 2 ổ (phải tắt deletion protection trước). `auto-cl` đang tắt.
  - T0.1: tạo repo git local; ép LF cho repo (máy đang đặt `core.autocrlf=true` global, sẽ làm hỏng script shell trên VM).
