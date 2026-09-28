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
| T0.2 | Xong (28/09) | `requirements.txt` + `requirements-dev.txt` (torch cài riêng theo máy), `pyproject.toml` (pytest, ruff). `.venv` trên Mac dùng lại torch 2.2.2 của Python gốc (x86_64 qua Rosetta; mạng tải torch quá chậm và Mac chỉ còn khoảng 7 GB trống); test đặt `USE_TF=0` vì Python gốc có TensorFlow làm crash transformers |
| T0.3 | Chưa | Cần bật VM |
| T0.4 | Chưa | Cần bật VM |
| T0.5 | Chờ | Chờ `kaggle.json` |

### Giai đoạn 1: dữ liệu

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| DL1 | Code xong (28/09) | `tools/audit_rpc.py`, `autocheckout/groups.py`; test trên dữ liệu giả. Chưa chạy trên dữ liệu thật (cần T0.5) |
| DL2 | Code xong (28/09) | `tools/resize.py`; test co bbox/area, số vật không đổi, chạy lại không đổi byte |
| DL3 | Code xong (28/09) | `tools/make_split.py`; test không rò rỉ nhóm, chỉ nhóm thuần test2019 vào val/test, thử seed khi thiếu vật/SKU, tái lập md5 |
| DL4 | Code xong (28/09) | `tools/make_task_config.py`; phân bổ theo nhóm hàng bằng largest remainder, kích thước task chính xác |
| DL5 | Code xong (28/09) | `tools/make_task_json.py`; nhiều nguồn train, file `_capped`, `_gt_full`, manifest; file đọc được bằng `CocoDetection` của PDP |
| DL6 | Code xong (28/09) | Cùng công cụ DL5 với `--tasks 1,2` trên `train_pilot`; `scripts/prepare_data.sh` chạy DL1→DL6 |

### Giai đoạn 2: sửa code PDP

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| F1 | Xong (28/09) | `task_info_rpc`, kiểm tra `--n_classes`; `tests/test_pdp_f1_task_config.py` |
| F2 | Xong (28/09) | Pool 224 slot; khởi tạo prompt task mới. **Test phát hiện thêm lỗi:** Gram-Schmidt gốc không trực giao được khi prompt cũ đã train (cos tới 0,17) → dùng phép chiếu QR. `tests/test_pdp_f2_private_pool.py` |
| F3 | Xong (28/09) | L_DDL khớp công thức paper, gradient tới pool chung và prompt task hiện tại; `tests/test_pdp_f3_ddl.py` |
| F4 | Xong (28/09) | L_Q có gradient vào `query_tf` (test trên model nhỏ); `tests/test_pdp_f4_query_loss.py` |
| F5 | Chưa | |
| F6 | Chưa | |
| F7 | Chưa | |
| F8 | Chưa | |
| F9 | Chưa | |
| F10 | Chưa | |
| F11 | Code xong (28/09) | Đường dẫn kernel trong gói transformers (đã xác nhận có mã nguồn), log khi rơi về PyTorch, `--require_kernel`. Test so kernel với PyTorch chỉ chạy trên VM |
| F12 | Xong (28/09) | `shuffle=True`; thứ tự khác giữa 2 epoch, tái lập theo seed; `tests/test_pdp_f12_shuffle.py` |

### Giai đoạn 3–6: hạ tầng chạy, đánh giá, cải tiến, baseline

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| R1 | Chưa | |
| R2 | Chưa | |
| R3 | Chưa | |
| R4 | Chưa | |
| R5 | Chưa | Tùy chọn |
| V1 | Chưa | |
| V2 | Code xong (28/09) | `autocheckout/cl_metrics.py`, `tools/eval_cl.py`; M1 khớp COCOeval chạy trên file GT theo nhóm (cách của code gốc) |
| V3 | Code xong (28/09) | `autocheckout/counting.py`, `tools/eval_count.py`; khớp công thức rpctool; lớp không có GT bị loại khỏi trung bình mCCD/mCIoU (ghi rõ trong file kết quả) |
| V4 | Chưa | |
| V5 | Chưa | |
| V6 | Code xong (28/09) | `tools/summarize.py`: bảng md/csv + biểu đồ |
| I1–I5 | Chưa | |
| B1–B3 | Đang làm | B3 (E5, truy xuất DINOv2) giao cho agent phụ ngày 28/09 |

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
  - T0.2 xong. F1, F2, F3, F4, F11, F12 xong, mỗi bản sửa một commit kèm test.
  - Agent phụ làm xong DL1–DL6 (Opus) và V2, V3, V6 (Sonnet); đã rà code, sửa lint, quyền file 0600, tăng tốc V3; đã merge. 91 test đạt trên Mac.
  - Giao B3 (E5) cho agent phụ.
