# Tiến độ AutoCheckout-CL

File này ghi tiến độ implement theo [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (v1.3). Mỗi khi xong hoặc bắt đầu một task thì cập nhật bảng và thêm một dòng vào nhật ký. Hạn credit GCP: **24/10/2026**.

Trạng thái: **Xong** = đạt tiêu chí nghiệm thu trong plan; **Đang làm**; **Chờ** = bị chặn, ghi rõ chờ gì; **Chưa** = chưa bắt đầu. Cột "Kiểm chứng" ghi test hoặc lệnh đã chạy để xác nhận.

## Việc đang chờ nhóm

| Việc | Ai | Ghi chú |
|---|---|---|
| Tạo tài khoản Kaggle (miễn phí) và API token (`kaggle.json`), cho biết đường dẫn trên Mac | Nhóm | Cần cho T0.5. Mirror HuggingFace không dùng được vì thiếu tên file và `level` (plan mục 4.1). File chỉ copy thẳng lên VM, không commit |
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
| F5 | Xong (28/09) | Teacher = bản sao đóng băng sau khi nạp trọng số task trước (task_count t-2), không nằm trong state_dict; suy luận 2 lượt có prompt; `--teacher_prompts 0` = gốc. `tests/test_pdp_f5_teacher.py` |
| F6 | Xong (28/09) | `pdp/ppg.py` (hàm thuần): top-k query theo lớp cũ tốt nhất, nhãn < PREV, đặc trưng theo chỉ số query; `--pseudo {ppg,threshold,none}`, `--ppg_legacy 1` = gốc. `tests/test_pdp_f6_ppg.py` (có test ghi lại lỗi `<=` gốc) |
| F7 | Xong (28/09) | Chỉ query phân loại đúng vào bộ nhớ; in lớp thiếu prototype cuối task. `tests/test_pdp_f7_prototypes.py` |
| F8 | Xong (28/09) | `run_task()`; `task_<t>/task_final.pth` (ghi nguyên tử); `--prev_ckpt`, `--train_suffix`, `--accelerator`; bỏ checkpoint mỗi epoch. Test chạy `main()` đầu-cuối trên CPU `tests/test_pdp_f8_paths.py` |
| F9 | Xong (28/09) | `pdp/inference.py`; validation 2 lượt không teacher; sau mỗi task ghi `pred_{val,test}.npz`. Test: khớp hậu xử lý gốc, validation không gọi teacher. `tests/test_pdp_f9_inference.py` |
| F10 | Xong (28/09) | Bảng tham số train theo nhóm/lr trong log; model thật 69,12M / 35,00M đúng như plan. `tests/test_pdp_f10_parameters.py` |
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
| V1 | Xong (28/09) | Lõi trong `pdp/inference.py`; CLI là `main.py --predict_only 1` (thay cho `tools/predict.py` trong plan) |
| V2 | Code xong (28/09) | `autocheckout/cl_metrics.py`, `tools/eval_cl.py`; M1 khớp COCOeval chạy trên file GT theo nhóm (cách của code gốc) |
| V3 | Code xong (28/09) | `autocheckout/counting.py`, `tools/eval_count.py`; khớp công thức rpctool; lớp không có GT bị loại khỏi trung bình mCCD/mCIoU (ghi rõ trong file kết quả) |
| V4 | Chưa | |
| V5 | Chưa | |
| V6 | Code xong (28/09) | `tools/summarize.py`: bảng md/csv + biểu đồ |
| I1–I5 | Chưa | |
| B1–B3 | B3 code xong (28/09) | `baselines/retrieval.py` (19 test, qua được eval_cl/eval_count). B1, B2 chưa làm |

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
  - Giao B3 (E5) cho agent phụ; đã merge (19 test).
  - Nhóm yêu cầu từ nay chỉ chạy 1 agent chính (không dùng agent phụ) để tiết kiệm token.
  - F5–F10, V1 xong. Test F2 phát hiện lỗi Gram-Schmidt khi prompt cũ đã train → sửa bằng QR. Toàn bộ 129 test đạt trên Mac (3 test GPU bỏ qua).

## Nhật ký tiếp

- **29/09/2026**: nhóm chốt E5 = phương án b; giữ 14 snapshot ổ cũ. Kiểm tra nguồn dữ liệu: mirror HuggingFace thiếu tên file và `level`, nên vẫn cần Kaggle. Ảnh quầy RPC không cố định 1800 px (khoảng 1750–1890, vuông); DL2 đã xử lý theo từng ảnh.

## Handoff (cập nhật 28/09/2026, trước khi compact)

Mọi thông tin cần để làm tiếp nằm ở đây, trong plan và `docs/formats.md`; không cần lịch sử hội thoại.

### Trạng thái code

- `main` sạch, 129 test đạt (`.venv/bin/python -m pytest`; khoảng 1,5 phút). Lint: `.venv/bin/python -m ruff check autocheckout tools tests baselines` (không lint `pdp/`, vì là code upstream).
- Mỗi bản sửa của `pdp/` là 1 commit có tiền tố mã (F1…F12). `git log -- pdp/` liệt kê mọi khác biệt so với upstream.
- Luồng một task (`pdp/main.py` → `run_task`): `set_task_id(t-1)` → nạp `task_{t-1}/task_final.pth` (hoặc `--prev_ckpt`) → `set_teacher()` (nếu `--pseudo != none`) → `init_task_prompts()` → `fit` → `save_task_final` → `write_task_predictions` (val/test full).

### Việc tiếp theo, theo thứ tự (quyết định thiết kế đã chốt)

1. **R1 resume**: `configure_optimizers` trả về cả `StepLR` (interval epoch) và bỏ `self.lr_scheduler.step()` trong `on_train_epoch_end`. Thêm `on_save_checkpoint`/`on_load_checkpoint` lưu `class_query_cache`, `class_prototypes`, `class_cache_count`, `batch_counter`. Viết callback riêng (không dùng `ModelCheckpoint`) ghi `task_<t>/last.ckpt` bằng `trainer.save_checkpoint(tmp)` rồi `os.replace`, giữ `last.ckpt.prev`. Callback ghi cả theo thời gian (khoảng 30 phút, chỉ ngay sau một bước optimizer) và cuối epoch. `run_task` gọi `fit(ckpt_path=last.ckpt nếu có)`. Teacher không nằm trong checkpoint, được dựng lại từ `task_final` của task trước. Khi task xong thì xóa `last.ckpt*`. Test: dừng giữa chừng rồi resume trên CPU, kiểm tra epoch, lr và prototype.
2. **R3**: ghi `task_<t>/run_info.json`: args, `git rev-parse HEAD` + `git diff`, phiên bản thư viện, tên GPU, md5 các file annotation, thời gian task, `torch.cuda.max_memory_allocated`.
3. **R2**: `scripts/run_exp.sh <configs/exp/X.sh> [--shutdown]`. File config là bash, được `source`, định nghĩa `EXP`, `N_TASKS`, `ARGS=(...)`, tùy chọn `REUSE_TASK1=<run>` (tạo symlink `task_1`). Mỗi task chạy một process `main.py --start_task t --n_tasks t`; bỏ qua task đã có `DONE`; chạy V2/V3 sau mỗi task; `trap` để `--shutdown` tắt VM kể cả khi lỗi. Cập nhật §7 của guide cho khớp (guide đang ghi `.args`).
4. **R4**: smoke test CPU đã có (`tests/test_pdp_f8_paths.py`, `test_pdp_f9_inference.py`). Cần thêm smoke GPU trên VM: 200 ảnh thật, đo giây/ảnh và bộ nhớ.
5. **B2** (các cờ còn thiếu): `--use_shared`, `--use_private` (đặt vào config, `Prompt.forward` bỏ pool tương ứng, DDL bỏ qua khi thiếu pool). Các cờ `--pseudo`, `--pseudo_topk`, `--ddl_lambda` và cờ hành vi gốc đã có.
6. **I1–I5**: I3 (`--pseudo_gt_iou 0.5`: bỏ nhãn giả chồng lên GT, IoU trên box cxcywh chuẩn hóa) và I4 (`--prototype_nearest`) thêm vào `ppg.select_pseudo_labels`. I5 (`--freeze_shared_after_task1`: đóng băng `input_proj`, `query_tf`, `query_position_embeddings`, `reference_points`, `level_embed`, `bbox_embed` khi t ≥ 2). I2 (augmentation khi train: xoay bội số 90°, đổi màu nhẹ, `shortest_edge` ngẫu nhiên 640–800, không lật) trong `CocoDetection`, chỉ áp cho dataset train. I1 (FSA) = B1b.
7. **B1**: không viết `baselines/adapt.py` riêng mà dùng `main.py` với cờ `--optim_groups detr` (nhánh else của `configure_optimizers`: lr 1e-4, backbone 1e-5, sampling_offsets/reference_points × 0,1), `--use_prompts 0`, `--freeze ''`, `--pseudo none`. E0 cần DL5 sinh thêm `train_joint(_capped).json` (mọi ảnh train, mọi nhãn) và chạy như một task có `seen_classes = 200`, ghi vào `task_5/`. FSA: fine-tune task 1 rồi `save_pretrained` + processor, sau đó PDP task 1 dùng `--repo_name <thư mục>`. B1c (detector 1 lớp cho E5): DL5 thêm tùy chọn class-agnostic, `--n_classes 2`. Cập nhật plan B1 theo cách này.
8. **V4** `tools/ppg_audit.py`: teacher + `ppg.py` trên `train_task_<t>_gt_full.json` (1.000 ảnh), đo precision/recall của nhãn giả theo nhánh (tin cậy cao / qua prototype) và theo nhóm hàng. **V5**: đo độ trễ và dung lượng.
9. Viết lại `GCP_TRAINING_GUIDE.md`: §5.1 lấy code bằng `git bundle` (`scripts/sync_to_vm.sh`, chưa viết); §5.2 cài bằng `requirements.txt` + `pip install -e .`; §6–7 chạy `scripts/prepare_data.sh` và `run_exp.sh`. Cập nhật plan: V1 dùng `--predict_only`, B1 như mục 7.
10. **Trên VM (tốn tiền, bật khi cần):** T0.3 (venv + `pytest tests/test_pdp_f11_kernel.py`), T0.4, T0.5 (cần `kaggle.json`), DL1 → đọc `results/data_audit/audit.md` → DL2–DL6 → mốc G0 → pilot.

### Lưu ý kỹ thuật (đã gặp)

- **Mac:**
  - `.venv` dùng lại torch 2.2.2 x86_64 (Rosetta) của Python pyenv 3.10.13, qua `--system-site-packages`. Python gốc này có cả TensorFlow, nên phải đặt `USE_TF=0` (tests/conftest.py đã đặt).
  - Đặt `HF_HUB_OFFLINE=1`, nếu không mỗi lần dựng model mất khoảng 60 giây chờ Hub.
  - Chạy script tay ngoài pytest thì tự đặt hai biến trên.
  - Mac chỉ còn khoảng 7 GB trống.
- **Test model nhỏ:**
  - `tests/pdp_helpers.py` gồm `use_tiny_detr`, `pdp_args`, `make_trainer`, `make_batch`, `make_toy_dataset`, `run_main`.
  - Ảnh test 96 px: ở 64 px, batch 1 ảnh làm GroupNorm backward trên CPU lỗi "Expected memory formats…".
- **Style của file trong `pdp/`:**
  - `engine.py` thụt lề bằng tab, `main.py` và `prompt.py` bằng 4 dấu cách.
  - File upstream không có newline cuối; giữ nguyên.
- **Sửa file an toàn:** dùng script Python `assert s.count(old) == 1` rồi `replace`.
  - Tránh `cd` trong Bash, dùng đường dẫn tuyệt đối.
  - Trong zsh, không dùng biến tên `path`.
- **Tách commit:** khi một file chứa nhiều bản sửa, dùng script `split_commits.py` (dựng trạng thái trung gian theo hunk, kiểm tra khớp working tree). Tốt hơn nữa là commit ngay sau mỗi bản sửa.

### Câu hỏi còn mở cho nhóm

- `kaggle.json` (T0.5); QĐ-5 demo. Snapshot ổ cũ: nhóm quyết định giữ lại (29/09).
- **E5 (đã chốt 29/09: phương án b)**, ghi lại để tham khảo: detector class-agnostic (B1c) train bằng box nào? (a) chỉ box của task 1, đúng giao thức nhưng detector sẽ học coi SKU tương lai là nền; (b) mọi box của ảnh task 1, không kèm tên SKU, tức giả định cửa hàng gán box "sản phẩm" từ đầu (lợi thế cho E5, phải ghi rõ). Đề xuất: (b) cho bản chính, (a) nếu còn thời gian.
- **E5 giữa chừng:** softmax không trả lời được "chưa biết", nên SKU chưa học bị gán nhãn SKU gần nhất. Chỉ ảnh hưởng chỉ số ở các task giữa; task cuối không sao.
