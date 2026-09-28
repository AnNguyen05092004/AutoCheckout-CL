# Tiến độ AutoCheckout-CL

File này ghi tiến độ implement theo [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (v1.3). Mỗi khi xong hoặc bắt đầu một task thì cập nhật bảng và thêm một dòng vào nhật ký. Hạn credit GCP: **24/10/2026**.

Trạng thái: **Xong** = đạt tiêu chí nghiệm thu trong plan; **Đang làm**; **Chờ** = bị chặn, ghi rõ chờ gì; **Chưa** = chưa bắt đầu. Cột "Kiểm chứng" ghi test hoặc lệnh đã chạy để xác nhận.

## Việc đang chờ nhóm

| Việc | Ai | Ghi chú |
|---|---|---|
| Thu hồi và tạo lại Kaggle API token (Settings → API) | Nhóm | Token đã xuất hiện trong đoạn chat; dữ liệu đã tải xong nên thu hồi không ảnh hưởng gì |
| QĐ-5: demo webcam có nằm trong phạm vi không | Nhóm | Cần trước giai đoạn demo |

## Bảng trạng thái

### Giai đoạn 0: nền tảng

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| T0.1 | Xong (28/09; push 28/09) | GitHub public `AnNguyen05092004/AutoCheckout-CL` (QĐ-7 chốt lại 28/09). `pdp/` giống hệt blob upstream `7702d91` (so bằng `git hash-object`); bỏ `__pycache__` của upstream |
| T0.2 | Xong (28/09) | `requirements.txt` + `requirements-dev.txt` (torch cài riêng theo máy), `pyproject.toml` (pytest, ruff). `.venv` trên Mac dùng lại torch 2.2.2 của Python gốc (x86_64 qua Rosetta; mạng tải torch quá chậm và Mac chỉ còn khoảng 7 GB trống); test đặt `USE_TF=0` vì Python gốc có TensorFlow làm crash transformers |
| T0.3 | Xong (28/09) | `setup_vm.sh`: torch 2.2.2+cu121 trên L4; kernel CUDA build 147 s, nhanh hơn bản PyTorch khoảng 20 lần. Image thiếu `g++` và `python3.10-dev`, script đã cài thêm. 181/181 test đạt trên VM |
| T0.4 | Xong (28/09) | `/data/rpc`, `/data/runs`; sau khi chuẩn bị dữ liệu còn trống khoảng 60 GB |
| T0.5 | Xong (28/09) | Token Kaggle kiểu mới (`KGAT_`, Kaggle CLI cho Python 3.10 không đọc được) → tải bằng `curl`; zip 25,3 GB trong khoảng 4 phút; 6.000 + 24.000 ảnh; đã xóa zip |

### Giai đoạn 1: dữ liệu

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| DL1 | Xong (28/09) | `results/data_audit/audit.md`. Cạnh ảnh 1751–1906 px (1 ảnh 1860×1859); mỗi hậu tố = 3 giỏ khác mức × 3 lần chụp; không hậu tố nào có ở cả val2019 lẫn test2019; không có ảnh trùng |
| DL2 | Xong (28/09) | 30.000 ảnh 800×800, số vật 367.935 trước = sau; ảnh gần vuông được co theo từng trục (sửa sau DL1) |
| DL3 | Xong (28/09) | `--stratify none` (nhóm trộn mức, sửa sau DL1). Seed 0 đạt ngay: test 6.003 (2.008/1.969/2.026), val 1.503, train 22.494, pilot 3.002. Khóa bằng md5 trong `configs/splits/` |
| DL4 | Xong (28/09) | `configs/tasks_100-4x25_seed0.json`: 100 + 4×25 phân bổ theo 17 nhóm hàng, 24 slot dự phòng |
| DL5 | Xong (28/09) | Task 1: 21.752 ảnh (capped 6.000); task 2–5: khoảng 12.000 ảnh mỗi task (capped 6.000); joint, class-agnostic; md5 `test_full.json` = `8a508281…` |
| DL6 | Xong (28/09) | Pilot task 1: 2.910 ảnh, task 2: 1.692 ảnh |

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
| F11 | Xong (28/09) | Kernel nạp được trên L4, khớp bản PyTorch (forward ≤ 1e-4, gradient), 0,07–0,10 ms so với 1,7–1,9 ms |
| F12 | Xong (28/09) | `shuffle=True`; thứ tự khác giữa 2 epoch, tái lập theo seed; `tests/test_pdp_f12_shuffle.py` |
| F13 | Xong (28/09) | Prior của focal loss cho classifier (HuggingFace đặt bias về 0 → p = 0,5); phát hiện ở pilot; `tests/test_pdp_f13_prior_init.py` |

### Giai đoạn 3–6: hạ tầng chạy, đánh giá, cải tiến, baseline

| ID | Trạng thái | Kiểm chứng / ghi chú |
|---|---|---|
| R1 | Xong (28/09) | `pdp/checkpointing.py`, scheduler qua Lightning, bộ nhớ prototype trong checkpoint; test ngắt giữa epoch cuối rồi resume: epoch, bước, scheduler, prototype khớp, tổng số bước không đổi |
| R2 | Xong (28/09) | `scripts/run_exp.sh` + `configs/exp/*.sh` (mọi thí nghiệm); test bằng interpreter giả: bỏ qua task xong, chỉ dự đoán lại, dùng lại task 1, tắt VM cả khi lỗi |
| R3 | Xong (28/09) | `autocheckout/runinfo.py` → `task_<t>/run_info.json` theo session |
| R4 | CPU xong; benchmark GPU xong (28/09) | Benchmark L4: 0,336 giây/ảnh (batch 4, 6,2 GB), suy luận 110 ms/ảnh → `BATCH_SIZE=4`. Còn smoke GPU trên ảnh thật |
| R5 | Chưa | Tùy chọn; làm khi chuyển Spot nếu cần |
| V1 | Xong (28/09) | Lõi trong `pdp/inference.py`; CLI là `main.py --predict_only 1` (thay cho `tools/predict.py` trong plan) |
| V2 | Code xong (28/09) | `autocheckout/cl_metrics.py`, `tools/eval_cl.py`; M1 khớp COCOeval chạy trên file GT theo nhóm (cách của code gốc) |
| V3 | Code xong (28/09) | `autocheckout/counting.py`, `tools/eval_count.py`; khớp công thức rpctool; lớp không có GT bị loại khỏi trung bình mCCD/mCIoU (ghi rõ trong file kết quả) |
| V4 | Code xong (28/09) | `pdp/ppg_audit.py`; chạy sau pilot |
| V5 | Code xong (28/09) | `pdp/benchmark.py` (độ trễ, bộ nhớ, dung lượng/lớp) |
| V6 | Code xong (28/09) | `tools/summarize.py`: bảng md/csv + biểu đồ |
| I1–I5 | Xong (28/09) | I1 = FSA qua `--save_hf` + `--repo_name`; I2 `pdp/augment.py`; I3/I4 trong `ppg.py`; I5 `--freeze_shared_after_task1`; mỗi mục có test |
| B1–B3 | Xong (28/09) | B1 = cờ của `main.py` (joint, save_hf, optim_groups, pred_ann_dir) + DL5 `--joint`/`--agnostic-out`; B2 `--use_shared/--use_private`; B3 `baselines/retrieval.py` |

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

- **28/09/2026 (trên VM):**
  - Lần bật VM đầu bị STOCKOUT, lần thứ hai bật được.
  - Cài môi trường; sửa 3 lỗi chỉ lộ ra trên VM: thiếu `g++`, thiếu `python3.10-dev`, test E5 để model khác thiết bị với ảnh.
  - Tải dữ liệu bằng token Kaggle.
  - DL1 cho thấy cấu trúc nhóm khác giả thuyết → sửa DL2, DL3.
  - Chuẩn bị dữ liệu xong; tập test đã khóa.
  - Benchmark: 0,336 giây/ảnh (batch 4) → khoảng 17 giờ L4 cho mỗi lần chạy 5 task ở cấu hình chuẩn.
  - Smoke GPU trên ảnh thật đạt (R4) → mốc G0 đạt. Bắt đầu pilot, dừng sau 5 phút vì `loss_ce` khoảng 730 → phát hiện và sửa F13, rồi chạy lại pilot.
  - Ghi chú: vài commit message đã đẩy lên ghi nhầm "29/09"; đúng là 28/09 (tài liệu đã sửa).
- **28/09/2026**: nhóm chốt E5 = phương án b; giữ 14 snapshot ổ cũ; đẩy repo lên GitHub (public). Xong R1, R2, R3, I1–I5, B1, B2, V4, V5 (code), cấu hình mọi thí nghiệm, script VM, viết lại guide; V2 nhanh hơn khoảng 5 lần (chính xác tuyệt đối). 178 test đạt.
- 28/09: nhóm chốt E5 = phương án b; giữ 14 snapshot ổ cũ. Kiểm tra nguồn dữ liệu: mirror HuggingFace thiếu tên file và `level`, nên vẫn cần Kaggle. Ảnh quầy RPC không cố định 1800 px (khoảng 1750–1890, vuông); DL2 đã xử lý theo từng ảnh.

## Handoff (cập nhật 28/09/2026, khoảng 21:50 giờ VN — đọc phần này trước tiên)

Mọi thông tin cần để làm tiếp nằm trong file này, `IMPLEMENTATION_PLAN.md` (v1.3 + phụ lục B, C; F13 ở §6.3), `GCP_TRAINING_GUIDE.md` và `docs/formats.md`.

### Trạng thái hiện tại

- **Code:** xong mọi task không cần GPU; mỗi bản sửa là một commit có tiền tố mã.
  - Sửa lỗi PDP: F1–F13.
  - Hạ tầng chạy: R1–R4.
  - Đánh giá: V1–V6.
  - Cải tiến và baseline: I1–I5, B1–B3.
  - Dữ liệu: DL1–DL6.
  - Mọi cấu hình thí nghiệm (`configs/exp/*.sh`) và script VM.
- **Repo:** https://github.com/AnNguyen05092004/AutoCheckout-CL (public, QĐ-7). Commit trên Mac rồi `git push`; VM chạy `git fetch -q origin && git reset -q --hard origin/main`.
- **Test:** 181 đạt trên VM (GPU), khoảng 190 trên Mac (3 test kernel chỉ chạy trên GPU). Mac: `.venv/bin/python -m pytest`; lint: `.venv/bin/python -m ruff check autocheckout tools tests baselines`.
- **VM `auto-cl`:** us-central1-c, L4, **on-demand**, đang bật.
  - Môi trường: `~/venvs/pdp`, kernel CUDA đã build. Code: `~/AutoCheckout-CL`.
  - Dữ liệu: `/data/rpc`, gồm `checkout_800/` (30.000 ảnh), `tasks/100-4x25_seed0`, `tasks/pilot_100-4x25_seed0`, `tasks/agnostic_task1`, `tasks/smoke`.
  - Kết quả: `/data/runs`. Token Kaggle ở `~/.kaggle/access_token`; nhóm nên thu hồi token này.
- **Tập test đã khóa:** `configs/splits/rpc_checkout_seed0.json`; md5 của `test_full.json` là `8a508281e591baebff547422be7899a3` (trong `configs/splits/manifest_100-4x25_seed0.json`).
- **Benchmark L4:** 0,336 giây/ảnh cho task ≥ 2 với batch 4 (6,2 GB) → `BATCH_SIZE=4`; suy luận 110 ms/ảnh. Khoảng 17 giờ cho mỗi lần chạy 5 task; khoảng 165 giờ cho cả plan.
- **Mốc G0:** đạt (smoke GPU trên ảnh thật).
- **ĐANG CHẠY: pilot** (bắt đầu 14:28 UTC = 21:28 giờ VN), tmux `pilot`, `scripts/run_pilot.sh`: P2 → P1 → FSA_pilot → P3 → V4 (`ppg_audit` trên P2 task 2).
  - Mất khoảng 6 giờ; **VM tự tắt khi xong, kể cả khi lỗi.**
  - Theo dõi từ Mac: `bash scripts/vm_status.sh` (tóm tắt) hoặc `bash scripts/vm_status.sh follow` (log trực tiếp). tmux trống vì output nằm trong `/data/runs/<tên>.log`.
  - Mỗi thí nghiệm xong ghi một dòng vào `/data/runs/pilot_chain.log`.
  - Lúc 21:50: P2, task 1, epoch 2/4; `loss_ce` khoảng 0,89 (đầu là khoảng 1,07, sau khi sửa F13).
  - Lúc 22:11: P2 train xong task 1 (43 phút, khoảng 10,5 phút/epoch; `loss_ce` cuối khoảng 0,60), không lỗi. **`WARNING: task 1: 40 classes have no prototype`** (40/100 lớp). Với 40 lớp này, ở task 2 PPG chỉ nhận ứng viên có điểm > τh = 0,5; ứng viên điểm trung bình bị bỏ vì không có prototype để so (`valid` = False trong `ppg.py`). Không làm hỏng quá trình chạy, nhưng nhánh prototype chỉ hoạt động với 60% lớp cũ.

### Việc tiếp theo, theo thứ tự

1. **Khi pilot xong** (VM đã tắt; `vm_status.sh` báo `TERMINATED`):
   - bật VM (`gcloud compute instances start auto-cl --zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0`, có thể gặp STOCKOUT thì thử lại);
   - đọc `pilot_chain.log`, `/data/runs/{P1,P2,P3}/metrics_cl_val.md`, `metrics_count_test.md`, `/data/runs/P2/task_2/ppg_audit.json`, và các dòng `WARNING: task ... classes have no prototype` trong `/data/runs/P2.log`.
2. **Đánh giá mốc G1 (plan §8), chỉ dựa trên val:**
   - (a) P2 phải tốt hơn P1 ở mAP@P của task 2; nếu không thì rà lại F2, F5, F6.
   - (b) Tốc độ thật đã có.
   - (c) FSA làm giảm mAP@C của task 2 ≥ 3 điểm (P3 so với P2) thì bỏ FSA hoặc giảm số epoch FSA.
   - **Rủi ro cần xem:** smoke 1 epoch có 24/25 lớp mới không có prototype, vì F7 chỉ lấy query phân loại đúng. Pilot 4 epoch: task 1 của P2 thiếu 40/100 lớp (xem trên); còn cần xem task 2 và V4. Nếu thiếu nhiều thì nhánh prototype của PPG hoạt động kém. Hướng xử lý: nới F7 (ví dụ lấy lớp đúng trong top-k, hoặc cập nhật prototype ở 2 epoch cuối), rồi so sánh trên val.
   - Ghi kết quả vào PROGRESS.md và plan.
3. **Nếu G1 đạt:** chuyển `auto-cl` sang Spot (guide §2.5, VM phải tắt).
   - Thứ tự chạy: FSA → E4 và E0, E1, E2, E3 → DET → E5 (guide §7.4) → A1–A9. A1, A4, A7, A8 dùng lại task 1 của E4.
   - Chạy mỗi thí nghiệm bằng `bash scripts/run_exp.sh configs/exp/<tên>.sh --shutdown` trong tmux, hoặc viết chuỗi tương tự `run_pilot.sh`.
   - Hạn credit **24/10/2026**: tải kết quả về (guide §9) và xóa VM cùng ổ trước ngày đó.
4. Làm sau: V6 (bảng tổng hợp, `tools/summarize.py`), V5 (bảng độ trễ), R5 (tùy chọn), demo (QĐ-5).

### Lưu ý kỹ thuật

- **Mac:**
  - `.venv` dùng lại torch 2.2.2 x86_64 (Rosetta) của pyenv 3.10.13.
  - Đặt `USE_TF=0` và `HF_HUB_OFFLINE=1` (conftest đã đặt; script chạy tay thì tự đặt).
  - Mac chỉ còn khoảng 7 GB trống. zsh: dùng mảng `GC=(--zone=... --project=...)` thay cho biến chuỗi `$VM`.
  - `gcloud` cần chạy ngoài sandbox, vì DNS của mạng trường chập chờn.
  - Không tải dataset về Mac.
- **Đặc điểm dữ liệu thật:**
  - Mỗi hậu tố tên file có 9 ảnh = 3 giỏ khác mức × 3 lần chụp; không hậu tố nào có ở cả val2019 lẫn test2019.
  - Cạnh ảnh 1751–1906 px; có 1 ảnh 1860×1859.
  - File zip Kaggle nặng 25,3 GB. Tải bằng `curl` với token `KGAT_`, vì Kaggle CLI 1.7.4.5 cho Python 3.10 không đọc được token mới.
- **Test model nhỏ:** `tests/pdp_helpers.py`; ảnh 96 px (ở 64 px, GroupNorm trên CPU lỗi với batch 1 ảnh).
- **Style trong `pdp/`:** `engine.py` thụt lề bằng tab; file upstream không có newline cuối.
- **Sửa file an toàn:** script Python `assert s.count(old) == 1`.
- **Khi dừng một job trên VM:** dùng `pkill -9` từng tiến trình; **không** `pkill -f run_pilot.sh` từ lệnh ssh (khớp với chính shell SSH). SIGKILL để không kích hoạt trap tắt VM.
- **Tham số dòng lệnh:** thêm cờ mới vào `main.py` thì chạy `tests/test_exp_configs.py`.
- **Commit message:** một số commit đã đẩy lên ghi nhầm "29/09"; đúng là 28/09.
- **Chỉ dùng một agent** (không tạo agent phụ), theo yêu cầu của nhóm.

### Câu hỏi còn mở cho nhóm

- QĐ-5: demo webcam.
- Thu hồi token Kaggle cũ.
- **E5 giữa chừng:** softmax không trả lời được "chưa biết", nên SKU chưa học bị gán nhãn SKU gần nhất. Chỉ ảnh hưởng chỉ số ở các task giữa.
