# Hướng dẫn máy ảo GCP cho AutoCheckout-CL

Tài liệu này hướng dẫn vận hành máy ảo `auto-cl` cho project AutoCheckout-CL (PDP trên bộ RPC). Các việc cần làm và lý do nằm trong [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md); tài liệu này chỉ ghi cách thao tác.

Một số lệnh chỉ dùng được sau khi task tương ứng trong plan đã làm xong; các lệnh đó được ghi chú *(cần task X)*.

---

## 0. Thông tin nhanh

Thông tin dưới đây được kiểm tra bằng `gcloud` và SSH ngày 28/09/2026.

| Mục | Giá trị |
|---|---|
| Repo | https://github.com/AnNguyen05092004/AutoCheckout-CL.git (public) |
| Project ID | `project-95a0d104-9d0f-4aa1-ba0` |
| Tên VM / zone | `auto-cl` / **`us-central1-c`** |
| Loại máy | `g2-standard-4` (4 vCPU, 16 GB RAM, trong VM thấy 15 GiB) |
| GPU | 1 × **NVIDIA L4**, 23 GB dùng được, compute capability 8.9 |
| Kiểu cấp phát | **STANDARD (on-demand)**, khoảng 18.400 VND/giờ. Chuyển sang Spot thì còn khoảng 11.100 VND/giờ (mục 2.5) |
| Ổ đĩa | 1 ổ boot 100 GB `pd-balanced` (NVMe), còn trống khoảng 80 GB, tốc độ ghi đo được khoảng 178 MB/s. Có snapshot tự động hằng ngày, giữ 14 ngày. Không có ổ dữ liệu riêng |
| Image | `common-cu129-ubuntu-2204-nvidia-580-v20260909`: Ubuntu 22.04.5, driver 580.178.04, CUDA 12.9 (`/usr/local/cuda`, có `nvcc`). **Không cài sẵn PyTorch** |
| Python | 3.10.12 (`/usr/bin/python3`). **Chưa có** `python3.10-venv` và `unzip` (cài ở mục 5.2) |
| User khi SSH bằng `gcloud` từ Mac | `an` |

Quota của project chỉ cho **1 GPU chạy cùng lúc**. Muốn bật VM `anmetarayban` (V100) thì phải tắt `auto-cl` trước.

Biến đường dẫn dùng trong tài liệu (nên thêm vào `~/.bashrc` trên VM):

```bash
export PROJ=~/AutoCheckout-CL        # code
export DATA=/data/rpc                # dữ liệu RPC
export RUNS=/data/runs               # kết quả train
export VENV=~/venvs/pdp              # môi trường Python
```

Để gõ lệnh ngắn hơn trên Mac:

```bash
export VM="auto-cl --zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0"
```

---

## 1. Nguyên tắc

1. **Dataset chỉ tải và xử lý trên VM**, không tải về Mac. Từ VM về Mac chỉ lấy kết quả (chỉ số, log, file dự đoán nhỏ).
2. Không commit `kaggle.json`, dữ liệu, checkpoint (`*.pth`, `*.ckpt`) lên repo. Repo đang để public.
3. Không chạy `sudo do-release-upgrade`: giữ Ubuntu 22.04 để không làm hỏng driver và CUDA.
4. Job dài luôn chạy trong `tmux`.
5. **Không dùng thì tắt VM.** VM on-demand vẫn tính tiền khi GPU đứng yên.
6. **Credit hết hạn ngày 24/10/2026.** Trước ngày này phải tải kết quả về và xóa VM và ổ. Nếu tài khoản đã nâng cấp lên trả phí, sau ngày đó phí sẽ tính vào phương thức thanh toán.

---

## 2. Bật, SSH và tắt VM (từ Terminal trên Mac)

### 2.1 Bật VM

```bash
gcloud compute instances start $VM
```

### 2.2 SSH

```bash
gcloud compute ssh $VM
```

Có thể SSH từ Console (Compute Engine → VM instances → `auto-cl` → SSH). Lưu ý:

- SSH trên trình duyệt có thể đăng nhập bằng user khác (thường là tên tài khoản Google), tức là thư mục home khác.
- Dữ liệu và kết quả để ở `/data` nên không phụ thuộc vào user; code và venv ở home của user `an`.

### 2.3 Tắt VM

```bash
gcloud compute instances stop $VM
```

Hoặc trong VM: `sudo shutdown -h now`. Khi VM tắt, file trên ổ vẫn còn; chỉ còn tính tiền ổ (khoảng 261 nghìn VND/tháng cho 100 GB `pd-balanced`).

### 2.4 Xem trạng thái

```bash
gcloud compute instances describe $VM --format="value(status,scheduling.provisioningModel)"
```

### 2.5 (Khuyên dùng) Chuyển sang Spot để giảm khoảng 40% chi phí

Nên chuyển khi code resume đã chạy được (task R1/R2), vì Spot có thể bị Google thu hồi giữa chừng. Lệnh chỉ chạy được khi VM **đang tắt**:

```bash
gcloud compute instances stop $VM
gcloud compute instances set-scheduling $VM \
  --provisioning-model=SPOT --instance-termination-action=STOP --no-restart-on-failure
gcloud compute instances start $VM
```

Lưu ý khi chạy Spot (chi tiết ở mục 3.6 của plan):

- Tốc độ không đổi.
- Google có thể tắt VM bất cứ lúc nào, chỉ báo trước tối đa 30 giây, và VM **không tự bật lại**.
- Có lúc hết GPU Spot nên không bật được VM. Nếu đang gấp, chuyển tạm về on-demand (VM phải đang tắt):

```bash
gcloud compute instances set-scheduling $VM --provisioning-model=STANDARD
```

Khi đã chạy Spot, xem VM có bị thu hồi không:

```bash
gcloud compute operations list --project=project-95a0d104-9d0f-4aa1-ba0 \
  --filter="operationType=compute.instances.preempted" --limit=5
```

---

## 3. Kiểm tra lần đầu trên VM

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
# mong đợi: NVIDIA L4, 23034 MiB, 580.178.04, 8.9
/usr/local/cuda/bin/nvcc --version | tail -1
python3 --version       # 3.10.12
df -h /                 # còn khoảng 80 GB
```

---

## 4. Ổ đĩa và thư mục dữ liệu

VM chỉ có một ổ 100 GB. Tạo thư mục dữ liệu dùng chung một lần:

```bash
sudo mkdir -p /data/rpc /data/runs && sudo chmod 1777 /data /data/rpc /data/runs
```

Dung lượng cần (mục 3.3 của plan): khoảng 50 GB nếu **xóa file zip RPC (15,9 GB) sau khi giải nén**. 80 GB trống là đủ, nhưng phải dọn checkpoint thường xuyên.

Nếu thiếu chỗ thì tăng dung lượng ổ. Việc này không mất dữ liệu; mỗi 50 GB thêm khoảng 130 nghìn VND/tháng:

```bash
gcloud compute disks resize auto-cl --size=150GB --zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0
```

Sau đó khởi động lại VM để hệ thống tự mở rộng phân vùng, rồi kiểm tra bằng `df -h /`.

---

## 5. Lấy code và cài môi trường

### 5.1 Lấy code

```bash
cd ~
if [ -d AutoCheckout-CL ]; then cd AutoCheckout-CL && git pull; \
else git clone https://github.com/AnNguyen05092004/AutoCheckout-CL.git; fi
```

### 5.2 Môi trường Python riêng

Image chưa có gói tạo venv (đã kiểm tra: thiếu `ensurepip`) và chưa có `unzip`, nên cài trước:

```bash
sudo apt-get update && sudo apt-get install -y python3.10-venv unzip
```

Tạo venv và cài thư viện. Dùng đúng các phiên bản code PDP cần; `torch 2.2.2 + cu121` chạy được trên L4.

```bash
python3 -m venv $VENV
source $VENV/bin/activate
pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" transformers==4.37.2 tokenizers==0.15.1 huggingface-hub==0.20.3 \
  safetensors==0.4.2 timm==0.9.12 lightning==2.1.3 pytorch-lightning==2.1.3 \
  torchmetrics==1.3.0.post0 pycocotools scipy scikit-learn matplotlib tqdm ninja kaggle pytest
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Kết quả mong đợi: `2.2.2+cu121 True NVIDIA L4`.

- Danh sách này sẽ được chuyển vào file `requirements-vm.txt` *(cần task T0.2)*.
- Mỗi lần SSH lại: `source $VENV/bin/activate`.

### 5.3 Kernel CUDA *(cần task F11)*

Code gốc không bao giờ nạp được kernel và luôn chạy bản PyTorch chậm hơn. Sau khi sửa F11:

```bash
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
cd $PROJ/pdp
python -c "import models.modeling_deformable_detr as m; print('kernel:', m.MultiScaleDeformableAttention is not None)"
```

- Lần chạy đầu sẽ biên dịch kernel, mất vài phút.
- Phải in ra `kernel: True`.
- Có thể có cảnh báo lệch phiên bản CUDA (toolkit 12.9 so với torch build cu121). Cảnh báo cùng major 12 thường không sao; nếu báo lỗi thì ghi lại nguyên văn.

### 5.4 Test *(cần giai đoạn 2–3 của plan)*

```bash
cd $PROJ && pytest -q
```

---

## 6. Tải dataset RPC (chỉ trên VM)

1. Lấy API token trên kaggle.com (Settings → API → Create New Token), được file `kaggle.json`.
2. Đưa token lên VM (**không commit**). Từ Mac:

```bash
gcloud compute scp ~/Downloads/kaggle.json auto-cl:~/kaggle.json \
  --zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0
```

3. Trên VM:

```bash
mkdir -p ~/.kaggle && mv ~/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
source $VENV/bin/activate
mkdir -p $DATA/raw && cd $DATA/raw
kaggle datasets download -d diyer22/retail-product-checkout-dataset -p .      # 15,9 GB
unzip -q retail-product-checkout-dataset.zip \
  'retail_product_checkout/val2019/*' 'retail_product_checkout/test2019/*' \
  'retail_product_checkout/instances_val2019.json' 'retail_product_checkout/instances_test2019.json'
ls retail_product_checkout/val2019 | wc -l     # mong đợi 6000
ls retail_product_checkout/test2019 | wc -l    # mong đợi 24000
rm retail-product-checkout-dataset.zip         # giải phóng 15,9 GB; khi cần ảnh sản phẩm đơn thì tải lại
```

Ghi chú:

- Giai đoạn 1 chỉ cần ảnh quầy (`val2019`, `test2019`).
- Các bước tiếp theo (kiểm tra, thu nhỏ ảnh, chia tập): *(cần task DL1–DL6)*.

---

## 7. Chạy train *(cần task R2)*

```bash
tmux new -s train
source $VENV/bin/activate && cd $PROJ
bash scripts/run_exp.sh configs/exp/<tên_thí_nghiệm>.args --shutdown 2>&1 | tee -a $RUNS/<tên_thí_nghiệm>.log
```

- Rời tmux mà job vẫn chạy: `Ctrl+B`, rồi nhấn `D`. Quay lại: `tmux attach -t train`.
- `--shutdown`: tắt VM khi job kết thúc, kể cả khi lỗi, để không tốn tiền lúc VM đứng yên.
- Nên chạy smoke test (R4) và pilot trước khi chạy dài.

### 7.1 Khi VM bị tắt giữa chừng

VM có thể bị tắt giữa chừng do Spot bị thu hồi, lỗi, hoặc bảo trì. Khi đó:

1. Bật VM lại (mục 2.1) và SSH vào.
2. Chạy lại **đúng lệnh cũ**. Script sẽ bỏ qua các task đã xong và resume task đang dở từ checkpoint gần nhất.

---

## 8. Theo dõi từ Mac (chỉ đọc, không ảnh hưởng job)

```bash
gcloud compute ssh $VM --command "tail -F /data/runs/<tên_thí_nghiệm>.log"          # log trực tiếp
gcloud compute ssh $VM --command "nvidia-smi"                                        # GPU có đang chạy
gcloud compute ssh $VM --command "tmux capture-pane -pt train:0.0 -S -120"           # 120 dòng cuối của tmux
gcloud compute ssh $VM --command "ls /data/runs/*/task_*/task_final.pth; df -h /"    # task đã xong, dung lượng
```

Nhấn `Ctrl+C` chỉ dừng việc xem, không dừng job.

---

## 9. Lấy kết quả về Mac

Chỉ lấy chỉ số, log và file dự đoán; không lấy dữ liệu, không lấy checkpoint.

```bash
mkdir -p "/Users/an/Documents/Do An/AutoCheckout-CL/results"
gcloud compute scp --recurse auto-cl:/data/runs/<tên_thí_nghiệm> \
  "/Users/an/Documents/Do An/AutoCheckout-CL/results/" \
  --zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0
```

Nếu thư mục run có checkpoint, nên nén riêng phần cần lấy trên VM trước, ví dụ `tar czf` với `--exclude='*.pth' --exclude='*.ckpt'`.

---

## 10. Checklist trước khi chạy dài

```text
[ ] nvidia-smi thấy NVIDIA L4; torch.cuda.is_available() = True trong venv
[ ] Kernel CUDA: in ra "kernel: True" (mục 5.3)
[ ] pytest -q xanh
[ ] Dữ liệu đã chia, md5 của tập test khớp với file trong repo
[ ] Đã chạy smoke test (R4) và pilot
[ ] df -h / còn đủ chỗ (mỗi lần chạy 5 task khoảng 2 GB)
[ ] Chạy trong tmux, có --shutdown
[ ] (Nếu chạy dài) đã chuyển sang Spot và resume đã chạy được
```

---

## 11. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `python3 -m venv` báo thiếu `ensurepip` | `sudo apt-get install -y python3.10-venv` |
| `CUDA out of memory` | Giảm `batch_size` xuống 1; script tự tăng số bước tích lũy gradient để giữ batch hiệu dụng bằng 32. L4 có 23 GB, dư hơn V100 16 GB |
| `kernel: False` / lỗi biên dịch | Kiểm tra `ninja --version` (trong venv), `echo $CUDA_HOME`, `/usr/local/cuda/bin/nvcc --version`; đặt `TORCH_CUDA_ARCH_LIST=8.9`; xóa cache `~/.cache/torch_extensions` rồi thử lại |
| `No space left on device` | `df -h /`, `du -sh $RUNS/*`; xóa `last.ckpt` của các task đã xong (vẫn giữ `task_final.pth`); xóa zip RPC; hoặc tăng dung lượng ổ (mục 4) |
| `gcloud` trên Mac báo `NameResolutionError ... compute.googleapis.com` | DNS của mạng đang dùng (ví dụ mạng trường) chập chờn. Thử lại sau vài giây, hoặc đổi mạng hoặc DNS (ví dụ 8.8.8.8) |
| VM tự tắt giữa chừng | Xem mục 2.4 và 2.5; làm theo mục 7.1 |
| Không bật được VM (báo hết tài nguyên GPU) | Đợi rồi thử lại. Nếu kéo dài: tạo VM ở zone khác, vì ổ đĩa nằm cố định ở `us-central1-c` |
| `git clone`/`git pull` hỏi mật khẩu | Repo public thì không cần. Nếu repo chuyển sang private: dùng GitHub Personal Access Token |
