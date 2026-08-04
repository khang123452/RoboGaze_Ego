# Bàn giao phiên làm việc HPC — RoboGaze-Ego + Qwen3.6-35B-A3B

Ghi chú cho đồng nghiệp tiếp tục vào ngày mai. Đọc phần "Việc cần làm tiếp" trước, phần còn lại là bối cảnh/chi tiết để tra cứu khi cần.

## 1. Mục tiêu phiên làm việc

Chạy pipeline RoboGaze-Ego trên cụm Foxconn H100 (Slurm), dùng model **Qwen3.6-35B-A3B** (thay vì Gemma 2 mà bạn đã setup trước đó) làm VLM backend, để chuẩn bị annotate dataset tại `RoboGaze_Ego_data`.

**Trạng thái hiện tại: bị chặn ở khâu serve model Qwen3.6-35B-A3B do xung đột CUDA driver.** Chưa serve được model này trên cụm. Song song đó đang validate lại pipeline bằng Gemma 2 (setup cũ, đã biết chạy được) để không bị block hoàn toàn.

## 2. Thông tin truy cập cụm

```bash
ssh -J rick@140.115.53.110 28144@ai-dxf01.ai.foxconn.com
```
- Login node: `lgn02`
- Partition: `hhai`
- Compute node ví dụ đã dùng: `dgpn04`
- Xin GPU (interactive):
```bash
srun --job-name=RoboGaze_Interactive --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=64G --partition=hhai --pty bash -i
```

**Quan trọng — dùng tmux ngay từ đầu.** Phiên làm việc hôm nay bị rớt SSH **3 lần**, mỗi lần mất hết session (job Slurm, venv activate, server đang chạy...). Từ giờ luôn làm việc trong tmux để rớt kết nối không làm mất tiến độ:
```bash
tmux new -s robogaze
# ... làm việc trong này ...
# nếu bị rớt kết nối, ssh lại rồi:
tmux attach -t robogaze
```

## 3. Đường dẫn quan trọng

| Thành phần | Đường dẫn |
|---|---|
| Repo RoboGaze-Ego | `/work/HHRI-AI/POC/public/RoboGaze_Ego` |
| Dataset cần annotate | `/work/HHRI-AI/POC/public/RoboGaze_Ego_data` |
| → cấu trúc thực tế | `data/qc_sample/<task>_v2.1__episode_NNNNNN/{video.mp4, before_model.srt, after_human.srt}` + `manifest.jsonl` |
| Model Qwen3.6-35B-A3B | `/work/HHRI-AI/POC/public/pretraining_weights/Alibaba-Qwen/qwen36/Qwen3.6-35B-A3B` |
| venv chia sẻ (Gemma 2, đã tune sẵn) | `/work/HHRI-AI/POC/public/RoboGaze_Ego/.venv` |
| venv thử nghiệm Qwen (cô lập, không đụng venv chính) | `.venv-qwen`, `.venv-qwen-hf`, `.venv-qwen-v17` |

## 4. Phát hiện chính — vì sao Qwen3.6-35B-A3B chưa chạy được

**Root cause: driver NVIDIA trên cụm quá cũ so với những gì Qwen3.6-35B-A3B cần.**

- `nvidia-smi`: Driver `535.161.08`, CUDA tối đa hỗ trợ là **12.2**.
- Checkpoint Qwen3.6-35B-A3B thực chất có `model_type: qwen3_5_moe` (không phải "3.6" như tên thư mục).
- `transformers==4.44.2` (đang pin cho Gemma 2) **không** nhận diện được kiến trúc này.
- Field `transformers_version` trong `config.json` của checkpoint ghi `4.57.1`, nhưng thử cài đúng bản `4.57.1` **vẫn không** có `qwen3_5_moe` trong registry — con số đó không đáng tin, có vẻ được ghi từ một bản nội bộ/pre-release chưa lên PyPI.
- **vLLM thì CÓ hỗ trợ** kiến trúc này (`Qwen3_5MoeForConditionalGeneration`, xác nhận qua `vllm.ModelRegistry`), nhưng **chỉ từ bản v0.17.0 trở lên** (tra cứu web, có nguồn). vLLM 0.17.0 bản dựng sẵn (prebuilt) mặc định cần **CUDA 12.9** — vượt quá driver 12.2 của cụm.

**Đã thử và đều thất bại (không phải chưa thử, mà thử kỹ rồi không được):**
1. `pip/uv install --upgrade vllm` (bản mới nhất, 0.26.0) → kéo theo `torch 2.11.0+cu130` → `torch.cuda.is_available() == False` vì driver không hỗ trợ CUDA 13.
2. `uv --torch-backend=auto` (tính năng tự dò driver để chọn bản torch phù hợp) → thất bại vì cần kết nối tới `download.pytorch.org`, **bị chặn firewall** trên cụm này.
3. Build `vllm==0.17.0` từ source, dùng CUDA toolkit cục bộ 12.2 (`module` LMOD) → thất bại vì GCC hệ thống là **8.5.0**, cần GCC ≥ 9. Kể cả nếu qua được bước này, torch được kéo về vẫn là bản mới (2.10.0) — nhiều khả năng vẫn gặp lại lỗi driver-không-đủ-mới ở bước chạy.
4. Kiểm tra **CUDA Forward Compatibility package** (giải pháp NVIDIA khuyến nghị cho đúng tình huống này, không cần nâng driver) — `dpkg`, `rpm`, `ldconfig`, `find` đều xác nhận **không có package này trên node**.

**Kết luận: cần 1 trong 2 hướng sau, không còn cách nào khác trong phạm vi user-space:**
- **(A) Ticket cho admin** xin cài `cuda-compat-13-x` (package nhỏ, không cần nâng driver, không ảnh hưởng job khác) — bản ticket đã soạn sẵn bên dưới. **Chưa rõ đã gửi chưa — cần hỏi lại / gửi ticket này.**
- **(B) Tiếp tục dò version** vllm/torch xem có bản nào vừa hỗ trợ `qwen3_5_moe` vừa build cho CUDA ≤ 12.2 hay không — rủi ro là có thể **không tồn tại bản nào thỏa cả hai điều kiện** (vLLM 0.17.0 là bản sớm nhất hỗ trợ kiến trúc này, và nó đã yêu cầu CUDA 12.9 rồi).

### Ticket đã soạn sẵn cho admin (nếu chưa gửi)

```
Subject: Request CUDA 13.0 forward-compatibility package on hhai partition (H100 nodes)

Node(s) affected: dgpn04 (và có thể toàn bộ node H100 trong partition hhai)
Current driver: 535.161.08 (CUDA 12.2 ceiling theo nvidia-smi)
Issue: Cần chạy workload yêu cầu stack PyTorch/vLLM build cho CUDA 13.0.
Đã xác nhận không có package cuda-compat-13-x trên node (đã kiểm tra dpkg/rpm/ldconfig, đều rỗng).
Request: Cài package CUDA 13.0 Forward Compatibility (cuda-compat-13-x).
Không cần nâng driver, không cần reboot, không ảnh hưởng job đang chạy của người khác —
chỉ thêm 1 thư viện compat mà ứng dụng có thể trỏ tới qua LD_LIBRARY_PATH.
Reference: https://docs.nvidia.com/deploy/cuda-compatibility/
```

## 5. Các "bẫy" môi trường đã gặp — nhớ để không lặp lại

1. **conda env `vllm_py312` rò rỉ vào shell**, làm `.venv` chính (venv Gemma 2) bị lỗi `import torch` (`undefined symbol: ncclCommWindowDeregister`) và làm `pip` (không phải `uv pip`) cài nhầm chỗ. Sửa: **luôn chạy `conda deactivate` (2 lần) trước khi `source .venv/bin/activate`**, mỗi phiên mới. Nên kiểm tra `~/.bashrc` xem có dòng nào tự động `conda activate` không, nếu có thì đây là nguồn gốc lặp lại vấn đề mỗi lần SSH vào.
2. **`ffmpeg`/`ffprobe` không tồn tại trên node** — không có module (`module spider ffmpeg` → not found), không có trong PATH hệ thống. RoboGaze-Ego cần 2 binary này để cắt video. Cách né được (đang thử, **chưa xác nhận chạy trót lọt do bị rớt SSH giữa chừng**):
   ```bash
   uv pip install imageio-ffmpeg   # dùng uv pip, KHÔNG dùng pip thường (pip thường bị conda chiếm PATH)
   FFMPEG_BIN=$(dirname "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')")
   export PATH="$FFMPEG_BIN:$PATH"
   ```
3. **`scripts/run_example.sh` dùng dataset mẫu của bản gốc (`gr1_real`)** không tồn tại trong checkout này — đừng dùng script này để test nhanh. Test trực tiếp bằng `uv run robogaze` trỏ thẳng vào 1 episode thật trong `RoboGaze_Ego_data`.
4. Video trong `RoboGaze_Ego_data` có vẻ encode bằng **AV1** — decoder built-in của `opencv-python-headless` không đọc được (`Failed to get pixel format`), phải dùng `ffmpeg` thật (có `libdav1d`) để trích frame, không dùng `cv2.VideoCapture`.

## 6. Việc cần làm tiếp (theo thứ tự ưu tiên)

- [ ] **Gửi ticket admin** ở mục 4 (nếu chưa gửi) — đây là việc quan trọng nhất vì đang block hoàn toàn hướng Qwen3.6.
- [ ] Reconnect, mở tmux, `conda deactivate` x2, `source .venv/bin/activate`, xác nhận `torch.cuda.is_available()` trả về `True` trước khi làm gì khác.
- [ ] Cài `ffmpeg` bằng `uv pip install imageio-ffmpeg` (mục 5.2), xác nhận `ffmpeg -version` chạy được.
- [ ] Trong 1 pane tmux khác: `bash scripts/serve_vlm.sh` (server Gemma 2).
- [ ] Pane còn lại: chạy thử 1 video thật để xác nhận pipeline RoboGaze-Ego hoạt động đúng trên cụm này:
  ```bash
  export LOCAL_BASE_URL=http://localhost:8000/v1
  export LOCAL_MODEL=gemma4
  EPISODE_DIR="/work/HHRI-AI/POC/public/RoboGaze_Ego_data/data/qc_sample/add_remove_lid_v2.1__episode_000054"
  ffmpeg -y -loglevel error -i "$EPISODE_DIR/video.mp4" -frames:v 1 /tmp/test_frame.jpg
  uv run robogaze \
    --task-instruction "Add or remove the lid." \
    --initial-frame /tmp/test_frame.jpg \
    --video "$EPISODE_DIR/video.mp4" \
    --output-dir outputs/robogaze \
    --video-id add_remove_lid_000054
  ```
  → kiểm tra `outputs/robogaze/add_remove_lid_000054/report.json`.
- [ ] Nếu chạy được: tạo file `scripts/prepare_egodex_dataset.py` (nội dung đã có sẵn trong lịch sử chat với Claude hôm nay — **chưa lưu vào repo**, cần copy-paste vào file này) để convert dataset `RoboGaze_Ego_data` sang format `metadata.csv` + `videos/` + `conditioning_frame/` mà `scripts/run_robogaze_datasets.py` cần.
- [ ] Patch `scripts/run_robogaze_datasets.py` để nhận thêm tên dataset mới:
  ```bash
  sed -i 's/DATASETS = ("gr1_real", "gr1_sim", "droid_mv")/DATASETS = ("gr1_real", "gr1_sim", "droid_mv", "vr_egodex_qc")/' scripts/run_robogaze_datasets.py
  ```
- [ ] Chạy thử batch với `--limit 1` trên Gemma 2 để xác nhận toàn bộ pipeline + dataset adapter hoạt động, trước khi chờ Qwen3.6 sẵn sàng.
- [ ] Khi admin xử lý xong ticket CUDA compat: quay lại thử serve Qwen3.6-35B-A3B bằng `vllm==0.17.0` — venv gợi ý dùng lại hoặc tạo mới `.venv-qwen-v17`.

## 7. Lưu ý

Các file `.sh` dùng Singularity mà Claude đưa ra ở đầu phiên làm việc (`serve_qwen36_35b.sh`, `run_robogaze_batch.sh`) **đã lỗi thời — bỏ qua**. Cách chạy đúng trên cụm này là dùng thẳng `uv` + `.venv` trên compute node, không qua Singularity, theo đúng setup gốc mà bạn đã làm cho Gemma 2.
