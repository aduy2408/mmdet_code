# Local → GitHub → Marimo workflow

Runbook này mô tả flow dùng cho PAHR/LEVIR-Ship. Mục tiêu là bảo đảm source
được kiểm thử và định danh bằng Git SHA trước khi notebook dùng nó, đồng thời
không chạy chồng experiment trên một GPU.

## 1. Làm việc local

Làm việc trên branch `haar` và kiểm tra trạng thái trước khi sửa:

```bash
git status --short --branch
git log -1 --oneline
```

Các thay đổi không liên quan thuộc về người dùng. Không stage `hf_results/`,
dataset, work directories, notebook runtime hoặc file dirty ngoài scope.

Môi trường chuẩn:

```bash
.venv-mmdet/bin/python -m pytest \
  mmdetection/projects/pahr/tests/test_pahr.py -q
```

Trước commit:

```bash
.venv-mmdet/bin/python train_all_haar.py \
  --variants <variant-list> \
  --image-size 768 \
  --epochs 20 \
  --data-root LevirShipData \
  --dataset-out mmdetection/data/levir_ship_coco \
  --batch-size 8 \
  --num-workers 4 \
  --work-dir mmdetection/work_dirs/<dry-run-dir> \
  --dry-run \
  --skip-upload

git diff --check
```

Đọc generated `patched_config.py` để xác nhận image size, schedule, model
knobs và optimizer custom keys; dry-run thành công không tự chứng minh các
giá trị này đúng.

## 2. Commit và push

Stage danh sách file tường minh:

```bash
git add <file-1> <file-2>
git diff --cached --check
git diff --cached --stat
git commit -m "feat(pahr): <imperative summary>"
```

Nếu HTTPS remote không có credential, push bằng SSH mà không cần đổi remote:

```bash
git push git@github.com:aduy2408/mmdet_code.git haar:haar
```

Xác nhận remote chứa đúng commit:

```bash
git ls-remote origin refs/heads/haar
```

Ghi lại full SHA. Không cho Marimo pull dựa trên mô tả như “latest” hoặc chỉ
short SHA.

## 3. Chờ run hiện tại

Trước khi pull source mới:

```python
returncode = active_process.poll()
```

- `None`: process vẫn chạy; không pull và không launch job mới.
- `0`: train/test/upload đã hoàn tất sạch.
- Khác `0`: dừng handoff và đọc log/traceback.

Khi cần queue tuần tự, worker gọi `active_process.wait()` thay vì polling GPU
memory. Chỉ một training process được phép dùng GPU tại một thời điểm.

## 4. Sync Marimo bằng exact SHA

Từ live kernel, chạy Git trong `/marimo/mmdet_code`:

```python
import subprocess

subprocess.run(
    ["git", "pull", "--ff-only", "origin", "haar"],
    cwd="/marimo/mmdet_code",
    check=True,
)
sha = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd="/marimo/mmdet_code",
    text=True,
    capture_output=True,
    check=True,
).stdout.strip()
assert sha == EXPECTED_FULL_SHA
```

`--ff-only` ngăn notebook tự tạo merge commit. SHA mismatch phải chặn launch.

## 5. Thay đổi notebook

Notebook runtime là source of truth. Chỉ tạo/sửa cell qua
`marimo._code_mode`:

```python
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    cell_id = ctx.create_cell(CELL_BODY, hide_code=False)
    ctx.run_cell(cell_id)
```

Không sửa trực tiếp notebook `.py` khi session đang mở. Dùng private names
cho intermediate variables để tránh lỗi multiply-defined names trong DAG.

Kết nối helper dùng placeholder/environment variable:

```bash
MARIMO_TOKEN="$MARIMO_TOKEN" execute-code.sh \
  --url "$MARIMO_URL" \
  --session "$MARIMO_SESSION" \
  -c 'print("connected")'
```

Không ghi URL riêng, session ID, HF token hoặc auth token thật vào repo.

## 6. Launch và theo dõi

Launcher dùng `subprocess.Popen`, work directory và log riêng:

```python
log_handle = open(RUN_LOG, "a", buffering=1)
process = subprocess.Popen(
    [
        "/marimo/mmdet-venv/bin/python",
        "/marimo/mmdet_code/train_all_haar.py",
        "--variants", VARIANT,
        "--image-size", "768",
        "--epochs", "20",
        "--data-root", "/marimo/LevirShipData",
        "--dataset-out",
        "/marimo/mmdet_code/mmdetection/data/levir_ship_coco",
        "--batch-size", "8",
        "--num-workers", "4",
        "--work-dir", RUN_DIR,
        "--skip-test",
        "--hf-repo-id", "duyle2408/fcos_test_haar",
        "--hf-repo-type", "dataset",
    ],
    cwd="/marimo/mmdet_code",
    env={**os.environ, "HF_TOKEN": hf_token},
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
log_handle.close()
```

Không truyền token trên command line và không print environment. Status cell
nên báo:

- PID và `poll()` return code.
- Variant/epoch hiện tại.
- Best validation mAP/AP50/AP75.
- Log tail.
- Số traceback/OOM.
- Checkpoint, predictions và upload completion.

## 7. Scheduled handoff

Queue giữa hai experiment dùng state:

```text
waiting_for_previous
syncing_git
running
completed
failed
```

Worker thực hiện:

1. `previous_process.wait()`.
2. Chặn nếu return code khác 0.
3. `git pull --ff-only`.
4. So sánh full SHA.
5. Kiểm tra state file/PID để tránh duplicate launch.
6. Launch process mới.
7. `wait()` và ghi return code cuối.

State JSON nằm trong work directory của experiment, không commit vào Git.
Không dùng thời gian ước lượng hoặc file `epoch_20.pth` thay cho process exit:
runner còn có thể đang test hoặc upload.

## 8. Failure recovery

- **Traceback/OOM:** không launch job kế tiếp; giữ log và checkpoint gần nhất.
- **Git pull lỗi:** không dùng source đang có; sửa dirty worktree hoặc remote
  trước.
- **SHA mismatch:** coi là hard failure.
- **Upload lỗi:** model có thể train xong nhưng workflow chưa hoàn tất; retry
  upload, không train lại.
- **Resume:** chỉ dùng khi generated config, optimizer và LR schedule giống
  run bị gián đoạn.
- **Kernel restart:** kiểm tra state JSON và PID trước khi rerun launcher cell.
- **Kết quả:** screening chỉ dùng validation; chỉ test winner đã chọn trước.

## Checklist bàn giao

- Local tests và dry-run pass.
- Diff chỉ chứa file đúng scope.
- Remote SHA được xác nhận.
- Previous process exit code 0.
- Marimo HEAD bằng expected SHA.
- Chỉ một GPU training process.
- Work-dir/HF folder không ghi đè run trước.
- Log, best checkpoint và metrics tồn tại.
- Không có secret trong source, output hoặc tài liệu.
