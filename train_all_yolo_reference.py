import os
import sys
import subprocess
import shutil
import csv
import argparse
from pathlib import Path

ROOT = Path("/marimo/yolo_code").resolve()
LOCAL = ROOT / "models_related" / "ultralytics"

# Đưa local ultralytics lên đầu sys.path
sys.path.insert(0, str(LOCAL))

# Xóa toàn bộ module ultralytics đã cache
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]

# Import lại từ đầu
import ultralytics
from ultralytics.utils import DEFAULT_CFG_DICT

print("ultralytics path:", ultralytics.__file__)
print("bbox_iou_loss:", "bbox_iou_loss" in DEFAULT_CFG_DICT)
print("wiou_monotonous:", "wiou_monotonous" in DEFAULT_CFG_DICT)
subprocess.run(
    [
        "python",
        "/marimo/yolo_code/prepare_dataset.py",
        "--root", "/marimo/data",
        "--out-dir", "datasets/varroa_yolo",
    ],
    check=True,
)
from ultralytics import YOLO


def find_best_weights(root: Path) -> list[Path]:
    """Find trained best.pt files across known Ultralytics output roots."""
    train_roots = [
        root / "runs/detect/runs/detect/yolo_related/runs/train",
        root / "runs/detect/yolo_related/runs/train",
        root / "runs/detect/train",
        Path("runs/detect/yolo_related/runs/train"),
        Path("runs/detect/train"),
    ]
    best_paths = []
    seen = set()
    for train_root in train_roots:
        for best_path in train_root.glob("*/weights/best.pt"):
            resolved = best_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                best_paths.append(best_path)
    return sorted(best_paths)


def run_test_inference(root: Path, data_test: str) -> Path | None:
    """Evaluate all discovered best.pt checkpoints on the test split and save a summary CSV."""
    test_root = root / "runs/detect/yolo_related/runs/test"
    best_paths = find_best_weights(root)

    print(f"Found {len(best_paths)} models")
    if not best_paths:
        return None

    results = []
    for best_path in best_paths:
        run_name = best_path.parents[1].name

        print("\n" + "=" * 60)
        print(f"Testing: {run_name}")
        print(f"Weight: {best_path}")
        print("=" * 60)

        model = YOLO(str(best_path))
        metrics = model.val(
            data=data_test,
            split="test",
            imgsz=640,
            batch=16,
            device="cuda",
            conf=0.001,
            iou=0.5,
            project=str(test_root),
            name=run_name,
            exist_ok=True,
        )

        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)

        results.append((run_name, map50, map5095, precision, recall))

        print(f"mAP50:    {map50:.4f}")
        print(f"mAP50-95: {map5095:.4f}")
        print(f"Precision:{precision:.4f}")
        print(f"Recall:   {recall:.4f}")

    print("\n\n===== SUMMARY =====")
    results = sorted(results, key=lambda x: x[1], reverse=True)

    for run_name, map50, map5095, precision, recall in results:
        print(
            f"{run_name:45s} | "
            f"mAP50={map50:.4f} | "
            f"mAP50-95={map5095:.4f} | "
            f"P={precision:.4f} | "
            f"R={recall:.4f}"
        )

    test_root.mkdir(parents=True, exist_ok=True)
    summary_path = test_root / "test_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
        writer.writerows(results)
    print(f"\nSaved test summary: {summary_path}")
    return test_root


def upload_runs_to_hf(root: Path, part_name: str, hf_token: str | None = None, repo_id: str | None = None) -> None:
    """Upload train/test outputs to Hugging Face when HF_TOKEN is available."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is not set; skipping Hugging Face upload.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed; skipping Hugging Face upload.")
        return

    repo_id = repo_id or os.environ.get("HF_REPO_ID", f"duyle2408/varroa-yolo-{part_name}-yolov8{MODEL_SCALE}")
    folder_path = root / "runs/detect/yolo_related/runs"
    if not folder_path.exists():
        print(f"Upload folder does not exist: {folder_path}")
        return

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    print(f"Uploading {folder_path} to https://huggingface.co/datasets/{repo_id}")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
    else:
        api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
    print("Hugging Face upload complete.")

# ==========================================
# CẤU HÌNH SIZE MÔ HÌNH Ở ĐÂY (n, s, m, l, x)
MODEL_SCALE = "n"
# ==========================================

# Đường dẫn tới thư mục chứa các file config
config_dir = Path("/marimo/yolo_code/models_related/models_config")
data_path = "/marimo/data/datasets/varroa_yolo/varroa.yaml"

KVPLACE_CONFIGS = [
    "yolov8_varroa_kvheads_repc2f.yaml",
    "yolov8_varroa_kvheads_repc2f_ensimam_p5.yaml",
    "yolov8_varroa_kvheads_repc2f_ensimam_p2p3.yaml",
    "yolov8_varroa_kvheads_repc2f_bifpn.yaml",
    "yolov8_varroa_kvheads_repc2f_bifpn_ensimam_p2p3.yaml",
    "yolov8_varroa_kvheads_repc2f_bifpn_asf.yaml",
    "yolov8_varroa_kvheads_repc2f_bifpn_asf_ensimam.yaml",
]

SPECIAL_CONFIGS = set(KVPLACE_CONFIGS)

# Lấy danh sách tất cả các file yaml gốc (chỉ lấy các file bắt đầu bằng yolov8_ để bỏ qua các file yolov8x_ sinh ra)
all_configs = sorted([
    f for f in os.listdir(config_dir) 
    if f.endswith('.yaml') and f.startswith('yolov8_') and f not in SPECIAL_CONFIGS
])

# Chia thành 2 nửa
part1_configs = all_configs[:7]
part2_configs = all_configs[7:]

parser = argparse.ArgumentParser()
parser.add_argument(
    "part",
    nargs="?",
    default="1",
    choices=("1", "2", "kvplace", "kvplace1", "kvplace2"),
    help="Config group to train.",
)
parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
args = parser.parse_args()
part = args.part

if part == "1":
    configs_to_run = part1_configs
    print(f"[*] ĐANG CHẠY PHẦN 1: {len(configs_to_run)} configs")
elif part == "2":
    configs_to_run = part2_configs
    print(f"[*] ĐANG CHẠY PHẦN 2: {len(configs_to_run)} configs")
elif part == "kvplace":
    configs_to_run = KVPLACE_CONFIGS
    print(f"[*] ĐANG CHẠY KVPLACE YOLOv8{MODEL_SCALE}: {len(configs_to_run)} configs")
elif part == "kvplace1":
    configs_to_run = KVPLACE_CONFIGS[:3]
    print(f"[*] ĐANG CHẠY KVPLACE 1 YOLOv8{MODEL_SCALE}: {len(configs_to_run)} configs")
elif part == "kvplace2":
    configs_to_run = KVPLACE_CONFIGS[3:]
    print(f"[*] ĐANG CHẠY KVPLACE 2 YOLOv8{MODEL_SCALE}: {len(configs_to_run)} configs")

# Chạy vòng lặp qua từng config
for config_name in configs_to_run:
    print("\n" + "="*50)
    print(f"[*] Bắt đầu training với config: {config_name}")
    print("="*50 + "\n")
    
    original_yaml = config_dir / config_name
    
    # Tạo tên file mới có scale tương ứng (vd: yolov8_ -> yolov8x_)
    new_config_name = config_name.replace("yolov8_", f"yolov8{MODEL_SCALE}_")
    model_yaml = config_dir / new_config_name
    
    # Tự động copy nội dung sang file yaml mới
    shutil.copy(original_yaml, model_yaml)
    
    # Truyền file có chữ 'x' vào YOLO
    model = YOLO(str(model_yaml))
    
    # Load đúng file pretrained weights theo scale
    model.load(f"yolov8{MODEL_SCALE}.pt",smart_transfer=True) 
    
    # Đặt tên run bỏ đuôi .yaml
    run_name = new_config_name.replace('.yaml', '')
    
    model.train(
        data=data_path,
        epochs=200,
        imgsz=640,
        batch=16,
        workers=4,
        device="cuda",

        patience=40,

        # box=10.0,
        # dfl=2.0,
        # cls=0.25,
        bbox_iou_loss="wiou",
        wiou_monotonous=False,
        # EXPERIMENTAL: uncomment for boundary-aware contrastive localization ablation.
        # boundary_contrast=0.05,
        # boundary_levels=2,
        # boundary_ring=1.0,
        # boundary_samples=16,
        # boundary_tau=0.2,

        # mosaic=0.2,
        # close_mosaic=40,
        # mixup=0.0,
        # copy_paste=0.0,
        # degrees=0.0,
        # perspective=0.0,
        # translate=0.1,
        # scale=0.25,
        # shear=0.0,

        # optimizer="AdamW",
        # lr0=0.001,
        # lrf=0.01,
        # cos_lr=True,
        # weight_decay=0.0005,
        # warmup_epochs=3,

        project=str(ROOT / "runs/detect/yolo_related/runs/train"),
        name=run_name,
    )

test_output_root = run_test_inference(ROOT, data_path)
if test_output_root is not None:
    upload_runs_to_hf(ROOT, part, hf_token=args.hf_token, repo_id=args.hf_repo_id)
