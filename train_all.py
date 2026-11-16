"""Automated end-to-end overnight training and evaluation pipeline for CCTV Vehicle Tracking.

Sequentially executes:
1. Standard YOLO26 training on UA-DETRAC.
2. VRAM cleanup and GPU cache flush.
3. Mamba-YOLO (CNN + Visual State Space) training on UA-DETRAC.
4. Evaluation and benchmark comparison across all 4 detector + tracker combinations.
5. Automatic regeneration of academic reports and presentations.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

# Ensure Lab1 directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import BASE_DIR, DEVICE, MODELS_DIR, REPORTS_DIR
from evaluate import run_benchmark, save_reports
from train_mamba import train_mamba_yolo
from train_yolo import train_yolo26


def clean_gpu_memory() -> None:
    """Forces garbage collection and clears PyTorch CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_full_pipeline(
    yolo_epochs: int = 25,
    mamba_epochs: int = 10,
    batch_size: int = 16,
    device: str = DEVICE,
    eval_frames: int = 50,
) -> None:
    """Executes the full automated training and benchmarking suite.

    Args:
        yolo_epochs: Epoch count for Standard YOLO26.
        mamba_epochs: Epoch count for Mamba-YOLO.
        batch_size: Mini-batch size.
        device: CUDA device identifier.
        eval_frames: Frame count for benchmark evaluation.
    """
    total_start = time.time()
    print("=" * 70)
    print("🌙 ЗАПУСК НІЧНОГО КОНВЕЄРА НАВЧАННЯ ТА БЕНЧМАРКУ (UA-DETRAC CCTV)")
    print(f"   • Standard YOLO26: {yolo_epochs} епох")
    print(f"   • Mamba-YOLO:      {mamba_epochs} епох")
    print(f"   • Batch size:      {batch_size}")
    print(f"   • Пристрій:        {device}")
    print("=" * 70)

    # -------------------------------------------------------------
    # ЕТАП 1: Навчання Standard YOLO26
    # -------------------------------------------------------------
    print("\n[Етап 1/4] 🚀 Початок тренування Standard YOLO26...")
    t0 = time.time()
    try:
        train_yolo26(epochs=yolo_epochs, batch_size=batch_size, device=device)
        elapsed_yolo = time.time() - t0
        print(f"[Етап 1/4] ✅ Standard YOLO26 успішно натреновано за {elapsed_yolo / 60:.1f} хв.")
    except Exception as exc:
        print(f"[ПОМИЛКА] Збій під час навчання YOLO26: {exc}")

    # Очищення VRAM перед запуском наступної нейромережі
    print("\n[Очищення VRAM] Звільнення відеопам'яті RTX 5060...")
    clean_gpu_memory()

    # -------------------------------------------------------------
    # ЕТАП 2: Навчання Mamba-YOLO
    # -------------------------------------------------------------
    print(f"\n[Етап 2/4] 🚀 Початок тренування Mamba-YOLO ({mamba_epochs} епох)...")
    t1 = time.time()
    try:
        train_mamba_yolo(epochs=mamba_epochs, batch_size=batch_size, device=device)
        elapsed_mamba = time.time() - t1
        print(f"[Етап 2/4] ✅ Mamba-YOLO успішно натреновано за {elapsed_mamba / 60:.1f} хв.")
    except Exception as exc:
        print(f"[ПОМИЛКА] Збій під час навчання Mamba-YOLO: {exc}")

    clean_gpu_memory()

    # -------------------------------------------------------------
    # ЕТАП 3: Бенчмарк та розрахунок CLEAR MOT метрик
    # -------------------------------------------------------------
    print("\n[Етап 3/4] 📊 Запуск порівняльного бенчмарку на UA-DETRAC...")
    try:
        results = run_benchmark(sequence_prefix="MVI_20012", split="val", num_frames=eval_frames)
        save_reports(results)
        print("[Етап 3/4] ✅ Бенчмарк успішно сформував metrics_summary.csv та графіки.")
    except Exception as exc:
        print(f"[ПОМИЛКА] Збій під час оцінювання метрик: {exc}")

    # -------------------------------------------------------------
    # ЕТАП 4: Оновлення академічних звітів та презентації
    # -------------------------------------------------------------
    print("\n[Етап 4/4] 📝 Оновлення підсумкових звітів Word (.docx) та презентації (.pptx)...")
    report_gen_script = BASE_DIR.parent.parent.parent / ".agents" / "scripts" / "generate_reports.py"
    if report_gen_script.exists():
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            subprocess.run(
                [sys.executable, str(report_gen_script)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            print("[Етап 4/4] ✅ Академічні звіти (.docx) та презентацію (.pptx) оновлено.")
        except Exception as exc:
            print(f"[ПОПЕРЕДЖЕННЯ] Не вдалося згенерувати звіти: {exc}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"🎉 ВСІ ЕТАПИ ЗАВЕРШЕНО ЗАГАЛОМ ЗА {total_elapsed / 3600:.2f} ГОД ({total_elapsed / 60:.1f} ХВ)!")
    print(f"📁 Нові моделі збережено у: {MODELS_DIR}")
    print(f"📊 Звіти та графіки у:      {REPORTS_DIR}")
    print("=" * 70)


def main() -> None:
    """CLI entry point for automated overnight training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-epochs",
        type=int,
        default=25,
        help="Training epochs for Standard YOLO26 (default: 25)",
    )
    parser.add_argument(
        "--mamba-epochs",
        type=int,
        default=10,
        help="Training epochs for Mamba-YOLO (default: 10)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--eval-frames",
        type=int,
        default=50,
        help="Number of frames for benchmark evaluation (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Compute device (default: cuda:0)",
    )
    args = parser.parse_args()

    run_full_pipeline(
        yolo_epochs=args.yolo_epochs,
        mamba_epochs=args.mamba_epochs,
        batch_size=args.batch,
        device=args.device,
        eval_frames=args.eval_frames,
    )


if __name__ == "__main__":
    main()
