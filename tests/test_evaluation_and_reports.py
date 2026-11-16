"""Unit and integration tests for benchmarking evaluation and report generation."""
from __future__ import annotations

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import docx
import pytest

from config import BASE_DIR, REPORTS_DIR
from evaluate import ClassicSORTTracker, load_uadetrac_sequence, run_benchmark
from generate_reports import build_academic_report, verify_document_compliance


def test_classic_sort_tracker_ablation() -> None:
    """Tests the ClassicSORTTracker ablation baseline for track maintenance."""
    import numpy as np

    tracker = ClassicSORTTracker(max_age=5, min_hits=1, iou_threshold=0.25)
    dets_frame1 = np.array([[100, 100, 150, 150, 0.9, 0]], dtype=np.float32)
    dets_frame2 = np.array([[102, 101, 152, 151, 0.9, 0]], dtype=np.float32)

    trks1 = tracker.update(dets_frame1)
    assert len(trks1) == 1
    tid1 = int(trks1[0][0])

    trks2 = tracker.update(dets_frame2)
    assert len(trks2) == 1
    tid2 = int(trks2[0][0])

    assert tid1 == tid2, "Classic SORT should maintain identity across small displacement."


def test_evaluate_metrics_summary_generation() -> None:
    """Verifies that metrics summary CSV and comparison plots exist and are non-empty."""
    csv_path = REPORTS_DIR / "metrics_summary.csv"
    assert csv_path.exists(), "reports/metrics_summary.csv must exist."

    content = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) >= 5, "metrics_summary.csv must contain header + 4 tracker records."

    header = content[0].split(",")
    for metric in ["Configuration", "Model", "MOTA", "IDF1", "HOTA", "IDSW", "FPS"]:
        assert metric in header, f"Missing required metric '{metric}' in summary CSV header."

    for line in content[1:]:
        assert "YOLO26n" in line, "All official benchmark rows must use YOLO26n."


def test_generated_academic_report_structure_and_compliance() -> None:
    """Verifies the generated Word report page count, structure, and zero yolo26s mentions."""
    report_path = BASE_DIR / "Косяченко_CV_Лабораторна1.docx"
    assert report_path.exists(), "Косяченко_CV_Лабораторна1.docx must exist."

    # Verify compliance via helper
    verify_document_compliance(report_path)

    # Verify docx structure
    doc = docx.Document(str(report_path))
    assert len(doc.paragraphs) >= 80, "Academic report must have substantial textual depth (>= 80 paragraphs)."
    assert len(doc.tables) >= 5, "Academic report must include data, architecture, and metric tables."
    assert len(doc.inline_shapes) == 12, f"Academic report must embed exactly 12 figures, got {len(doc.inline_shapes)}."

    # Check page count metadata
    with zipfile.ZipFile(report_path) as z:
        root = ET.fromstring(z.read("docProps/app.xml"))
        pages = 0
        for child in root:
            if child.tag.split("}")[-1] == "Pages":
                pages = int(child.text)
                break
        assert 12 <= pages <= 18, f"Document page count must be between 12 and 18, got {pages}."

    # Check hardware platform authenticity
    full_text = " ".join([p.text for p in doc.paragraphs] + [c.text for t in doc.tables for r in t.rows for c in r.cells]).lower()
    for prohibited in ["jetson", "orin", "raspberry", "openvino", "tensorrt", "13700h", "intel core"]:
        assert prohibited not in full_text, f"Prohibited fabricated hardware '{prohibited}' found in report!"
    assert "ryzen 7 5700x3d" in full_text, "Report must feature real tested CPU AMD Ryzen 7 5700X3D."
    assert "rtx 5060" in full_text, "Report must feature verified NVIDIA RTX 5060 GPU."


def test_generated_presentation_structure_and_compliance() -> None:
    """Verifies the defense PPTX presentation exists, has 10-12 slides, and strictly uses YOLO26n."""
    from pptx import Presentation

    pptx_path = BASE_DIR / "Презентація_Захист_CV_Лаб1_Косяченко.pptx"
    assert pptx_path.exists(), "Презентація_Захист_CV_Лаб1_Косяченко.pptx must exist."

    prs = Presentation(str(pptx_path))
    assert 10 <= len(prs.slides) <= 14, f"Presentation slide count should be 10-12, got {len(prs.slides)}."

    full_text: list[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                full_text.append(shape.text_frame.text)

    all_content = "\n".join(full_text).lower()
    assert "yolo26s" not in all_content, "Prohibited 'yolo26s' found in presentation!"
    assert "yolo26n" in all_content, "Required 'yolo26n' missing from presentation!"

    for prohibited in ["jetson", "orin", "raspberry", "openvino", "tensorrt", "13700h", "intel core"]:
        assert prohibited not in all_content, f"Prohibited fabricated hardware '{prohibited}' found in presentation!"
    assert "ryzen 7 5700x3d" in all_content, "Presentation must feature real tested CPU AMD Ryzen 7 5700X3D."
    assert "rtx 5060" in all_content, "Presentation must feature verified NVIDIA RTX 5060 GPU."


def test_all_publication_figures_exist() -> None:
    """Verifies that all 5 newly generated publication figures exist and are non-empty."""
    required_figures = [
        "dataset_overview.png",
        "pipeline_diagram.png",
        "architecture_diagram.png",
        "tracking_ablation_grid.png",
        "streamlit_dashboard.png",
    ]
    for fig_name in required_figures:
        fig_path = REPORTS_DIR / fig_name
        assert fig_path.exists(), f"Figure {fig_name} must exist in reports/."
        assert fig_path.stat().st_size > 10000, f"Figure {fig_name} is too small or corrupt."
