"""Streamlit Interactive Web Platform for CCTV Vehicle Detection and Tracking.

Features:
- Live video & CCTV sequence inference with real-time trajectory visualization.
- On-the-fly switching between Standard YOLO26 and Mamba-YOLO detectors.
- On-the-fly switching between SORT (Kalman) and SSM-Tracker.
- Live max_age slider to demonstrate the impact of track coasting on MOTA.
- Metrics dashboard with comparative graphs and theoretical review.
"""
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    BASE_DIR,
    DATA_DIR,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_TRACK_MAX_AGE,
    DEFAULT_TRACK_MIN_HITS,
    MODELS_DIR,
    REPORTS_DIR,
    UA_DETRAC_DIR,
    VEHICLE_CLASSES,
)
from models.baseline_yolo26 import DEFAULT_YOLO26_WEIGHTS, StandardYOLO26
from models.mamba_yolo import MambaYOLO
from modules.sort_tracker import SORTTracker
from modules.ssm_tracker import SSMTracker
from track import get_color

st.set_page_config(
    page_title="CCTV Vehicle Analytics: Mamba-YOLO & SSM Tracking",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Мультиоб'єктний трекінг автомобілів на дорожніх відеопотоках CCTV")
st.markdown(
    """
    **Лабораторна робота №1 з дисципліни «Комп'ютерний зір» (Тема 4) / Дослідницька практика (3 семестр)**  
    *Порівняння Standard YOLO26 (UA-DETRAC CCTV) та Mamba-YOLO (CNN + Visual State Space) з алгоритмами SORT та SSM-Tracker.*
    """
)

# Sidebar configuration
st.sidebar.header("⚙️ Параметри конвеєра")
model_type = st.sidebar.selectbox(
    "Детекційна модель",
    [
        "Standard YOLO26 (UA-DETRAC CCTV Trained) ⭐",
        "Mamba-YOLO (CNN + SS2D, UA-DETRAC)",
        "Standard YOLO26 (COCO Baseline)",
    ],
    index=0,
    help="1. UA-DETRAC CCTV Trained: натренована на повнокадрових дорожніх CCTV-сценах (точно виявляє всі окремі авто на смугах руху vid1/vid2). 2. Mamba-YOLO: гібридний детектор Visual State Space.",
)
tracker_type = st.sidebar.selectbox(
    "Алгоритм відстеження (MOT)",
    ["SSM-Tracker (State Space Memory)", "SORT (Kalman Filter)"],
    index=0,
)
conf_thresh = st.sidebar.slider(
    "Поріг впевненості детектора (Confidence)",
    min_value=0.05,
    max_value=0.95,
    value=float(DEFAULT_CONF_THRESHOLD),
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Налаштування трекінгу та MOTA")
max_age = st.sidebar.slider(
    "Час життя втраченого треку (max_age)",
    min_value=1,
    max_value=30,
    value=int(DEFAULT_TRACK_MAX_AGE),
    step=1,
    help="Кількість кадрів збереження треку без детекцій. Значення > 3 при швидкій зміні ракурсу створює фантомні рамки (False Positives), що знижує MOTA!",
)
min_hits = st.sidebar.slider(
    "Мінімум підтверджень (min_hits)",
    min_value=1,
    max_value=5,
    value=int(DEFAULT_TRACK_MIN_HITS),
    step=1,
    help="Кількість послідовних детекцій перед тим, як трек вважається активним.",
)
show_trails = st.sidebar.checkbox("Відображати шлейфи траєкторій (Trails)", value=True)


# Cache model initializations
@st.cache_resource
def load_detector(name: str):
    if "Mamba" in name:
        return MambaYOLO()
    elif "COCO" in name:
        return StandardYOLO26(weights_path=DEFAULT_YOLO26_WEIGHTS)
    return StandardYOLO26(weights_path=MODELS_DIR / "yolo26_uadetrac_best.pt")


detector = load_detector(model_type)

tab_live, tab_bench, tab_theory = st.tabs(
    ["📹 Живий трекінг відео / CCTV", "📊 Порівняльна аналітика та метрики", "📖 Теоретичні основи"]
)

with tab_live:
    st.subheader("Відеоаналітика дорожнього трафіку")

    source_type = st.radio(
        "Оберіть джерело відеопотоку:",
        [
            "Відеофайли з папки dataset/ (випадаючий список)",
            "Послідовність UA-DETRAC CCTV (MVI_...)",
            "Завантажити новий відеофайл через форму",
        ],
        horizontal=True,
    )

    run_tracking = False
    img_files = []
    video_path_to_run = None
    max_frames_to_run = 1000

    if source_type == "Відеофайли з папки dataset/ (випадаючий список)":
        video_exts = {".mp4", ".mkv", ".avi", ".mov"}
        available_videos = sorted(
            [p for p in DATA_DIR.glob("*") if p.suffix.lower() in video_exts and p.is_file()],
            key=lambda p: p.name,
        )

        if not available_videos:
            st.warning("У папці dataset/ не знайдено відеофайлів.")
        else:
            col_v1, col_v2 = st.columns([2, 1])
            with col_v1:
                selected_video_name = st.selectbox(
                    "Оберіть відеофайл для трекінгу:",
                    [v.name for v in available_videos],
                    index=0,
                )
            selected_video_path = DATA_DIR / selected_video_name

            # Probe video metadata
            cap_probe = cv2.VideoCapture(str(selected_video_path))
            v_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            v_fps = cap_probe.get(cv2.CAP_PROP_FPS)
            v_total = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
            v_dur = v_total / max(1.0, v_fps)
            cap_probe.release()

            st.success(
                f"📹 **Файл:** `{selected_video_name}` | **Роздільна здатність:** {v_w}x{v_h} | "
                f"**FPS:** {v_fps:.1f} | **Всього кадрів:** {v_total} (~{v_dur:.1f} с)"
            )

            with col_v2:
                max_frames_to_run = st.slider(
                    "Кількість кадрів",
                    min_value=10,
                    max_value=max(10, v_total),
                    value=min(v_total, 275),
                    step=10,
                )

            start_vid_btn = st.button(f"▶️ Запустити трекінг {selected_video_name}", type="primary")
            video_path_to_run = str(selected_video_path)
            run_tracking = start_vid_btn

    elif source_type == "Послідовність UA-DETRAC CCTV (MVI_...)":
        seq_img_dir = UA_DETRAC_DIR / "val" / "images"
        available_seqs = ["MVI_20012", "MVI_20051", "MVI_40732", "MVI_39781", "MVI_63554"]

        col_s1, col_s2, col_s3 = st.columns([2, 2, 2])
        with col_s1:
            selected_seq = st.selectbox("Послідовність UA-DETRAC", available_seqs, index=0)
        with col_s2:
            num_frames = st.slider("Кількість кадрів", 10, 100, 50, step=10)
        with col_s3:
            start_seq_btn = st.button("▶️ Запустити аналіз послідовності", type="primary")

        if selected_seq and seq_img_dir.exists():
            img_files = sorted(list(seq_img_dir.glob(f"{selected_seq}_img*.jpg")))[:num_frames]

        run_tracking = start_seq_btn and len(img_files) > 0

    else:
        uploaded_file = st.file_uploader(
            "Завантажте відеофайл дорожнього руху (MP4 / AVI / MKV)", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_file is not None:
            save_dest = DATA_DIR / uploaded_file.name
            if not save_dest.exists():
                save_dest.write_bytes(uploaded_file.getbuffer())
                st.success(f"Файл {uploaded_file.name} збережено до dataset/ і тепер доступний у випадаючому списку!")
            video_path_to_run = str(save_dest)
            run_tracking = st.button("▶️ Запустити аналіз завантаженого відео", type="primary")

    if run_tracking:
        from modules.sort_tracker import KalmanBoxTracker
        from modules.ssm_tracker import SSMTargetState
        KalmanBoxTracker.count = 0
        SSMTargetState.count = 0

        tracker = (
            SSMTracker(max_age=max_age, min_hits=min_hits)
            if "SSM" in tracker_type
            else SORTTracker(max_age=max_age, min_hits=min_hits)
        )
        trail_history: dict[int, list[tuple[int, int]]] = {}

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        m_fps = col_m1.empty()
        m_lat = col_m2.empty()
        m_active = col_m3.empty()
        m_frame = col_m4.empty()

        frame_window = st.image([])
        stop_button = st.button("⏹️ Зупинити обробку")

        if source_type == "Послідовність UA-DETRAC CCTV (MVI_...)":
            for frame_idx, img_file in enumerate(img_files):
                if stop_button:
                    break
                frame = cv2.imread(str(img_file))
                if frame is None:
                    continue

                t0 = time.perf_counter()
                dets = detector.predict(frame, conf=conf_thresh)
                tracks = tracker.update(dets)
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0
                fps = 1000.0 / max(1.0, latency_ms)

                # Visual annotations
                for trk in tracks:
                    track_id = int(trk[0])
                    x1, y1, x2, y2 = [int(v) for v in trk[1:5]]
                    score = float(trk[5])
                    cls_id = int(trk[6])
                    cls_name = VEHICLE_CLASSES.get(cls_id, "car")

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if track_id not in trail_history:
                        trail_history[track_id] = []
                    trail_history[track_id].append((cx, cy))
                    if len(trail_history[track_id]) > 25:
                        trail_history[track_id].pop(0)

                    color = get_color(track_id)
                    if show_trails:
                        pts = trail_history[track_id]
                        for i in range(1, len(pts)):
                            cv2.line(frame, pts[i - 1], pts[i], color, 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"#{track_id} {cls_name} {score:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                    )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, use_container_width=True)

                m_fps.metric("Швидкодія (FPS)", f"{fps:.1f}")
                m_lat.metric("Затримка інференсу", f"{latency_ms:.1f} мс")
                m_active.metric("Активних авто", f"{len(tracks)}")
                m_frame.metric("Поточний кадр", f"{frame_idx + 1} / {len(img_files)}")
                time.sleep(0.01)

        elif video_path_to_run:
            cap = cv2.VideoCapture(video_path_to_run)
            frame_idx = 0

            while cap.isOpened() and not stop_button and frame_idx < max_frames_to_run:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.perf_counter()
                dets = detector.predict(frame, conf=conf_thresh)
                tracks = tracker.update(dets)
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0
                fps = 1000.0 / max(1.0, latency_ms)

                for trk in tracks:
                    track_id = int(trk[0])
                    x1, y1, x2, y2 = [int(v) for v in trk[1:5]]
                    score = float(trk[5])
                    cls_id = int(trk[6])
                    cls_name = VEHICLE_CLASSES.get(cls_id, "car")

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if track_id not in trail_history:
                        trail_history[track_id] = []
                    trail_history[track_id].append((cx, cy))
                    if len(trail_history[track_id]) > 25:
                        trail_history[track_id].pop(0)

                    color = get_color(track_id)
                    if show_trails:
                        pts = trail_history[track_id]
                        for i in range(1, len(pts)):
                            cv2.line(frame, pts[i - 1], pts[i], color, 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"#{track_id} {cls_name} {score:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                    )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, use_container_width=True)

                m_fps.metric("Швидкодія (FPS)", f"{fps:.1f}")
                m_lat.metric("Затримка інференсу", f"{latency_ms:.1f} мс")
                m_active.metric("Активних авто", f"{len(tracks)}")
                m_frame.metric("Поточний кадр", f"{frame_idx + 1} / {max_frames_to_run}")
                frame_idx += 1
                time.sleep(0.01)

            cap.release()

    elif source_type == "Послідовність UA-DETRAC CCTV (MVI_...)":
        st.info("💡 Натисніть кнопку «▶️ Запустити аналіз послідовності» для покадрового відтворення та трекінгу.")
    else:
        st.info("💡 Завантажте відеопотік з камери відеоспостереження або оберіть одне з доступних джерел вище.")

with tab_bench:
    st.subheader("Результати експериментального порівняння (Benchmark)")
    csv_path = REPORTS_DIR / "metrics_summary.csv"
    plot_path = REPORTS_DIR / "benchmark_comparison.png"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(
            df.style.highlight_max(subset=["MOTA", "IDF1", "FPS"], color="#d4edda").format(
                {"Latency_ms": "{:.2f}", "FPS": "{:.1f}", "MOTA": "{:.3f}", "IDF1": "{:.3f}"}
            ),
            use_container_width=True,
        )
    else:
        st.warning("Файл metrics_summary.csv ще не згенеровано. Запустіть evaluate.py.")

    if plot_path.exists():
        st.image(
            str(plot_path),
            caption="Порівняльний аналіз швидкодії (FPS) та точності асоціації (MOTA / IDF1)",
            use_container_width=True,
        )

    with st.expander("🔍 Чому показник MOTA критично залежить від параметра max_age?"):
        st.markdown(
            r"""
            Метрика **MOTA (Multi-Object Tracking Accuracy)** обчислюється як:
            $$\text{MOTA} = 1 - \frac{\sum_t (\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}$$

            1. **Точність детектора на UA-DETRAC CCTV:**  
               Детектор YOLO26, натренований на 17 600+ кадрах UA-DETRAC, досягає $\text{mAP}_{50} = 90.6\%$ та $\text{Recall} = 81.5\%$ на реальних багатооб'єктних дорожніх сценах.
            2. **Фантомні треки ($\text{FP}$):**  
               При завеликому $\text{max\_age} = 15 \dots 30$ фільтр Калмана або SSM за інерцією продовжує прогнозувати координати авто навіть після його повороту або виїзду за межі кадру. Це генерує хибні рамки ($\text{FP}$), що зменшує MOTA.
            3. **Оптимальний баланс:**  
               При $\text{max\_age} = 2$ трекер ефективно переживає короткочасні оклюзії без генерації фантомних рамок.
            """
        )

with tab_theory:
    st.subheader("Математичне формулювання моделей простору станів (SSM)")
    st.markdown(
        r"""
        Неперервна селективна модель простору станів задається системою диференціальних рівнянь:
        $$h'(t) = \mathbf{A} h(t) + \mathbf{B} x(t)$$
        $$y(t) = \mathbf{C} h(t) + \mathbf{D} x(t)$$

        де $\mathbf{A} \in \mathbb{R}^{N \times N}$ — матриця переходу стану, а дискретизація за нульовим порядком екстраполятора (ZOH) з кроком $\Delta$ визначається як:
        $$\overline{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \overline{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$
        $$h_t = \overline{\mathbf{A}} h_{t-1} + \overline{\mathbf{B}} x_t, \quad y_t = \mathbf{C} h_t + \mathbf{D} x_t$$

        **2D Visual State Space (SS2D):**
        У гібридному блоці `Mamba-YOLO` карта просторових ознак розгортається у 4 ортогональні напрямки:
        1. Зліва направо (Row-major forward)
        2. Справа наліво (Row-major reverse)
        3. Зверху вниз (Column-major forward)
        4. Знизу вгору (Column-major reverse)

        Це забезпечує глобальне рецептивне поле з лінійною складністю $\mathcal{O}(H \cdot W)$ проти квадратичної $\mathcal{O}((HW)^2)$ у стандартних трансформерах Vision Transformer (ViT).
        """
    )
