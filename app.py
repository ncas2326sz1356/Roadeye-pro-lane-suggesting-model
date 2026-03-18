import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image

from pipeline import RoadEyePipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoadEye Pro",
    page_icon="🚗",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 4px 0;
        border-left: 4px solid #00c9ff;
    }
    .alert-red   { border-left-color: #ff4b4b !important; }
    .alert-green { border-left-color: #00e676 !important; }
    .alert-amber { border-left-color: #ffa726 !important; }
    h1 { color: #00c9ff; }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
        color: #0e1117;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 RoadEye Pro")
st.markdown("**Multi-lane Vehicle Detection · Optical Flow · Driver Assistance**")
st.divider()

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    input_mode = st.radio(
        "Input Source",
        ["📁 Upload Video File", "📷 Webcam"],
        index=0
    )

    st.divider()
    st.markdown("### 🔧 Detection Settings")
    confidence   = st.slider("YOLO Confidence",   0.20, 0.90, 0.40, 0.05)
    px_per_meter = st.slider("Pixels / Metre (calibration)", 5.0, 60.0, 20.0, 1.0)
    show_flow    = st.toggle("Show Optical Flow vectors", value=False)

    st.divider()
    st.markdown("### 📊 Live Stats")
    stat_lanes    = st.empty()
    stat_vehicles = st.empty()
    stat_message  = st.empty()

    st.divider()
    st.markdown("### 🚦 Lane Status")
    stat_left   = st.empty()
    stat_right  = st.empty()
    stat_center = st.empty()

    st.divider()
    st.markdown("### 🏎️ Vehicles by Lane")
    stat_by_lane = st.empty()

    st.divider()
    st.markdown(
        "<small>RoadEye Pro — YOLOv8 + Optical Flow ADAS Demo<br>"
        "⚠️ For research & demonstration only.</small>",
        unsafe_allow_html=True
    )

# ── Main content ──────────────────────────────────────────────────────────────
col_video, col_info = st.columns([3, 1])

with col_video:
    video_placeholder = st.empty()
    progress_bar      = st.empty()
    status_text       = st.empty()

with col_info:
    st.markdown("### 📋 Decision Log")
    decision_log = st.empty()
    log_history  = []

# ── Session state ─────────────────────────────────────────────────────────────
if "running"  not in st.session_state: st.session_state.running  = False
if "pipeline" not in st.session_state: st.session_state.pipeline = None


def make_pipeline():
    return RoadEyePipeline(
        yolo_model   = "yolov8n.pt",
        confidence   = confidence,
        fps          = 30,
        px_per_meter = px_per_meter,
        show_flow    = show_flow,
    )


def update_sidebar(stats):
    lane_emoji = "🛣️" if stats["lane_count"] == 3 else "🛤️"
    stat_lanes.markdown(
        f'<div class="metric-card">{lane_emoji} Lanes: <b>{stats["lane_count"]}</b></div>',
        unsafe_allow_html=True
    )
    stat_vehicles.markdown(
        f'<div class="metric-card">🚙 Vehicles: <b>{stats["vehicles"]}</b></div>',
        unsafe_allow_html=True
    )

    if stats["urgent"]:
        cls = "alert-red"
    elif stats["slow_down"]:
        cls = "alert-amber"
    else:
        cls = "alert-green"

    stat_message.markdown(
        f'<div class="metric-card {cls}">💬 {stats["message"]}</div>',
        unsafe_allow_html=True
    )

    left_icon  = "✅" if stats.get("move_left")  else "❌"
    right_icon = "✅" if stats.get("move_right") else "❌"
    stat_left.markdown(
        f'<div class="metric-card">⬅️ Left lane: {left_icon}</div>',
        unsafe_allow_html=True
    )
    stat_right.markdown(
        f'<div class="metric-card">➡️ Right lane: {right_icon}</div>',
        unsafe_allow_html=True
    )

    # By-lane table
    by_lane = stats.get("by_lane", {})
    if by_lane:
        rows = []
        for lane, vehs in by_lane.items():
            for v in vehs:
                spd = v["speed"]
                arrow = "🔴▲" if spd > 5 else ("🟢▼" if spd < -5 else "🟡~")
                rows.append(f"| {lane} | {v['label']} | {arrow}{abs(spd):.0f} km/h |")
        table = "| Lane | Vehicle | Rel. Speed |\n|------|---------|------------|\n" + "\n".join(rows)
        stat_by_lane.markdown(table)
    else:
        stat_by_lane.markdown("_No vehicles detected_")


# ── Video file mode ───────────────────────────────────────────────────────────
if "Upload" in input_mode:
    uploaded = st.file_uploader(
        "Upload a dashcam video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        start_btn = st.button("▶️ Start Processing", use_container_width=True)
    with col2:
        stop_btn  = st.button("⏹️ Stop",             use_container_width=True)
    with col3:
        reset_btn = st.button("🔄 Reset",             use_container_width=True)

    if reset_btn:
        st.session_state.running  = False
        st.session_state.pipeline = None
        st.rerun()

    if stop_btn:
        st.session_state.running = False

    if start_btn and uploaded:
        st.session_state.running  = True
        st.session_state.pipeline = make_pipeline()

    if st.session_state.running and uploaded:
        # Save upload to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        cap        = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src    = cap.get(cv2.CAP_PROP_FPS) or 30
        pipeline   = st.session_state.pipeline

        frame_idx  = 0
        t_start    = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                status_text.success("✅ Video processing complete!")
                break

            frame_idx += 1
            output = pipeline.process_frame(frame)
            stats  = pipeline.get_stats()

            # Display
            rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            # Progress
            if total_frames > 0:
                progress_bar.progress(
                    min(frame_idx / total_frames, 1.0),
                    text=f"Frame {frame_idx}/{total_frames}"
                )

            # Sidebar
            update_sidebar(stats)

            # Decision log
            msg = stats["message"]
            if not log_history or log_history[-1] != msg:
                log_history.append(msg)
                if len(log_history) > 12:
                    log_history.pop(0)
            decision_log.markdown(
                "\n\n".join([f"• {m}" for m in reversed(log_history)])
            )

        cap.release()
        os.unlink(tfile.name)

    elif not uploaded:
        video_placeholder.info("👆 Upload a dashcam video to get started.")


# ── Webcam mode ───────────────────────────────────────────────────────────────
else:
    st.info("📷 Webcam mode — allow camera access when your browser asks.")

    pipeline = make_pipeline()

    img_file = st.camera_input("Point your camera at the road")

    if img_file is not None:
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(img_file.getvalue()))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output = pipeline.process_frame(frame)
        stats  = pipeline.get_stats()

        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, channels="RGB", use_container_width=True)

        update_sidebar(stats)

        msg = stats["message"]
        if not log_history or log_history[-1] != msg:
            log_history.append(msg)
            if len(log_history) > 12:
                log_history.pop(0)
        decision_log.markdown(
            "\n\n".join([f"• {m}" for m in reversed(log_history)])
        )
