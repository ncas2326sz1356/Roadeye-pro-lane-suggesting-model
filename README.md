# 🚗 RoadEye Pro

**Multi-lane Vehicle Detection · Optical Flow · Driver Assistance System**

> A dashcam-based ADAS (Advanced Driver Assistance System) built with YOLOv8 + Optical Flow that detects lanes, tracks vehicles, estimates relative speeds, and gives real-time lane-change suggestions.

---

## 🖥️ Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🛣️ Lane Detection | Detects 2-lane and 3-lane roads automatically using Hough Transform |
| 🚙 Vehicle Detection | YOLOv8 detects cars, trucks, buses, motorcycles in real time |
| 💨 Optical Flow | Farneback dense flow computes per-vehicle motion vectors |
| ⚡ Relative Speed | Estimates approaching/receding speed of each vehicle |
| 🧠 Decision Engine | Suggests: Slow down / Move Left / Move Right / Stay in lane |
| 🖥️ HUD Overlay | Clean heads-up display overlaid on the video |

---

## 📁 Project Structure

```
roadeye_pro/
├── app.py                    # Streamlit main app
├── pipeline.py               # Master pipeline
├── requirements.txt
├── modules/
│   ├── lane_detector.py      # Hough-based lane line detection
│   ├── vehicle_detector.py   # YOLOv8 vehicle detection
│   ├── speed_estimator.py    # Optical flow + relative speed
│   ├── decision_engine.py    # Lane-change decision logic
│   └── hud.py                # HUD overlay drawing
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/roadeye-pro.git
cd roadeye-pro
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run locally
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repo → `app.py`
4. Click **Deploy** ✅

> **Note:** First run will auto-download `yolov8n.pt` (~6MB). Streamlit Cloud handles this automatically.

---

## 🎮 How to Use

### Video File Mode
1. Click **Upload Video File**
2. Upload any dashcam `.mp4` / `.avi` / `.mov` video
3. Click **▶️ Start Processing**
4. Watch the HUD overlay with lane detection + vehicle tracking

### Webcam Mode
1. Select **📷 Webcam**
2. Click **▶️ Start Webcam**
3. Point your camera at a road (or a screen playing dashcam footage)

---

## 🧠 How It Works

```
Dashcam Video Frame
       ↓
Lane Line Detection (Hough Transform)
→ Defines LEFT / CENTER / RIGHT zones
       ↓
YOLOv8 Vehicle Detection
→ Bounding boxes assigned to lane zones
       ↓
Farneback Optical Flow
→ Per-vehicle motion vectors
→ Relative speed = vehicle flow − ego/background flow
       ↓
Decision Engine
→ Checks: ahead vehicle approaching? Adjacent lanes clear?
→ Output: Slow down / Move Left / Move Right
       ↓
HUD Overlay drawn on frame
```

---

## ⚙️ Calibration

| Setting | Default | Notes |
|---|---|---|
| YOLO Confidence | 0.40 | Lower = more detections, more false positives |
| Pixels/Metre | 20.0 | Tune to your camera's perspective for accurate speed |
| Show Optical Flow | Off | Enable to visualize motion vectors |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **YOLOv8** — Ultralytics
- **OpenCV** — Lane detection + Optical Flow
- **Streamlit** — Web UI + deployment
- **NumPy** — Array operations

---

## ⚠️ Disclaimer

This project is for **research and demonstration purposes only**. It is not a certified safety system and should NOT be used as a sole basis for driving decisions.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Contributing

Pull requests welcome! Open an issue for bugs or feature ideas.

---

*Built with ❤️ using YOLOv8 + OpenCV + Streamlit*
