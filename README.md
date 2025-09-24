
# Backpack Detection (Streamlit + YOLO11)

A simple, themeable Streamlit app that detects **backpacks** in images or videos using **Ultralytics YOLO11**. It draws bounding boxes with confidences, shows a per-frame count overlay in videos, and lets you download the processed video. The UI includes a **Dark/Light theme toggle** and **dynamic logo** (light/dark variants).

> Notes
> - On **Apple Silicon (M1/M2/M3)**, the app will automatically use **PyTorch MPS** (Apple GPU) in a normal macOS Python environment. In Docker containers, MPS is not available; the app will run on CPU there.
> - The first run will download the YOLO11 model weights automatically.

---

## Features

- Image & video upload; draws boxes and confidence for **class 24: backpack (COCO)**  
- Progress bar while processing videos; processed **MP4** playback + **download**  
- **Dark/Light** theme toggle (pure CSS) + **dynamic logo** swap  
- H.264/AVC preferred for in-browser video playback (falls back to `mp4v` if unavailable)

---

## Project structure

```

.
├─ app.py                    # your Streamlit app (paste your final code here)
├─ assets/
│  ├─ slalom-white.png
│  └─ slalom-blue.png
└─ Dockerfile                # (see below)

````

---

## Quick start (local, from scratch)

### 0) Prereqs
- Python **3.11+** (3.10 also OK)
- macOS, Linux, or Windows

### 1) Create & activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
````

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install --upgrade \
  streamlit ultralytics \
  torch torchvision torchaudio \
  opencv-python-headless pillow numpy
```

* YOLO11 ships with the **Ultralytics** package (`pip install ultralytics`).
* PyTorch wheels will be selected for your platform automatically. On **Apple Silicon**, this enables **MPS** in a native macOS environment; you can verify with:

```python
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 3) Run the app

```bash
streamlit run app.py
```

Streamlit defaults to port **8501**; it will open your browser automatically.
If you need a specific host/port:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### 4) Usage tips / troubleshooting

* **Video playback:** browsers prefer **H.264/AVC** for MP4. If your OpenCV build can’t write H.264, the app falls back to `mp4v`. You can optionally install **ffmpeg** and convert the output to H.264 if needed.
* **Upload size:** the default per-file limit is **200 MB** (configurable via `server.maxUploadSize` in Streamlit config).
* **Apple GPU (MPS):** available only in **native macOS Python**, not inside Linux Docker containers.

---

## Optional: requirements.txt

If you prefer:

```
streamlit
ultralytics
torch
torchvision
torchaudio
opencv-python-headless
pillow
numpy
```

Then:

```bash
pip install -r requirements.txt
```

---

## Notes on performance

* Increase `imgsz` for small objects (e.g., `imgsz=896` or `960`).
* On macOS, **MPS** uses the Apple GPU for speedups; no extra code needed beyond your `device = "mps" if torch.backends.mps.is_available() else "cpu"`.

---

## Security

If you bind to `0.0.0.0` to access the app from other machines, ensure you’re running behind a trusted network or proxy and consider authentication.

````

**References:** Streamlit run & config basics, Streamlit in Docker, file-uploader limits, PyTorch MPS, Apple MPS overview, YOLO11 quick start. :contentReference[oaicite:0]{index=0}

---

# 2) Dockerfile (CPU inference, multi-arch, includes FFmpeg)

> MPS (Apple GPU) is **not** available inside a Linux container; this image runs CPU inference. On an M-series Mac, Docker Desktop will pull the **arm64** Python base automatically; on Intel it will use **amd64**.

```dockerfile
# ---- Base image ----
FROM python:3.11-slim

# Prevent Python buffering & keep pip lean
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

# System deps for OpenCV + FFmpeg (H.264 playback/convert), and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# (Optional) install build tools if you add libs needing compilation
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy app code & assets
COPY . /app

# Install Python deps
# You can switch to `COPY requirements.txt . && pip install -r requirements.txt`
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      streamlit ultralytics \
      torch torchvision torchaudio \
      opencv-python-headless pillow numpy

# Expose Streamlit's default port
EXPOSE 8501

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Run the app
CMD ["bash", "-lc", "streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT}"]
````

## Build & run

```bash
# Build (from the folder containing app.py, assets/, Dockerfile)
docker build -t backpack-app .

# Run
docker run --rm -p 8501:8501 backpack-app
# Then open http://localhost:8501
```

> If your `assets/` folder is large and you want to iterate quickly without rebuilding, you can bind-mount it:

```bash
docker run --rm -p 8501:8501 -v "$PWD/assets:/app/assets" backpack-app
```

**References:** Streamlit Docker guide and CLI flags for address/port; notes on OpenCV/FFmpeg H.264 support. ([Streamlit Docs][1])

---

[1]: https://docs.streamlit.io/deploy/tutorials/docker?utm_source=chatgpt.com "Deploy Streamlit using Docker - Streamlit Docs"