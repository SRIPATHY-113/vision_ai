# 🛡️ AI-Powered ATM Surveillance Cloud Dashboard

Demo link:https://visionai-gofcmkyumreeqtbkda7ezb.streamlit.app/

Real-time anomaly and loitering detection for ATM enclosures, built with
[Streamlit](https://streamlit.io), [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc),
and [YOLOv8](https://github.com/ultralytics/ultralytics).

The app pulls a live video feed from the browser (webcam or connected camera),
runs YOLOv8 object tracking on each frame, and raises on-screen alerts for:

- **Multi-person enclosure** — more than one person detected in frame at once
- **Loitering** — a tracked person remaining in frame past a configurable time threshold

## Features

- Live in-browser video via WebRTC (no server-side camera hardware required)
- Per-person tracking with persistent IDs across frames
- Visual bounding boxes, ID labels, and elapsed dwell time per person
- Runs entirely on YOLOv8n (nano) for CPU-friendly inference

## Demo

Deployed at: `visionai-gofcmkyumreeqtbkda7ezb.streamlit.app`

## Project structure

```
.
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
├── packages.txt      # System (apt) dependencies for Streamlit Community Cloud
├── Dockerfile         # Container build for self-hosted / VM deployment
├── DEPLOY.md          # Detailed deployment guide (Streamlit Cloud, Docker, Cloud Run)
└── README.md          # This file
```

## Quickstart (local)

```bash
git clone <this-repo>
cd vision_ai
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and grant camera
permission when prompted.

## Deployment

See [`DEPLOY.md`](./DEPLOY.md) for full instructions covering:
- Streamlit Community Cloud (fastest path)
- Docker on a cloud VM (AWS/GCP/Azure/DigitalOcean)
- Managed containers (Cloud Run, Fargate)
- Adding a TURN server for reliable WebRTC across NATs/firewalls

## Configuration

| Setting | Where | Purpose |
|---|---|---|
| `loitering_threshold` | `app.py`, `ATMVideoTransformer.__init__` | Seconds before a tracked person triggers a loitering alert (default: 15) |
| `TURN_URL`, `TURN_USERNAME`, `TURN_CREDENTIAL` | Streamlit secrets / env vars | Optional TURN relay for viewers behind restrictive NATs |

## Known environment notes

Streamlit Community Cloud's base image has been tracking Python 3.14 closely,
which is newer than what some pinned computer-vision packages ship wheels for.
If you see build or import errors after a fresh deploy, check first for:

- **apt package name changes** between Debian releases (e.g. `libglib2.0-0` →
  `libglib2.0-0t64`) — fix in `packages.txt`.
- **Missing prebuilt wheels** for pinned versions of `av`, `opencv-python-headless`,
  or `ultralytics` — prefer `>=` minimum-version constraints in `requirements.txt`
  over exact pins so pip can resolve to a release with a compatible wheel.
- **YOLO tracker dependency** — `model.track()` requires `lap` at runtime;
  it must be listed explicitly in `requirements.txt` (Ultralytics' runtime
  auto-install fails on read-only deployment filesystems).

## Compliance note

This app processes live video of identifiable individuals. Recording and
analyzing footage at an ATM may fall under privacy/biometric regulations
depending on jurisdiction (e.g. GDPR, US state biometric privacy laws).
Confirm signage, retention policy, and legal basis for processing before
using this in production — this is separate from the technical setup above.
