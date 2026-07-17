import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import time
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(page_title="Cloud ATM Security Monitor", layout="wide")
st.title("🛡️ AI-Powered ATM Surveillance Cloud Dashboard")
st.write("Real-time anomaly and loitering detection powered by YOLOv8.")

# 2. WebRTC configuration for public deployment
# STUN alone often fails across NATs/firewalls on the open internet.
# Add a TURN server (Twilio, Metered.ca, Xirsys, or self-hosted coturn) for reliability.
# Set these as environment variables / Streamlit secrets, not hardcoded.
TURN_URL = st.secrets.get("TURN_URL", "")
TURN_USERNAME = st.secrets.get("TURN_USERNAME", "")
TURN_CREDENTIAL = st.secrets.get("TURN_CREDENTIAL", "")

ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
if TURN_URL:
    ice_servers.append(
        {
            "urls": [TURN_URL],
            "username": TURN_USERNAME,
            "credential": TURN_CREDENTIAL,
        }
    )

RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice_servers})


# 3. Load YOLOv8 Model (Cached so it doesn't reload every frame)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# 4. Video Processing Class (current streamlit-webrtc API)
class ATMVideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.track_history = {}
        self.loitering_threshold = 15  # seconds for demonstration

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.track(img, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            if len(ids) > 1:
                cv2.putText(
                    img,
                    "ALERT: MULTI-PERSON ENCLOSURE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            for box, track_id in zip(boxes, ids):
                if track_id not in self.track_history:
                    self.track_history[track_id] = time.time()
                duration = time.time() - self.track_history[track_id]
                x1, y1, x2, y2 = map(int, box)

                color = (0, 255, 0)
                label = f"ID: {track_id} ({int(duration)}s)"
                if duration > self.loitering_threshold:
                    color = (0, 0, 255)
                    cv2.putText(
                        img,
                        "ALERT: LOITERING",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# 5. Streamlit Component to Display WebRTC Stream
webrtc_streamer(
    key="atm-surveillance",
    video_processor_factory=ATMVideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
