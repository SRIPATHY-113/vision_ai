import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import time
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(page_title="Cloud ATM Security Monitor", layout="wide")
st.title("🛡️ AI-Powered ATM Surveillance Cloud Dashboard")
st.write("Real-time anomaly and loitering detection powered by YOLOv8.")

# 2. WebRTC configuration for public deployment (uses Google STUN servers)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 3. Load YOLOv8 Model (Cached so it doesn't reload every frame)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 4. Video Processing Class
class ATMVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.track_history = {}
        self.loitering_threshold = 15  # seconds for demonstration

    def transform(self, frame):
        # Convert WebRTC frame to standard OpenCV BGR image
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 tracking
        results = model.track(img, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            # Multi-person logic
            if len(ids) > 1:
                cv2.putText(img, "ALERT: MULTI-PERSON ENCLOSURE", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Loitering logic per ID
            for box, track_id in zip(boxes, ids):
                if track_id not in self.track_history:
                    self.track_history[track_id] = time.time()

                duration = time.time() - self.track_history[track_id]
                x1, y1, x2, y2 = map(int, box)
                
                color = (0, 255, 0)
                label = f"ID: {track_id} ({int(duration)}s)"

                if duration > self.loitering_threshold:
                    color = (0, 0, 255)
                    cv2.putText(img, "ALERT: LOITERING", (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

# 5. Streamlit Component to Display WebRTC Stream
webrtc_streamer(
    key="atm-surveillance",
    video_transformer_factory=ATMVideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)