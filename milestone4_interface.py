# milestone4_interface.py
import streamlit as st
import cv2
import numpy as np
import math
import time
import plotly.graph_objects as go
import mediapipe as mp
import comtypes

# Ensure COM initialized for pycaw on Windows
comtypes.CoInitialize()

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYAUDIO_AVAILABLE = True
except Exception:
    PYAUDIO_AVAILABLE = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Gesture Volume Control", layout="wide")
st.markdown(
    """
    <style>
    .title {text-align:center; font-size:28px; font-weight:700; color:#FF6A3D; margin-bottom:6px;}
    .card {background:#fff; border-radius:10px; padding:12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);}
    .small-muted {color:#6b7280; font-size:13px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SESSION DEFAULTS ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "running" not in st.session_state:
    st.session_state.running = False
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []
if "min_dist" not in st.session_state:
    st.session_state.min_dist = 40
if "max_dist" not in st.session_state:
    st.session_state.max_dist = 200
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

# --- Global Variables for Audio Control ---
volume_ctrl = None
vol_range = None


# ---------------- AUDIO INIT ----------------
# put these imports near top of file if not already:
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
# pycaw imports assumed available

def init_audio():
    """Return (volume_ctrl, vol_range) or (None, None). Tries multiple pycaw methods."""
    try:
        import comtypes
        comtypes.CoInitialize()
    except Exception:
        pass

    try:
        # 1) Preferred: direct GetAudioEndpointVolume (if available)
        try:
            from pycaw.utils import AudioUtilities as _AU
            vol = _AU.GetAudioEndpointVolume()
            rng = vol.GetVolumeRange()
            return vol, (rng[0], rng[1])
        except Exception:
            pass

        # 2) Classic: GetSpeakers -> Activate or _ctl.QueryInterface
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        try:
            # try Activate first (works on many versions)
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        except Exception:
            # fallback to internal _ctl if Activate not present
            if hasattr(devices, "_ctl"):
                interface = devices._ctl.QueryInterface(IAudioEndpointVolume)
            else:
                # last fallback: try attribute iid usage
                interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        rng = volume.GetVolumeRange()
        return volume, (rng[0], rng[1])
    except Exception as e:
        # Do NOT crash app; return None so app can simulate
        print("init_audio failed:", repr(e))
        return None, None


# ---------------- LOGIN PAGE ----------------
def show_login():
    st.markdown("<div class='title'>👋 Welcome — Gesture Volume Control</div>", unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.success("✅ Login successful")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Invalid credentials (use admin / 1234)")

# ---------------- DASHBOARD ----------------
def show_dashboard():
    st.markdown("<div class='title'>🎛 Gesture Volume Control — Dashboard</div>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([1.5,1,1,1,1])
    with col1:
        st.markdown("<div class='small-muted'>Control system volume using hand gestures.</div>", unsafe_allow_html=True)
    with col2:
        if st.button("▶ Start"):
            st.session_state.running = True
    with col3:
        if st.button("⏸ Stop"):
            st.session_state.running = False
    with col4:
        if st.button("⚙ Settings"):
            st.session_state.show_settings = not st.session_state.show_settings
            st.rerun()
    with col5:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.running = False
            st.rerun()

    # ----- SETTINGS -----
    if st.session_state.show_settings:
        st.markdown("### ⚙ Settings")
        st.session_state.min_dist = st.slider("Minimum Finger Distance (px)", 10, 100, st.session_state.min_dist)
        st.session_state.max_dist = st.slider("Maximum Finger Distance (px)", 120, 400, st.session_state.max_dist)
        if st.button("✅ Save Settings"):
            st.success("Settings updated successfully.")
            time.sleep(0.3)
            st.session_state.show_settings = False
            st.rerun()
        st.markdown("---")

    # Layout split
    cam_col, info_col = st.columns([2,1])
    cam_col.markdown("### 🎥 Live Camera")
    camera_ph = cam_col.empty()
    chart_ph = cam_col.empty()
    info_col.markdown("### ✋ Gesture Recognition")
    gesture_box = info_col.empty()
    info_col.markdown("### 📊 Performance Metrics")
    perf_box = info_col.empty()

    gesture_box.info("No gesture detected yet.")
    perf_box.info("Metrics will appear once the camera starts.")

    if st.session_state.running:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65, min_tracking_confidence=0.65)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Camera not accessible.")
            st.session_state.running = False
            return

        start_time = time.time()
        frame_count = 0
        hist = st.session_state.volume_history

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = "No Hand"
            distance = 0
            vol_percent = 0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
                x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                distance = math.hypot(x2 - x1, y2 - y1)

                min_d = st.session_state.min_dist
                max_d = st.session_state.max_dist
                vol_percent = np.clip(np.interp(distance, [min_d, max_d], [0, 100]), 0, 100)

                if volume_ctrl and vol_range:
                    try:
                        db_val = np.interp(distance, [min_d, max_d], [vol_range[0], vol_range[1]])
                        volume_ctrl.SetMasterVolumeLevel(float(db_val), None)
                    except:
                        pass

                if distance < (min_d + 10):
                    gesture = "Closed"
                elif distance > (max_d - 20):
                    gesture = "Open Hand"
                else:
                    gesture = "Pinch"

                cv2.circle(frame, (x1, y1), 8, (255, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 8, (255, 0, 255), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{int(vol_percent)}%", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            camera_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            g_html = f"""
            <div class="card">
            <b>Open Hand:</b> {'✅' if gesture=='Open Hand' else '—'}<br>
            <b>Pinch:</b> {'✅' if gesture=='Pinch' else '—'}<br>
            <b>Closed:</b> {'✅' if gesture=='Closed' else '—'}<br>
            </div>
            """
            gesture_box.markdown(g_html, unsafe_allow_html=True)

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            perf_html = f"""
            <div class="card">
            <b>🔊 Current Volume:</b> {int(vol_percent)}%<br>
            <b>📏 Finger Distance:</b> {int(distance)} px<br>
            <b>⚡ Response Time:</b> {elapsed:.2f} s<br>
            <b>FPS:</b> {fps:.1f}
            </div>
            """
            perf_box.markdown(perf_html, unsafe_allow_html=True)

            hist.append(float(vol_percent))
            if len(hist) > 100:
                hist.pop(0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hist, mode="lines", line=dict(color="#FF6A3D", width=3)))
            fig.update_layout(title="📈 Volume Trend", yaxis=dict(range=[0,100]),
                              template="plotly_white", height=300)
            chart_ph.plotly_chart(fig, use_container_width=True)


            time.sleep(0.05)

        cap.release()
        cv2.destroyAllWindows()
        st.session_state.running = False
        st.success("🛑 Camera stopped.")

# ---------------- MAIN ----------------
def main():
    global volume_ctrl, vol_range
    if not st.session_state.logged_in:
        show_login()
    else:
        if volume_ctrl is None or vol_range is None:
            volume_ctrl, vol_range = init_audio()
        show_dashboard()


if __name__ == "__main__":
    main()

















