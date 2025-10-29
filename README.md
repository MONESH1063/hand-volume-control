Gesture Volume Control

A webcam-based application that allows you to control your system volume using hand gestures.
It uses MediaPipe for hand tracking, OpenCV for video processing, Streamlit for the interface, and Pycaw for Windows audio control.

🚀 Features

Real-time hand gesture detection using webcam

Adjust system volume based on thumb-index finger distance

Interactive Streamlit dashboard with:

▶ Start / ⏸ Pause / ⚙ Settings / 🚪 Logout buttons

Volume trend graph (Plotly)

Current volume %, finger distance, FPS, and response time

Simple login authentication (Username: admin, Password: 1234)

🛠 Installation & Setup

Clone or Download the Repository
cd gesture-volume-control


Create a Virtual Environment

python -m venv venv
venv\Scripts\activate      # (on Windows)


Install Dependencies

pip install -r requirements.txt


Run the Application

streamlit run milestone4_interface.py

🎮 Usage

Login using

Username: admin

Password: 1234

Click “Start” to activate the webcam.

Control volume with gestures:

✋ Open Hand → High Volume

🤏 Pinch → Medium Volume

✊ Closed Hand → Mute / Low Volume

Use ⚙ Settings to adjust min & max finger distance.

Click Pause to stop camera or Logout to exit.

📂 Project Structure
hand-volume-control/
├── milestone4_interface.py
├── requirements.txt
├── README.md
└── venv/

🧰 Tech Stack

Python 3.8+

Streamlit

OpenCV

MediaPipe

Plotly

Pycaw (for Windows system volume control)

comtypes

⚠️ Notes & Limitations

Works best on Windows OS with correct audio drivers.

Webcam permission must be enabled.

If Pycaw fails, app runs in simulation mode (no real volume change).

Frame rate depends on your camera and CPU speed.

👨‍💻 Author

Monesh S
💡 Project: Gesture-Based Volume Control using AI & Computer Vision