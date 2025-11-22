# Hand Cursor Control (Work in progress...)

<p align="center">
  <img src="logo.png" width="200">
  <br>
  <em></em>
</p>



> Control your mouse cursor using hand gestures and movements (computer vision + ML).

## Project Overview

This repository contains code and resources to build a system that controls the computer cursor using hand gestures and movements captured by a webcam. The system uses computer vision to detect hands and landmarks, translates hand positions/gestures into cursor coordinates and actions (move, click, drag), and provides smoothing and calibration to make control stable and practical.

## Goals

* Real-time cursor control using a standard webcam.
* Support for basic actions: move, left-click, right-click, double-click, drag, scroll.
* Low-latency and smooth cursor motion with configurable smoothing/filtering.
* Simple calibration and sensitivity adjustment for different users and environments.
* Modular design so components (detector, mapper, gesture recognizer) can be swapped or upgraded.

## Key Features

* Hand detection and landmark extraction (e.g. Mediapipe, OpenCV, or custom models).
* Cursor mapping from normalized hand coordinates to screen coordinates.
* Gesture recognition for clicks and special actions.
* Noise reduction (exponential smoothing / Kalman filter) and deadzone support.
* Optional machine-learning model for improved gesture classification.
* Cross-platform cursor control (Windows/Linux/macOS) via platform-appropriate libraries.

## Tech Stack (suggested)

* Python 3.8+
* OpenCV
* MediaPipe (recommended) or a custom hand-pose model (TensorFlow / PyTorch)
* `pyautogui` or `pynput` for cursor control
* `numpy`, `scipy` for processing
* Optional: `torch`/`tensorflow` for custom gesture models

## Repository Structure (suggested)

```
hand-cursor-control/
├── README.md
├── requirements.txt
├── notebooks/                # experiments & demos
├── src/
│   ├── capture.py            # camera capture and frame provider
│   ├── detector.py           # hand detection & landmark extraction wrapper
│   ├── tracker.py            # optional tracking & smoothing (Kalman/EMA)
│   ├── gestures.py           # gesture recognition logic
│   ├── mapper.py             # map landmarks -> screen coordinates
│   ├── controller.py         # cursor control (pyautogui/pynput interface)
│   ├── calibrate.py          # calibration utility
│   └── main.py               # demo application / CLI entrypoint
├── models/                   # optional ML models
├── data/                     # sample data or recorded gestures
├── tests/                    # unit tests
└── docs/                     # design docs and diagrams
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MerakiElysian/Cursor-Control.git
cd Cursor-Control
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> Example `requirements.txt` entries:
>
> ```
> opencv-python
> mediapipe
> numpy
> pyautogui
> pynput
> scipy
> ```

## Quick Start

1. Run the demo (after installing dependencies):

```bash
python src/main.py
```

2. Use the on-screen instructions to calibrate (move your hand to the corners, set sensitivity).
3. Try gestures documented in `docs/gestures.md` (e.g., index-finger point to move, thumb+index pinch to click).

## Calibration

* `calibrate.py` will guide the user to map a comfortable working area in front of the camera to the screen coordinates.
* Save calibration to a JSON file so the application can load user settings.

## Gesture Design (example)

* **Move**: Index finger extended — cursor follows index finger tip.
* **Left Click**: Pinch (index + thumb) or close index and middle fingers.
* **Right Click**: Hold three-finger gesture for 1 second.
* **Double Click**: Quick double pinch.
* **Drag**: Pinch and hold while moving.
* **Scroll**: Two-finger vertical motion or rotate wrist.

Gesture thresholds, timings, and detection logic should be configurable.

## Contributing

Contributions are welcome! Suggested workflow:

1. Fork the repo
2. Create a feature branch `feature/your-feature`
3. Commit changes and open a pull request

Please follow the code style in the project and add tests for new functionality.

## Roadmap / Ideas

* Add ML-based gesture classification for more complex commands.
* Support multi-hand and multi-user scenarios.
* Mobile / browser-based version using WebRTC + TensorFlow.js.
* Firmware integration for external devices (Leap Motion, depth cameras).

## Troubleshooting

* If hand detection is unstable, try improving lighting or switching background.
* Increase the detection confidence thresholds in `detector.py`.
* If cursor lags, reduce smoothing strength or downsample gesture recognition frequency.

## License

This project is available under the MIT License. See `LICENSE` for details.

## Contact

Created by Himanshu Dubey.

---

*Ready to control your cursor with a wave of the hand?*
