# Hand Gesture Recognition for Media Controls (Capstone 27)

A real-time, privacy-conscious computer vision application that maps human hand gestures to standard OS-level media playback controls using Google MediaPipe and OpenCV.

---

## 1. Project Overview & Scope
- **Problem Statement**: Can a lightweight, camera-based system recognize distinct hand gestures to manage non-critical media playback with clear error handling?
- **Scope**: Controls non-destructive media actions (Play/Pause, Next Track, Volume Up, Mute).
- **Core Stack**: Python 3.12+, MediaPipe Tasks (Vision), OpenCV, PyAutoGUI, scikit-learn.

## 2. Gesture Mapping & Interaction Design
| Gesture | Pose Definition | OS Action | Virtual Key |
| :--- | :--- | :--- | :--- |
| **Open Palm** | All 5 fingers extended upright | Play / Pause Toggle | `playpause` |
| **Fist** | All 4 fingers curled into palm | Mute / Unmute Audio | `volumemute` |
| **V-Sign** | Index and Middle extended, Ring and Pinky curled | Next Track | `nexttrack` |
| **Thumbs Up** | Thumb pointing upward, other 4 fingers curled | Volume Increment | `volumeup` |
| **Neutral** | Hand at rest or transitional movement | No Action | None |

## 3. Debounce & Error Handling
- **Temporal Smoothing**: Gestures must remain held continuously for at least `0.5s` to confirm recognition.
- **Event Latch & Cooldown**: Once triggered, a gesture is latched so holding it does not fire repeated duplicate events. A minimum `1.2s` cooldown is enforced before another action can be fired.
- **Safe Lifecycle**: All camera pipelines and MediaPipe Landmarker instances are bound within `try ... finally` blocks to guarantee resource release.

## 4. Privacy & Ethical Safeguards
- **In-Memory Frame Analysis**: All image tensors exist solely in volatile RAM during inference and are dropped immediately after landmark extraction.
- **Zero Disk/Network Persistence**: No webcam frames, video files, or microphone audio are recorded or transmitted off-device.
- **User Agency**: The active capture window clearly displays on-screen tracking status and exits immediately upon pressing `q`.

## 5. Setup & Execution

### Prerequisites
- Python 3.10+ (compatible with Python 3.12 / 3.13)
- Dedicated or integrated webcam

### Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt