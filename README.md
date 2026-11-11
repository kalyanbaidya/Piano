# 🎹 Gesture-Controlled Musical Instruments

Turn your hand gestures into music! This project features two fun, interactive, gesture-controlled musical instruments built using Python, OpenCV, MediaPipe, NumPy, and Pygame.

Using a webcam, the system tracks your hands, recognizes finger gestures, and converts them into musical notes and chords in real time.

No physical keyboard, piano, or guitar is required. Just your hands and a webcam!

## 🎵 Projects Included

### 1. Piano Gesture – Play Piano with Your Fingers

A virtual two-octave C Major piano controlled entirely through hand gestures.

**Features:**
- Real-time hand tracking using MediaPipe.
- Recognizes the number of extended fingers.
- Each finger count selects a different musical note.
- Two octaves available: C4–C5 and C5–C6.
- Move your hand into the upper or lower camera region to switch octaves.
- Fist gesture (0 extended fingers) acts as idle.
- Live camera feed with hand landmark and skeleton visualization.
- Dynamically generated piano sounds using harmonic sine waves.
- Keyboard controls available as an alternative to gestures.

**Gesture Controls:**

| Extended Fingers | Piano Note |
|---|---|
| 1 | C |
| 2 | D |
| 3 | E |
| 4 | F |
| 5 | G |
| 6 | A |
| 7 | B |
| 8 | C (Higher) |
| 0 (Fist) | Idle |

**Octave Selection:**

| Hand Position in Camera | Octave |
|---|---|
| Upper half | High Octave (C5–C6) |
| Lower half | Low Octave (C4–C5) |

The system tracks up to two hands and adds their extended-finger counts together to select the note.

### 2. Chord Gesture – Play Guitar with Your Hands

A virtual acoustic guitar that lets you play major and minor chords using two-hand gestures.

One hand controls the chords, while the other controls the capo position, just like playing a real guitar.

**Features:**
- Real-time tracking of both hands.
- Right-hand gestures select guitar chords.
- Left-hand gestures control the capo position.
- Eight major and eight minor chord options.
- Upper and lower camera regions select chord types.
- Fist gesture on the left hand resets the capo to fret 0.
- Dynamically generated acoustic guitar-like sounds.
- Simulated string strumming with staggered string activation.
- Capo changes shift chord frequencies.
- Live camera feed with colour-coded hand tracking.
- Keyboard controls available as an alternative.

**Right-Hand Controls – Chord Selection:**

| Gesture ID | Finger Pattern | Chord |
|---|---|---|
| 1 | Index | A |
| 2 | Index + Middle | B |
| 3 | Index + Middle + Ring | C |
| 4 | Index + Middle + Ring + Pinky | D |
| 5 | All five fingers | E |
| 6 | Thumb | F |
| 7 | Thumb + Index | G |
| 8 | Thumb + Index + Middle | A (Higher) |

**Chord Type Selection:**

| Right-Hand Position | Chord Type |
|---|---|
| Upper half | Minor Chords |
| Lower half | Major Chords |

**Left-Hand Controls – Capo Selection:**

| Gesture ID | Finger Pattern | Capo Position |
|---|---|---|
| 0 | Fist | Fret 0 (Open) |
| 1 | Index | Fret 1 |
| 2 | Index + Middle | Fret 2 |
| 3 | Index + Middle + Ring | Fret 3 |
| 4 | Four fingers | Fret 4 |
| 5 | All five fingers | Fret 5 |
| 6 | Thumb | Fret 6 |
| 7 | Thumb + Index | Fret 7 |
| 8 | Thumb + Index + Middle | Fret 8 |

The capo shifts the frequencies of all three notes in a chord using a semitone-based frequency multiplier.

## 🛠️ Technologies Used

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV (`cv2`) | Webcam capture and image processing |
| MediaPipe | Real-time hand tracking and landmark detection |
| NumPy | Finger-distance calculations and audio signal generation |
| Pygame | Musical audio playback and graphical interface |
| MediaPipe Tasks | Hand Landmarker model and video-mode detection |

## 📦 Requirements

**Python libraries:**

The following external packages are required:

```bash
pip install numpy pygame opencv-python mediapipe
```

The following modules are included with Python:

- `sys`

The following MediaPipe modules are imported as part of the MediaPipe package:

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
```

## 📁 Repository Structure

Keep both Python scripts and the MediaPipe model file in the same project directory.

```text
Gesture-Musical-Instruments/
│
├── Piano_Gesture.py
├── Chord_Gesture.py
│
└── hand_landmarker.task
```

**Important:** The `hand_landmarker.task` file is required by both programs. It is the MediaPipe hand landmark model used for detecting and tracking hands.

Download the official model from:

https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

Place the downloaded model in the same directory as the Python scripts.

## 🚀 Installation & Execution

**Step 1: Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/Gesture-Musical-Instruments.git
```

**Step 2: Open the project directory**

```bash
cd Gesture-Musical-Instruments
```

**Step 3: Install dependencies**

```bash
pip install numpy pygame opencv-python mediapipe
```

**Step 4: Download the MediaPipe model**

Download `hand_landmarker.task` from the official MediaPipe documentation and place it in the project folder.

**Step 5: Run the desired instrument**

For Piano Gesture:

```bash
python Piano_Gesture.py
```

For Chord Gesture:

```bash
python Chord_Gesture.py
```

Make sure your webcam is connected and accessible. The programs use camera index `0` by default.

## ⌨️ Keyboard Controls

Both instruments also support keyboard input, allowing you to play without performing gestures.

**Piano Gesture:**

| Keyboard | Action |
|---|---|
| 1–8 | Play notes in the lower octave |
| Shift + 1–8 | Play notes in the higher octave |

**Chord Gesture:**

| Keyboard | Action |
|---|---|
| 1–8 | Play major chords |
| Shift + 1–8 | Play minor chords |

The chord gesture program retains the currently selected capo position when playing through the keyboard.

## ⚙️ How It Works

Both applications follow a similar processing pipeline:

1. **Webcam Input:** OpenCV captures live video from the webcam.
2. **Hand Detection:** MediaPipe detects and tracks hand landmarks.
3. **Gesture Recognition:** Finger extension is determined using distances between hand landmarks.
4. **Gesture Stabilization:** A three-frame stability threshold reduces accidental changes caused by noisy detections.
5. **Musical Mapping:** Recognized gestures are mapped to piano notes or guitar chords.
6. **Sound Generation:** NumPy generates audio waveforms, and Pygame plays the resulting sounds.
7. **Visualization:** Pygame displays the instrument interface, active notes/chords, and live camera feed.

## 🎧 Audio Generation

The musical sounds are generated programmatically rather than using prerecorded instrument samples.

**Piano:**
- Fundamental sine-wave frequency.
- Additional harmonic components.
- Exponential decay envelope.
- Stereo audio output at 44.1 kHz.

**Guitar:**
- Multiple frequencies combined to form chords.
- Harmonic components for each string.
- Staggered string activation to simulate strumming.
- Attack and decay envelopes for a more guitar-like sound.
- Capo-based pitch shifting.

## ⚠️ Notes & Troubleshooting

- Ensure that the webcam is not being used by another application.
- Make sure `hand_landmarker.task` is present in the project directory.
- Keep the model file and Python scripts in the same working directory.
- Ensure that your system has working audio output.
- Hold gestures steadily for a few frames to allow recognition.
- Both programs use the default webcam index `0`. Change `cv2.VideoCapture(0)` if a different camera is required.

## 💡 About the Project

This project explores the combination of computer vision, hand gesture recognition, and digital sound synthesis to create interactive musical experiences.

The goal is to make music more accessible and playful by replacing traditional instrument controls with natural hand movements.

Whether you want to play a piano with your fingers or strum guitar chords through gestures, these applications turn simple hand movements into music!

---

⭐ If you enjoy this project, consider giving the repository a star!

🎶 Happy Gesture Playing!
