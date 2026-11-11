import sys
import numpy as np
import pygame
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------
# MediaPipe Hand Tracking Setup (2 Hands)
# ---------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def get_finger_states(landmarks):
    """
    Returns boolean states [thumb, index, middle, ring, pinky] for extended fingers.
    Uses distance ratios relative to wrist & palm for rotation and tilt invariant detection.
    """
    def get_dist(p1, p2):
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

    wrist = landmarks[0]

    # 4 Main Fingers: Extended if Tip distance to Wrist > PIP joint distance to Wrist * 1.05
    index = get_dist(landmarks[8], wrist) > (get_dist(landmarks[6], wrist) * 1.05)
    middle = get_dist(landmarks[12], wrist) > (get_dist(landmarks[10], wrist) * 1.05)
    ring = get_dist(landmarks[16], wrist) > (get_dist(landmarks[14], wrist) * 1.05)
    pinky = get_dist(landmarks[20], wrist) > (get_dist(landmarks[18], wrist) * 1.05)

    # Thumb: Extended if Tip distance to Pinky Knuckle (17) is > 1.22x Palm Width (Landmark 2 to 17)
    pinky_mcp = landmarks[17]
    thumb_mcp = landmarks[2]
    
    thumb_tip_dist = get_dist(landmarks[4], pinky_mcp)
    palm_width = get_dist(thumb_mcp, pinky_mcp)

    thumb = thumb_tip_dist > (palm_width * 1.22)

    return [thumb, index, middle, ring, pinky]

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

# ---------------------------------------------------------
# Pygame & Audio Initialization
# ---------------------------------------------------------
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
pygame.mixer.init()

WIDTH, HEIGHT = 1020, 580
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2-Octave C Major Piano (Robust Fist Detection)")
font = pygame.font.SysFont("Arial", 18, bold=True)
title_font = pygame.font.SysFont("Arial", 24, bold=True)

# Tone Generator
def create_note_sound(freq, duration=2.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    tone = (
        1.00 * np.sin(2 * np.pi * freq * t) +
        0.40 * np.sin(2 * np.pi * 2 * freq * t) +
        0.20 * np.sin(2 * np.pi * 3 * freq * t) +
        0.10 * np.sin(2 * np.pi * 4 * freq * t)
    )

    envelope = np.exp(-2.5 * t)
    tone *= envelope

    max_val = np.max(np.abs(tone))
    if max_val > 0:
        tone = tone / max_val * 0.8

    stereo = np.column_stack((tone, tone))
    audio_data = (stereo * 32767).astype(np.int16)
    
    return pygame.sndarray.make_sound(audio_data)

# Lower Octave Notes (C4 to C5)
LOW_NOTES = {
    pygame.K_1: {"name": "1: C4", "freq": 261.63, "label": "Do"},
    pygame.K_2: {"name": "2: D4", "freq": 293.66, "label": "Re"},
    pygame.K_3: {"name": "3: E4", "freq": 329.63, "label": "Mi"},
    pygame.K_4: {"name": "4: F4", "freq": 349.23, "label": "Fa"},
    pygame.K_5: {"name": "5: G4", "freq": 392.00, "label": "Sol"},
    pygame.K_6: {"name": "6: A4", "freq": 440.00, "label": "La"},
    pygame.K_7: {"name": "7: B4", "freq": 493.88, "label": "Ti"},
    pygame.K_8: {"name": "8: C5", "freq": 523.25, "label": "Do"}
}

# Higher Octave Notes (C5 to C6)
HIGH_NOTES = {
    pygame.K_1: {"name": "1: C5", "freq": 523.25, "label": "Do"},
    pygame.K_2: {"name": "2: D5", "freq": 587.33, "label": "Re"},
    pygame.K_3: {"name": "3: E5", "freq": 659.25, "label": "Mi"},
    pygame.K_4: {"name": "4: F5", "freq": 698.46, "label": "Fa"},
    pygame.K_5: {"name": "5: G5", "freq": 783.99, "label": "Sol"},
    pygame.K_6: {"name": "6: A5", "freq": 880.00, "label": "La"},
    pygame.K_7: {"name": "7: B5", "freq": 987.77, "label": "Ti"},
    pygame.K_8: {"name": "8: C6", "freq": 1046.50, "label": "Do"}
}

FINGER_KEY_MAP = {i: getattr(pygame, f"K_{i}") for i in range(1, 9)}

KEY_TO_NOTE_INDEX = {
    pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
    pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8,
}

low_sounds = {key: create_note_sound(data["freq"]) for key, data in LOW_NOTES.items()}
high_sounds = {key: create_note_sound(data["freq"]) for key, data in HIGH_NOTES.items()}

# GUI Cards Setup
cards = [
    {"key": key, "rect": pygame.Rect(20 + i * 122, 370, 110, 180)}
    for i, key in enumerate(LOW_NOTES.keys())
]

active_key = None
active_octave = "LOW"
frame_timestamp_ms = 0

# Debounce Variables
current_pressed_count = -1
current_octave_zone = "LOW"
candidate_count = -1
candidate_octave = "LOW"
stable_frames = 0
STABILITY_THRESHOLD = 3

running = True
clock = pygame.time.Clock()

while running:
    screen.fill((30, 32, 40))

    # 1. Capture Camera Frame & Process MediaPipe
    ret, frame = cap.read()
    detected_fingers_count = 0
    hands_detected = False
    detected_octave_zone = "LOW"

    if ret:
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += 33
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        all_y_coords = []

        if detection_result.hand_landmarks:
            hands_detected = True
            for hand in detection_result.hand_landmarks:
                landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                for lm in hand:
                    all_y_coords.append(lm.y)

                # Draw Hand Skeletons
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, landmarks_px[start_idx], landmarks_px[end_idx], (255, 0, 0), 2)
                for pt in landmarks_px:
                    cv2.circle(frame, pt, 4, (0, 255, 0), -1)

                finger_states = get_finger_states(hand)
                detected_fingers_count += sum(finger_states)

            avg_y = sum(all_y_coords) / len(all_y_coords)
            detected_octave_zone = "HIGH" if avg_y < 0.5 else "LOW"

        # 2. Draw Horizontal Screen Split Line on Camera Frame
        cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 255), 2)
        cv2.putText(frame, "UPPER: HIGH OCTAVE (C5-C6)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "LOWER: LOW OCTAVE (C4-C5)", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_resized = cv2.resize(cam_rgb, (320, 220))
        cam_surface = pygame.image.frombuffer(cam_resized.tobytes(), (320, 220), "RGB")

    # 3. Gesture & Zone Stabilization Logic
    if hands_detected:
        if detected_fingers_count == candidate_count and detected_octave_zone == candidate_octave:
            stable_frames += 1
        else:
            candidate_count = detected_fingers_count
            candidate_octave = detected_octave_zone
            stable_frames = 0

        if stable_frames >= STABILITY_THRESHOLD:
            if (current_pressed_count != candidate_count) or (current_octave_zone != candidate_octave):
                current_pressed_count = candidate_count
                current_octave_zone = candidate_octave
                active_octave = current_octave_zone

                if current_pressed_count in FINGER_KEY_MAP:
                    key_to_play = FINGER_KEY_MAP[current_pressed_count]
                    active_key = key_to_play
                    
                    if active_octave == "HIGH":
                        high_sounds[key_to_play].play()
                    else:
                        low_sounds[key_to_play].play()
                else:
                    active_key = None  # Fist (0 fingers) = Idle
    else:
        candidate_count = -1
        current_pressed_count = -1
        stable_frames = 0
        active_key = None

    # 4. Handle Pygame Keyboard & Quit Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key in KEY_TO_NOTE_INDEX:
                note_num = KEY_TO_NOTE_INDEX[event.key]
                key_code = FINGER_KEY_MAP[note_num]
                active_key = key_code

                is_shift = bool(event.mod & pygame.KMOD_SHIFT)
                
                if is_shift:
                    active_octave = "HIGH"
                    high_sounds[key_code].play()
                else:
                    active_octave = "LOW"
                    low_sounds[key_code].play()

        elif event.type == pygame.KEYUP:
            if event.key in KEY_TO_NOTE_INDEX:
                note_num = KEY_TO_NOTE_INDEX[event.key]
                key_code = FINGER_KEY_MAP[note_num]
                if key_code == active_key:
                    active_key = None

    # 5. Render Header & Instructions
    title_text = title_font.render("Dual-Octave C Major Scale Piano", True, (240, 240, 240))
    sub_text = font.render(
        f"Active Octave: {active_octave} | Gestures (1-8 fingers) | Fist = Idle",
        True, (100, 200, 255) if active_octave == "HIGH" else (255, 200, 100)
    )
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 15))
    screen.blit(sub_text, (WIDTH // 2 - sub_text.get_width() // 2, 50))

    # 6. Render Live Camera PIP
    if ret:
        cam_x = WIDTH // 2 - 160
        cam_y = 85
        screen.blit(cam_surface, (cam_x, cam_y))
        pygame.draw.rect(screen, (0, 255, 255), (cam_x, cam_y, 320, 220), width=2)

    # 7. Render Dynamic Note Cards
    active_notes = HIGH_NOTES if active_octave == "HIGH" else LOW_NOTES

    for card in cards:
        key_code = card["key"]
        rect = card["rect"]
        is_pressed = (active_key == key_code)

        color = (100, 200, 255) if is_pressed else (50, 55, 70)
        border_color = (255, 255, 255) if is_pressed else (80, 85, 100)

        pygame.draw.rect(screen, color, rect, border_radius=8)
        pygame.draw.rect(screen, border_color, rect, width=2, border_radius=8)

        note_info = active_notes[key_code]
        num_str = note_info["name"].split(":")[0]
        note_name = note_info["name"].split(":")[1].strip()

        lbl_num = font.render(f"[{num_str}]", True, (20, 20, 20) if is_pressed else (200, 200, 200))
        lbl_name = font.render(note_name, True, (10, 10, 10) if is_pressed else (255, 255, 255))
        lbl_notes = pygame.font.SysFont("Arial", 15, bold=True).render(
            f"{note_info['label']}",
            True,
            (40, 40, 40) if is_pressed else (150, 150, 160)
        )

        screen.blit(lbl_num, (rect.centerx - lbl_num.get_width() // 2, rect.y + 20))
        screen.blit(lbl_name, (rect.centerx - lbl_name.get_width() // 2, rect.y + 65))
        screen.blit(lbl_notes, (rect.centerx - lbl_notes.get_width() // 2, rect.y + 125))

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()
sys.exit()