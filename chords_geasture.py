import sys
import numpy as np
import pygame
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------
# MediaPipe Hand Tracking Setup (2 Hands Enabled)
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
    Uses strict distance ratios relative to wrist and palm width.
    """
    def get_dist(p1, p2):
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    palm_width = get_dist(landmarks[2], pinky_mcp)

    # 4 Main Fingers: Must extend straight away from wrist
    index = get_dist(landmarks[8], wrist) > (get_dist(landmarks[6], wrist) * 1.08)
    middle = get_dist(landmarks[12], wrist) > (get_dist(landmarks[10], wrist) * 1.08)
    ring = get_dist(landmarks[16], wrist) > (get_dist(landmarks[14], wrist) * 1.08)
    pinky = get_dist(landmarks[20], wrist) > (get_dist(landmarks[18], wrist) * 1.08)

    # Strict Thumb check
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    
    thumb_dist_pinky = get_dist(thumb_tip, pinky_mcp)
    thumb_ip_dist_pinky = get_dist(thumb_ip, pinky_mcp)
    thumb_dist_index = get_dist(thumb_tip, index_mcp)

    thumb = (thumb_dist_pinky > thumb_ip_dist_pinky * 1.15) and (thumb_dist_index > palm_width * 0.45)

    return (thumb, index, middle, ring, pinky)

# Exact gesture pattern mapping (Thumb, Index, Middle, Ring, Pinky) -> Gesture ID (0-8)
EXACT_GESTURES = {
    (False, False, False, False, False): 0, # 0: Fist (Resets Capo to 0)
    (False, True, False, False, False): 1,  # 1: Index finger
    (False, True, True, False, False): 2,   # 2: Index + Middle
    (False, True, True, True, False): 3,    # 3: Index + Middle + Ring
    (False, True, True, True, True): 4,     # 4: Index + Middle + Ring + Pinky
    (True, True, True, True, True): 5,      # 5: Thumb + Index + Middle + Ring + Pinky
    (True, False, False, False, False): 6,  # 6: Thumb only
    (True, True, False, False, False): 7,   # 7: Thumb + Index
    (True, True, True, False, False): 8,    # 8: Thumb + Index + Middle
}

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
pygame.mixer.set_num_channels(32)

WIDTH, HEIGHT = 1020, 580
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Guitar Gesture Control (Right Hand = Chords | Left Hand = Capo)")
font = pygame.font.SysFont("Arial", 18, bold=True)
title_font = pygame.font.SysFont("Arial", 24, bold=True)

# Acoustic Guitar Sound Generator
def create_guitar_chord_sound(freqs, duration=2.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    chord_wave = np.zeros_like(t)

    strum_delay = 0.025

    for idx, f in enumerate(freqs):
        start_sample = int(idx * strum_delay * sample_rate)
        if start_sample >= len(t):
            continue

        t_string = t[start_sample:] - t[start_sample]

        tone = (
            1.00 * np.sin(2 * np.pi * f * t_string) +
            0.50 * np.sin(2 * np.pi * 2 * f * t_string) +
            0.25 * np.sin(2 * np.pi * 3 * f * t_string) +
            0.12 * np.sin(2 * np.pi * 4 * f * t_string) +
            0.06 * np.sin(2 * np.pi * 5 * f * t_string)
        )

        attack = 1 - np.exp(-500 * t_string)
        decay = np.exp(-3.2 * t_string)
        envelope = attack * decay

        string_wave = tone * envelope
        chord_wave[start_sample:] += string_wave

    max_val = np.max(np.abs(chord_wave))
    if max_val > 0:
        chord_wave = (chord_wave / max_val) * 0.85

    stereo = np.column_stack((chord_wave, chord_wave))
    audio_data = (stereo * 32767).astype(np.int16)
    
    return pygame.sndarray.make_sound(audio_data)

# Lower Screen: MAJOR Base Frequencies
MAJOR_CHORDS = {
    pygame.K_1: {"name": "1: A Maj", "freqs": [220.00, 277.18, 329.63], "label": "A Major"},
    pygame.K_2: {"name": "2: B Maj", "freqs": [246.94, 311.13, 369.99], "label": "B Major"},
    pygame.K_3: {"name": "3: C Maj", "freqs": [261.63, 329.63, 392.00], "label": "C Major"},
    pygame.K_4: {"name": "4: D Maj", "freqs": [293.66, 369.99, 440.00], "label": "D Major"},
    pygame.K_5: {"name": "5: E Maj", "freqs": [329.63, 415.30, 493.88], "label": "E Major"},
    pygame.K_6: {"name": "6: F Maj", "freqs": [349.23, 440.00, 523.25], "label": "F Major"},
    pygame.K_7: {"name": "7: G Maj", "freqs": [392.00, 493.88, 587.33], "label": "G Major"},
    pygame.K_8: {"name": "8: A Maj(H)", "freqs": [440.00, 554.37, 659.25], "label": "A Maj (High)"}
}

# Upper Screen: MINOR Base Frequencies
MINOR_CHORDS = {
    pygame.K_1: {"name": "1: A Min", "freqs": [220.00, 261.63, 329.63], "label": "A Minor"},
    pygame.K_2: {"name": "2: B Min", "freqs": [246.94, 293.66, 369.99], "label": "B Minor"},
    pygame.K_3: {"name": "3: C Min", "freqs": [261.63, 311.13, 392.00], "label": "C Minor"},
    pygame.K_4: {"name": "4: D Min", "freqs": [293.66, 349.23, 440.00], "label": "D Minor"},
    pygame.K_5: {"name": "5: E Min", "freqs": [329.63, 392.00, 493.88], "label": "E Minor"},
    pygame.K_6: {"name": "6: F Min", "freqs": [349.23, 415.30, 523.25], "label": "F Minor"},
    pygame.K_7: {"name": "7: G Min", "freqs": [392.00, 466.16, 587.33], "label": "G Minor"},
    pygame.K_8: {"name": "8: A Min(H)", "freqs": [440.00, 523.25, 659.25], "label": "A Min (High)"}
}

FINGER_KEY_MAP = {i: getattr(pygame, f"K_{i}") for i in range(1, 9)}
KEY_TO_NOTE_INDEX = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
                     pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8}

sound_cache = {}

def get_chord_sound(zone, key_code, capo_fret):
    cache_key = (zone, key_code, capo_fret)
    if cache_key not in sound_cache:
        base_dict = MINOR_CHORDS if zone == "MINOR" else MAJOR_CHORDS
        base_freqs = base_dict[key_code]["freqs"]
        shifted_freqs = [f * (2 ** (capo_fret / 12.0)) for f in base_freqs]
        sound_cache[cache_key] = create_guitar_chord_sound(shifted_freqs)
    return sound_cache[cache_key]

cards = [
    {"key": key, "rect": pygame.Rect(20 + i * 122, 370, 110, 180)}
    for i, key in enumerate(MAJOR_CHORDS.keys())
]

active_key = None
active_zone = "MAJOR"
current_capo_fret = 0
frame_timestamp_ms = 0

right_current_gesture = None
right_current_zone = "MAJOR"
right_candidate_gesture = None
right_candidate_zone = "MAJOR"
right_stable_frames = 0

left_candidate_fret = 0
left_stable_frames = 0
STABILITY_THRESHOLD = 3

running = True
clock = pygame.time.Clock()

while running:
    screen.fill((30, 32, 40))

    ret, frame = cap.read()
    right_detected_gesture = None
    right_detected_zone = "MAJOR"
    left_detected_fret = None

    if ret:
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += 33
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        if detection_result.hand_landmarks and detection_result.handedness:
            for idx, hand in enumerate(detection_result.hand_landmarks):
                handedness_label = detection_result.handedness[idx][0].category_name
                is_right_hand = (handedness_label == "Left")

                landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                skeleton_color = (255, 100, 0) if is_right_hand else (0, 255, 100)
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, landmarks_px[start_idx], landmarks_px[end_idx], skeleton_color, 2)
                for pt in landmarks_px:
                    cv2.circle(frame, pt, 4, (0, 255, 0), -1)

                finger_states = get_finger_states(hand)
                gesture_id = EXACT_GESTURES.get(finger_states, None)

                if is_right_hand:
                    avg_y = sum(lm.y for lm in hand) / len(hand)
                    right_detected_zone = "MINOR" if avg_y < 0.5 else "MAJOR"
                    right_detected_gesture = gesture_id
                    
                    cv2.putText(frame, f"RIGHT: G{gesture_id if gesture_id else 'Fist'}", 
                                (landmarks_px[0][0] - 40, landmarks_px[0][1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                else:
                    # LEFT HAND: Accepts 0 (Fist) through 8 (Capo Frets)
                    if gesture_id in range(0, 9):
                        left_detected_fret = gesture_id

                    cv2.putText(frame, f"LEFT: Capo {gesture_id if gesture_id is not None else 'Keep'}", 
                                (landmarks_px[0][0] - 40, landmarks_px[0][1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 255), 2)
        cv2.putText(frame, "UPPER: MINOR CHORDS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, "LOWER: MAJOR CHORDS", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_resized = cv2.resize(cam_rgb, (320, 220))
        cam_surface = pygame.image.frombuffer(cam_resized.tobytes(), (320, 220), "RGB")

    # Left Hand Capo Logic (Fist explicitly resets Capo to 0)
    if left_detected_fret is not None:
        if left_detected_fret == left_candidate_fret:
            left_stable_frames += 1
        else:
            left_candidate_fret = left_detected_fret
            left_stable_frames = 0

        if left_stable_frames >= STABILITY_THRESHOLD:
            if current_capo_fret != left_candidate_fret:
                current_capo_fret = left_candidate_fret
    else:
        left_candidate_fret = None
        left_stable_frames = 0

    # Right Hand Gesture Logic
    if right_detected_gesture is not None:
        if right_detected_gesture == right_candidate_gesture and right_detected_zone == right_candidate_zone:
            right_stable_frames += 1
        else:
            right_candidate_gesture = right_detected_gesture
            right_candidate_zone = right_detected_zone
            right_stable_frames = 0

        if right_stable_frames >= STABILITY_THRESHOLD:
            if (right_current_gesture != right_candidate_gesture) or (right_current_zone != right_candidate_zone):
                right_current_gesture = right_candidate_gesture
                right_current_zone = right_candidate_zone
                active_zone = right_current_zone

                if right_current_gesture in FINGER_KEY_MAP:
                    key_to_play = FINGER_KEY_MAP[right_current_gesture]
                    active_key = key_to_play
                    get_chord_sound(active_zone, key_to_play, current_capo_fret).play()
                else:
                    active_key = None
    else:
        right_candidate_gesture = None
        right_current_gesture = None
        right_stable_frames = 0
        active_key = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key in KEY_TO_NOTE_INDEX:
                note_num = KEY_TO_NOTE_INDEX[event.key]
                key_code = FINGER_KEY_MAP[note_num]
                active_key = key_code

                is_shift = bool(event.mod & pygame.KMOD_SHIFT)
                active_zone = "MINOR" if is_shift else "MAJOR"
                
                get_chord_sound(active_zone, key_code, current_capo_fret).play()

        elif event.type == pygame.KEYUP:
            if event.key in KEY_TO_NOTE_INDEX:
                note_num = KEY_TO_NOTE_INDEX[event.key]
                key_code = FINGER_KEY_MAP[note_num]
                if key_code == active_key:
                    active_key = None

    title_text = title_font.render("Dual-Hand Gesture Guitar", True, (240, 240, 240))
    capo_str = f"CAPO: FRET {current_capo_fret}" if current_capo_fret > 0 else "CAPO: OPEN (FRET 0)"
    sub_text = font.render(
        f"Right Hand: {active_zone} Chords 1-8 | Left Hand: {capo_str} (Fist = Reset Capo 0)",
        True, (255, 180, 100) if active_zone == "MINOR" else (100, 200, 255)
    )
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 12))
    screen.blit(sub_text, (WIDTH // 2 - sub_text.get_width() // 2, 48))

    if ret:
        cam_x = WIDTH // 2 - 160
        cam_y = 85
        screen.blit(cam_surface, (cam_x, cam_y))
        pygame.draw.rect(screen, (0, 255, 255), (cam_x, cam_y, 320, 220), width=2)

    active_chords = MINOR_CHORDS if active_zone == "MINOR" else MAJOR_CHORDS

    for card in cards:
        key_code = card["key"]
        rect = card["rect"]
        is_pressed = (active_key == key_code)

        color = (255, 180, 100) if (is_pressed and active_zone == "MINOR") else (
            (100, 200, 255) if is_pressed else (50, 55, 70)
        )
        border_color = (255, 255, 255) if is_pressed else (80, 85, 100)

        pygame.draw.rect(screen, color, rect, border_radius=8)
        pygame.draw.rect(screen, border_color, rect, width=2, border_radius=8)

        chord_info = active_chords[key_code]
        num_str = chord_info["name"].split(":")[0]
        chord_name = chord_info["name"].split(":")[1].strip()

        lbl_num = font.render(f"[{num_str}]", True, (20, 20, 20) if is_pressed else (200, 200, 200))
        lbl_name = font.render(chord_name, True, (10, 10, 10) if is_pressed else (255, 255, 255))
        
        capo_label = f"Capo +{current_capo_fret}" if current_capo_fret > 0 else "Standard"
        lbl_notes = pygame.font.SysFont("Arial", 13, bold=True).render(
            capo_label,
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