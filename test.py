import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import sounddevice as sd
import librosa
import random
import asyncio
import bhaptics_python as bh # Χρήση του SDK 2

# ================================================================================================
# Audio Engine & Globals (Παραμένουν ίδια)
# ================================================================================================
audio, sr = librosa.load("music.mp3", sr=None, mono=True)
audio_lock = threading.Lock()
music_speed, music_playing, current_volume, music_pitch = 1.0, False, 50.0, 0
challenge_active, current_challenge, challenge_player, challenge_end_time = False, None, None, 0.0
scores = [0, 0]
prev_volume, prev_speed = 50.0, 1.0
original_audio = audio.copy()
audio_buffer = original_audio.copy()
audio_position = 0

# ================================================================================================
# Haptics Manager (SDK 2 Implementation)
# ================================================================================================
class HapticManager:
    def __init__(self):
        self.connected = False
        # Στο SDK 2 χρησιμοποιούμε το App ID και το API Key από το bHaptics Portal
        self.app_id = "YOUR_APP_ID" 
        self.api_key = "YOUR_API_KEY"
        
    def connect(self):
        """Αρχικοποίηση σύνδεσης SDK 2"""
        try:
            # Το SDK 2 απαιτεί εγγραφή της εφαρμογής
            bh.initialize(self.app_id, self.api_key)
            time.sleep(2)
            self.connected = True
            print("✅ bHaptics SDK 2 Connected")
        except Exception as e:
            print(f"❌ SDK 2 Connection Error: {e}")

    def play_dot(self, key, x, y, intensity):
        """Αντίστοιχο του submit_dot στο SDK 2"""
        if self.connected:
            # Στο SDK 2 η εντολή στέλνει dots σε συγκεκριμένα frame
            bh.submit_dot(key, "Front", [{"index": 10, "intensity": int(intensity)}], 100)

    def play_event(self, event_name):
        """Η προτεινόμενη μέθοδος του SDK 2 μέσω προ-σχεδιασμένων Events"""
        if self.connected:
            bh.submit_registered(event_name)

# 

# ================================================================================================
# Gesture & Audio Logic (Προσαρμοσμένο για SDK 2)
# ================================================================================================
def rebuild_audio():
    global audio_buffer, audio_position
    audio_out = librosa.effects.time_stretch(original_audio, rate=music_speed)
    if music_pitch != 0:
        audio_out = librosa.effects.pitch_shift(audio_out, sr=sr, n_steps=music_pitch)
    with audio_lock:
        audio_buffer = audio_out
        audio_position = 0

def audio_callback(outdata, frames, time_info, status):
    global audio_position
    if not music_playing:
        outdata.fill(0)
        return
    with audio_lock:
        indices = (np.arange(audio_position, audio_position + frames)) % len(audio_buffer)
        outdata[:, 0] = audio_buffer[indices] * (current_volume / 100.0)
        audio_position = (audio_position + frames) % len(audio_buffer)

# 

class GestureManager:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
        self.mp_draw = mp.solutions.drawing_utils

    def analyze(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            return list(zip(res.multi_hand_landmarks, res.multi_handedness))
        return []

# ================================================================================================
# Main loop
# ================================================================================================
def main():
    global music_speed, current_volume, music_playing, music_pitch, challenge_active
    
    cap = cv2.VideoCapture(0)
    haptics = HapticManager()
    haptics.connect() # Σύνδεση SDK 2
    gestures = GestureManager()
    
    stream = sd.OutputStream(samplerate=sr, channels=1, callback=audio_callback)
    stream.start()

    print("🎵 Maestro SDK 2 System Started")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        results = gestures.analyze(frame)

        if results:
            for hand_lms, handedness in results:
                label = handedness.classification[0].label
                # Ανίχνευση κινήσεων (Pinch, Fist κλπ.)
                # Εδώ καλούμε τα haptics.play_dot(...)
                
                # Παράδειγμα: Δόνηση στο Pinch
                thumb, index = hand_lms.landmark[4], hand_lms.landmark[8]
                dist = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
                if dist < 0.05:
                    haptics.play_dot("VestFront", 0.5, 0.5, 80)

        cv2.imshow("Maestro SDK 2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    stream.stop()

if __name__ == "__main__":
    main()