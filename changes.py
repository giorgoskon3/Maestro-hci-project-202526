#Fist: play-pause
#Pinch (thumb + index): volume (> distance --> higher) --> feedback to front vest
#Wrist Height: speed (< height --> lower) --> feedback to back vest
#Challenge at random time intervals:
#signal to front and back vest --> user opens hand --> music plays at higher pitch

import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import sounddevice as sd    # For audio playback
import librosa              # For changing speed
import random

#=================================================================================================

#Audio setup
original_audio, sr = librosa.load("music.mp3", sr=None, mono=True)
audio = original_audio.copy()
music_speed = 1.0
music_playing = False
current_volume = 50  # 0-100
challenge_active = False

#==================================================================================================

#Haptics setup
class HapticManager:
    def __init__(self):
        try:
            from bhaptics import haptic_player
            self.player = haptic_player.HapticPlayer()
            self.connected = True
            self.vest_name = "TactVest"
            print("✅ bHaptics Connected")
        except:
            self.connected = False
            print("⚠️ Running in Simulation Mode (No Device Found)")

    def play_volume_feedback(self, intensity):
        if self.connected:
            self.player.submit_dot("VestFront", intensity=intensity, device_name=self.vest_name)

    def play_speed_feedback(self, intensity):
        if self.connected:
            self.player.submit_dot("VestBack", intensity=intensity, device_name=self.vest_name)

    def challenge_signal(self, intensity=100, duration=0.5):
        if self.connected:
            self.player.play("VestFront", intensity, duration)
            self.player.play("VestBack", intensity, duration)

#=========================================================================================================

#Music loop
def music_loop():
    global music_speed, music_playing, audio, sr

    pos = 0
    chunk_size = 2048

    while True:
        if not music_playing:
            sd.stop()
            sd.sleep(50)
            continue

        chunk = audio[pos:pos + chunk_size]
        if len(chunk) < 32:
            pos = 0
            continue

        stretched = librosa.effects.time_stretch(chunk.astype(np.float32), music_speed)
        sd.play(stretched, sr, blocking=True)

        pos += int(chunk_size * music_speed)
        sd.sleep(5)

#========================================================================================================

def challenge_loop(haptics):
    global challenge_active
    while True:
        wait_time = random.uniform(5, 15)
        time.sleep(wait_time)
        challenge_active = True
        haptics.challenge_signal()
        print("⚡ Challenge triggered! Open all fingers!")

#==========================================================================================================


#Gesture manager (glove)
class GestureManager:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_fist_time = 0
        self.fist_cooldown = 1.0

    def get_hand_data(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else []

    def recognize_fingers(self, hand_lms):
        fingers = []
        fingers.append(1 if hand_lms.landmark[4].x < hand_lms.landmark[3].x else 0)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(1 if hand_lms.landmark[tip].y < hand_lms.landmark[pip].y else 0)
        return fingers

    def analyze_gesture(self, hand_lms):
        thumb_tip = hand_lms.landmark[4]
        index_tip = hand_lms.landmark[8]
        pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        fingers = self.recognize_fingers(hand_lms)
        return {
            'fingers': fingers,
            'fingers_count': sum(fingers),
            'pinch_dist': pinch_dist,
            'is_fist': sum(fingers) == 0
        }

    def on_fist_detected(self):
        global music_playing
        current_time = time.time()
        if current_time - self.last_fist_time > self.fist_cooldown:
            self.last_fist_time = current_time
            music_playing = not music_playing
            print("▶️ PLAY" if music_playing else "⏸ PAUSE")
            return True
        return False

#=======================================================================================================

#Main
def main():
    global music_speed, current_volume, audio, challenge_active

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    haptics = HapticManager()
    gestures = GestureManager()

    threading.Thread(target=music_loop, daemon=True).start()
    threading.Thread(target=challenge_loop, args=(haptics,), daemon=True).start()

    print("\n🚀 MAESTRO HAND TRACKER")
    print("👊 Fist = Play / Pause")
    print("🤏 Pinch → Volume (Front Vest)")
    print("🖐 Wrist Height → Speed (Back Vest)")
    print("⚡ Random Challenge → Open all fingers → Pitch up")
    print("Press 'q' to quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        hands = gestures.get_hand_data(frame)

        if hands:
            for hand_lms in hands:
                gestures.mp_draw.draw_landmarks(frame, hand_lms, gestures.mp_hands.HAND_CONNECTIONS)
                data = gestures.analyze_gesture(hand_lms)

                # 👊 Fist → Play/Pause
                if data['is_fist']:
                    gestures.on_fist_detected()

                # 🤏 Pinch → Volume
                if data['fingers'][0] == 1 and data['fingers'][1] == 1:
                    volume_target = np.interp(data['pinch_dist'], [0.02, 0.15], [0, 100])
                    current_volume = current_volume * 0.85 + volume_target * 0.15
                    current_volume = max(0, min(100, current_volume))
                    haptics.play_volume_feedback(int(current_volume))
                    print(f"🔊 Volume: {int(current_volume)}%")

                # 🖐 Wrist height → Speed (independent)
                wrist_y = hand_lms.landmark[0].y
                target_speed = np.interp(wrist_y, [0.8, 0.2], [0.6, 1.6])
                music_speed = music_speed * 0.9 + target_speed * 0.1
                music_speed = max(0.5, min(2.0, music_speed))
                intensity_speed = int(np.interp(music_speed, [0.5, 2.0], [0, 100]))
                haptics.play_speed_feedback(intensity_speed)
                print(f"🎵 Speed: {music_speed:.2f}x")

                # ⚡ Challenge: open all fingers to complete
                if challenge_active and sum(data['fingers']) == 5:
                    print("✅ Challenge completed! Pitch up!")
                    audio = librosa.effects.pitch_shift(original_audio, sr, n_steps=2)
                    challenge_active = False

                # UI
                cv2.putText(frame, f"Volume: {int(current_volume)}%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"Speed: {music_speed:.2f}x", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

        cv2.imshow("Maestro Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
