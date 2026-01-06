#Fist: play-pause
#Pinch (thumb + index): volume (> distance --> higher) --> feedback to front vest (left for low, right for high)
#Wrist Height: speed (< height --> lower) --> feedback to back vest (left for low, right for hight)
#Challenge at random time intervals:
#signal to front and back vest --> user points index up --> music plays at higher pitch

import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import sounddevice as sd        #For audio playback
import librosa                  #For changing audio
import random

#=================================================================================================
# Audio setup
audio, sr = librosa.load("music.mp3", sr=None, mono=True)
music_speed = 1.0
music_playing = False
current_volume = 50             #0-100 scale
music_pitch = 0                 #0 = normal pitch
challenge_active = False        #true when challenge event happens
prev_volume = current_volume
prev_speed = music_speed

#=================================================================================================
# Haptics setup
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

    # Front vest
    def front_left(self, intensity):
        if self.connected:
            self.player.submit_dot("VestFrontLeft", intensity=intensity, device_name=self.vest_name)

    def front_right(self, intensity):
        if self.connected:
            self.player.submit_dot("VestFrontRight", intensity=intensity, device_name=self.vest_name)

    # Back vest
    def back_left(self, intensity):
        if self.connected:
            self.player.submit_dot("VestBackLeft", intensity=intensity, device_name=self.vest_name)

    def back_right(self, intensity):
        if self.connected:
            self.player.submit_dot("VestBackRight", intensity=intensity, device_name=self.vest_name)

    # Challenge feedback
    def challenge_signal(self, intensity=100):
        if self.connected:
            self.player.submit_dot("VestFront", intensity=intensity, device_name=self.vest_name)
            self.player.submit_dot("VestBack", intensity=intensity, device_name=self.vest_name)

#=================================================================================================
# Music loop
def music_loop():
    global music_speed, music_playing, audio, sr, music_pitch

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
        pitched = librosa.effects.pitch_shift(stretched, sr, n_steps=music_pitch)
        sd.play(pitched, sr, blocking=True)

        pos += int(chunk_size * music_speed)
        sd.sleep(5)

#===================================================================================================
# Challenge loop
def challenge_loop(haptics):
    global challenge_active
    while True:
        time.sleep(random.uniform(5, 15))
        challenge_active = True
        print("⚡ Challenge started! Point your index finger up!")
        haptics.challenge_signal()

#=================================================================================================
# Gesture manager (glove)
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
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
            'pinch_dist': pinch_dist,
            'is_fist': sum(fingers) == 0
        }

    def on_fist_detected(self):
        global music_playing
        now = time.time()
        if now - self.last_fist_time > self.fist_cooldown:
            self.last_fist_time = now
            music_playing = not music_playing
            print("▶️ PLAY" if music_playing else "⏸ PAUSE")

#=================================================================================================
# Main
def main():
    global music_speed, current_volume, prev_volume, prev_speed, challenge_active

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    haptics = HapticManager()
    gestures = GestureManager()

    threading.Thread(target=music_loop, daemon=True).start()
    threading.Thread(target=challenge_loop, args=(haptics,), daemon=True).start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hands = gestures.get_hand_data(frame)

        if hands:
            for hand_lms in hands:
                gestures.mp_draw.draw_landmarks(frame, hand_lms, gestures.mp_hands.HAND_CONNECTIONS)
                data = gestures.analyze_gesture(hand_lms)

                #👊 Fist → Play / Pause
                if data['is_fist']:
                    gestures.on_fist_detected()

                #🤏 Pinch → Volume (thumb+index)
                if data['fingers'][0] and data['fingers'][1]:
                    target = np.interp(data['pinch_dist'], [0.02, 0.15], [0, 100])
                    current_volume = current_volume * 0.85 + target * 0.15
                    current_volume = np.clip(current_volume, 0, 100)

                    if current_volume < prev_volume:
                        haptics.front_left(int(current_volume))
                    else:
                        haptics.front_right(int(current_volume))

                    prev_volume = current_volume

                #🖐 Wrist height → Speed
                wrist_y = hand_lms.landmark[0].y
                target_speed = np.interp(wrist_y, [0.8, 0.2], [0.6, 1.6])
                music_speed = music_speed * 0.9 + target_speed * 0.1
                music_speed = np.clip(music_speed, 0.5, 2.0)

                intensity = int(np.interp(music_speed, [0.5, 2.0], [0, 100]))
                if music_speed < prev_speed:
                    haptics.back_left(intensity)
                else:
                    haptics.back_right(intensity)

                prev_speed = music_speed

                #⚡ Challenge → Index finger up
                if challenge_active and data['fingers'] == [0, 1, 0, 0, 0]:
                    music_pitch += 2
                    challenge_active = False

        cv2.imshow("Maestro Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

#=================================================================================================
if __name__ == "__main__":
    main()
