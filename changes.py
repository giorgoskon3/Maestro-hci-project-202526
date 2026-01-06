import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import sounddevice as sd    #for music playing
import librosa              #for music speed

#===================================================================

#Audio setup
audio, sr = librosa.load("music.mp3", sr=None, mono=True)
music_speed = 1.0
music_playing = False

#=====================================================================

#Haptics setup
class HapticManager:
    def __init__(self):
        try:
            from bhaptics import haptic_player
            self.player = haptic_player.HapticPlayer()
            self.connected = True
            print("✅ bHaptics Connected")
        except:
            self.connected = False
            print("⚠️ Running in Simulation Mode (No Device Found)")
    
    def play_pinch(self, intensity):
        """Βασική δόνηση ανάλογα με την ένταση (0-100)"""
        if self.connected:
            pass
#========================================================================

def music_loop():
    global music_speed, music_playing

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

#=================================================================================        

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
        """Επεξεργάζεται το frame και επιστρέφει τα landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else []

    def recognize_fingers(self, hand_lms):
        """Επιστρέφει λίστα με το ποια δάχτυλα είναι ανοιχτά"""
        fingers = []
        
        # Αντίχειρας
        if hand_lms.landmark[4].x < hand_lms.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Υπόλοιπα δάχτυλα
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        for tip, pip in zip(tips, pips):
            if hand_lms.landmark[tip].y < hand_lms.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

    def analyze_gesture(self, hand_lms):
        """Υπολογίζει pinch και επιστρέφει δεδομένα"""
        """pinch: αντίχειρας + δείκτης"""
        thumb_tip = hand_lms.landmark[4]
        index_tip = hand_lms.landmark[8]
        
        # Απόσταση για Pinch
        pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        pinch_intensity = int(np.interp(pinch_dist, [0.02, 0.15], [100, 0]))
        
        fingers = self.recognize_fingers(hand_lms)
        
        return {
            'fingers': fingers,
            'fingers_count': sum(fingers),
            'pinch_intensity': max(0, min(100, pinch_intensity)),
            'is_fist': sum(fingers) == 0
        }

    def on_fist_detected(self):
        """Συνάρτηση που καλείται όταν ανιχνευθεί γροθιά"""
        """play-pause"""

        global music_playing

        current_time = time.time()
        if current_time - self.last_fist_time > self.fist_cooldown:
            self.last_fist_time = current_time
            music_playing = not music_playing
            print("▶️ PLAY" if music_playing else "⏸ PAUSE")
            return True
        return False

#======================================================================================


def main():
    global music_speed

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    haptics = HapticManager()
    gestures = GestureManager()

    threading.Thread(target=music_loop, daemon=True).start()

    print("\n🚀 MAESTRO HAND TRACKER")
    print("👊 Fist = Play / Pause")
    print("🤏 Pinch + Hand Height = Speed")
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
                gestures.mp_draw.draw_landmarks(
                    frame, hand_lms, gestures.mp_hands.HAND_CONNECTIONS
                )

                data = gestures.analyze_gesture(hand_lms)

                # 👊 Fist → Play / Pause
                if data["is_fist"]:
                    gestures.on_fist_detected()

                # 🤏 Pinch + Y → Speed
                if data["fingers"][0] == 1 and data["fingers"][1] == 1:
                    wrist_y = hand_lms.landmark[0].y
                    target = np.interp(wrist_y, [0.8, 0.2], [0.6, 1.6])
                    music_speed = music_speed * 0.9 + target * 0.1
                    music_speed = max(0.5, min(2.0, music_speed))

                    intensity = int(np.interp(music_speed, [0.5, 2.0], [0, 100]))
                    haptics.play_pinch(intensity)
                    print(f"🎵 Speed: {music_speed:.2f}x")

                # UI
                cv2.putText(frame, f"Speed: {music_speed:.2f}x", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 3)

        cv2.imshow("Maestro Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
