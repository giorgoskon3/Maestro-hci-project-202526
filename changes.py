import cv2
import mediapipe as mp
import time
import numpy as np

import pygame

pygame.mixer.init()
pygame.mixer.music.load("music.mp3")

is_playing = False      #is music playing at the moment
has_started = False     #did music ever start playing

current_volume = 0.5
pygame.mixer.music.set_volume(current_volume)


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

        global is_playing, has_started

        current_time = time.time()
        if current_time - self.last_fist_time > self.fist_cooldown:
            self.last_fist_time = current_time

            #fist to play-pause
            print("👊 FIST DETECTED! Action triggered!")
            if not has_started:
                pygame.mixer.music.play()
                has_started = True
                is_playing = True
                print("▶️ START")
            elif is_playing:
                pygame.mixer.music.pause()
                is_playing = False
                print("⏸ PAUSE")
            else:
                pygame.mixer.music.unpause()
                is_playing = True
                print("⏯ RESUME")
            return True
        return False


def main():
    global current_volume

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    haptics = HapticManager()
    gestures = GestureManager()
    
    print("\n🚀 MAESTRO HAND TRACKER")
    print("👊 Fist = Trigger Action")
    print("🤏 Pinch (Thumb+Index) = Volume Control")
    print("Press 'q' to quit\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        hands_landmarks = gestures.get_hand_data(frame)
        
        if hands_landmarks:
            for hand_lms in hands_landmarks:
                gestures.mp_draw.draw_landmarks(
                    frame, hand_lms, gestures.mp_hands.HAND_CONNECTIONS
                )
                
                data = gestures.analyze_gesture(hand_lms)
                
                # Fist Detection
                if data['is_fist']:
                    if gestures.on_fist_detected():
                        cv2.putText(frame, "FIST!", (w//2 - 100, h//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                
                # Volume Control με Pinch
                
                #if data['fingers'][0] == 1 and data['fingers'][1] == 1:
                 #   haptics.play_pinch(data['pinch_intensity'])
                  #  print(f"Volume: {data['pinch_intensity']}%")

                if data['fingers'][0] == 1 and data['fingers'][1] == 1:
                    target_volume = data['pinch_intensity'] / 100.0

                    #Smooth volume change
                    current_volume = current_volume * 0.85 + target_volume * 0.15

                    #Clamp volume to [0.0, 1.0]
                    current_volume = max(0.0, min(1.0, current_volume))

                    pygame.mixer.music.set_volume(current_volume)
                    haptics.play_pinch(data['pinch_intensity'])
                    print(f"🔊 Volume: {int(current_volume * 100)}%")
                
                # UI
                cv2.putText(frame, f"Fingers: {data['fingers_count']}", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Volume Bar
                cv2.rectangle(frame, (50, 150), (100, 450), (255, 255, 255), 2)
                fill_level = int(data['pinch_intensity'] * 3)
                cv2.rectangle(frame, (50, 450 - fill_level), (100, 450), (0, 255, 255), -1)
                cv2.putText(frame, f"{data['pinch_intensity']}%", (50, 480), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Maestro Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
