import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import sounddevice as sd
import librosa
import random

# ================================================================================================
# Audio setup
# ================================================================================================
audio, sr = librosa.load("music.mp3", sr=None, mono=True)
audio_lock = threading.Lock()
music_speed = 1.0
music_playing = False
current_volume = 50.0 
music_pitch = 0
challenge_active = False

# Challenge system state
current_challenge = None
challenge_player = None
challenge_end_time = 0.0

# Scoring
scores = [0, 0]
prev_volume = current_volume
prev_speed = music_speed
original_audio = audio.copy()
processed_audio = audio.copy()
audio_buffer = processed_audio.copy()
audio_position = 0  

# ================================================================================================
# Haptics setup - ΔΙΟΡΘΩΜΕΝΟ ΓΙΑ SDK1
# ================================================================================================
class HapticManager:
    def __init__(self):
        try:
            from bhaptics import haptic_player
            # Αρχικοποίηση WebSocket σύνδεσης
            self.player = haptic_player.HapticPlayer("Maestro_Full_System", "Maestro_Full_System")
            
            # Κρίσιμο delay για να προλάβει να συνδεθεί η Python με το Player App
            print("⏳ Connecting to bHaptics Player...")
            time.sleep(3) 
            
            self.connected = True
            # Ορισμός Standard Keys που αναγνωρίζει το γιλέκο και τα μανίκια
            self.VEST = "VestFront"
            self.VEST_BACK = "VestBack"
            self.ARM_R = "ArmR"
            self.ARM_L = "ArmL"
            print("✅ bHaptics Connected")
        except Exception as e:
            self.connected = False
            print(f"⚠️ Running in Simulation Mode: {e}")

    def front_left(self, intensity):
        if self.connected:
            self.player.submit_dot(self.VEST, position={"x": 0.2, "y": 0.5}, 
                                 intensity=int(intensity), duration_ms=100)

    def front_right(self, intensity):
        if self.connected:
            self.player.submit_dot(self.VEST, position={"x": 0.8, "y": 0.5}, 
                                 intensity=int(intensity), duration_ms=100)

    def back_left(self, intensity):
        if self.connected:
            self.player.submit_dot(self.VEST_BACK, position={"x": 0.2, "y": 0.5}, 
                                 intensity=int(intensity), duration_ms=100)

    def back_right(self, intensity):
        if self.connected:
            self.player.submit_dot(self.VEST_BACK, position={"x": 0.8, "y": 0.5}, 
                                 intensity=int(intensity), duration_ms=100)

    def challenge_signal(self, intensity=100):
        if self.connected:
            self.player.submit_dot(self.VEST, position={"x": 0.5, "y": 0.5}, 
                                 intensity=intensity, duration_ms=300)

    def glove_pulse(self, intensity=100):
        if self.connected:
            # Χρήση ArmL για προσομοίωση γαντιού/μανικιού
            self.player.submit_dot(self.ARM_L, position={"x": 0.5, "y": 0.5}, 
                                 intensity=intensity, duration_ms=200)

    def sleeve_pulse(self, intensity=100):
        if self.connected:
            self.player.submit_dot(self.ARM_R, position={"x": 0.5, "y": 0.5}, 
                                 intensity=intensity, duration_ms=200)

    def success_signal(self, player_id, intensity=100):
        if not self.connected: return
        if player_id == 0:
            self.glove_pulse(intensity)
            time.sleep(0.1)
            self.front_right(intensity)
        else:
            self.sleeve_pulse(intensity)

# ===================================================================================================
# Audio Engine
# ===================================================================================================
def rebuild_audio():
    global processed_audio, original_audio, music_speed, music_pitch, audio_buffer, audio_position
    
    # Apply time-stretch
    audio_out = librosa.effects.time_stretch(original_audio, rate=music_speed)
    
    # Apply pitch shift
    if music_pitch != 0:
        audio_out = librosa.effects.pitch_shift(audio_out, sr=sr, n_steps=music_pitch)
    
    with audio_lock:
        processed_audio = audio_out
        audio_buffer = processed_audio.copy()
        audio_position = 0  

def audio_callback(outdata, frames, time_info, status):
    global audio_buffer, current_volume, music_playing, audio_position
    if not music_playing:
        outdata[:] = np.zeros((frames, 1), dtype=np.float32)
        return
    
    with audio_lock:
        buffer_len = len(audio_buffer)
        if buffer_len == 0:
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)
            return
        
        end_pos = audio_position + frames
        if end_pos <= buffer_len:
            chunk = audio_buffer[audio_position:end_pos]
            audio_position = end_pos
        else:
            chunk = np.concatenate([audio_buffer[audio_position:], audio_buffer[:end_pos - buffer_len]])
            audio_position = end_pos - buffer_len
        
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')
        outdata[:, 0] = chunk * (current_volume / 100.0)

def start_audio_stream():
    stream = sd.OutputStream(samplerate=sr, channels=1, callback=audio_callback, blocksize=1024, dtype='float32')
    stream.start()
    return stream

# ===================================================================================================
# Game Logic & Turns
# ===================================================================================================
class TurnManager:
    def __init__(self, haptics):
        self.current_player = 0
        self.haptics = haptics
        self.turn_duration = 30  
        self.last_switch = time.time()
        self.signal_turn_start()

    def signal_turn_start(self):
        if self.current_player == 0:
            print("🎮 Player 0's Turn")
            self.haptics.glove_pulse(100)
        else:
            print("🎮 Player 1's Turn")
            self.haptics.sleeve_pulse(100)

    def update(self):
        if time.time() - self.last_switch > self.turn_duration:
            self.switch_turn()

    def switch_turn(self):
        self.current_player = 1 - self.current_player
        self.last_switch = time.time()
        self.signal_turn_start()

    def is_active(self, player_id):
        return self.current_player == player_id

def challenge_loop(haptics, turns):
    global challenge_active, current_challenge, challenge_player, challenge_end_time, scores
    challenge_types = ["PITCH", "SPEED", "VOLUME"]
    time.sleep(10)  
    
    while True:
        time.sleep(random.uniform(10, 20))
        challenge_active = True
        current_challenge = random.choice(challenge_types)
        challenge_player = turns.current_player
        challenge_end_time = time.time() + 5
        
        print(f"⚡ Challenge: {current_challenge} for Player {challenge_player}")
        haptics.challenge_signal()
        
        while time.time() < challenge_end_time and challenge_active:
            time.sleep(0.1)
        
        if challenge_active: # Timeout
            print(f"⏱️ Timeout! Player {challenge_player} -5 points")
            scores[challenge_player] = max(0, scores[challenge_player] - 5)
            challenge_active = False

# ===================================================================================================
# Gesture Recognition
# ===================================================================================================
class GestureManager:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
        self.mp_draw = mp.solutions.drawing_utils
        self.last_fist_time = 0

    def get_hand_data(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks and results.multi_handedness:
            return list(zip(results.multi_hand_landmarks, results.multi_handedness))
        return []

    def analyze_gesture(self, hand_lms, hand_label):
        # Recognize fingers
        fingers = []
        if hand_label == 'Right':
            fingers.append(1 if hand_lms.landmark[4].x < hand_lms.landmark[3].x else 0)
        else:
            fingers.append(1 if hand_lms.landmark[4].x > hand_lms.landmark[3].x else 0)
        
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(1 if hand_lms.landmark[tip].y < hand_lms.landmark[pip].y else 0)
        
        thumb_tip = hand_lms.landmark[4]
        index_tip = hand_lms.landmark[8]
        pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        
        return {'fingers': fingers, 'pinch_dist': pinch_dist, 'is_fist': sum(fingers) == 0}

# ===================================================================================================
# Main Application
# ===================================================================================================
def main():
    global music_speed, current_volume, prev_volume, prev_speed, music_pitch, music_playing
    global challenge_active, current_challenge, challenge_player, scores
    
    cap = cv2.VideoCapture(0)
    haptics = HapticManager()
    turns = TurnManager(haptics)
    gestures = GestureManager()
    audio_stream = start_audio_stream()
    
    threading.Thread(target=challenge_loop, args=(haptics, turns), daemon=True).start()
    
    while cap.isOpened():
        turns.update()
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        hands = gestures.get_hand_data(frame)
        
        if hands:
            for hand_lms, handedness in hands:
                hand_label = handedness.classification[0].label
                player_id = 0 if hand_label == 'Left' else 1
                if not turns.is_active(player_id): continue
                
                gestures.mp_draw.draw_landmarks(frame, hand_lms, gestures.mp_hands.HAND_CONNECTIONS)
                data = gestures.analyze_gesture(hand_lms, hand_label)
                
                # Fist -> Toggle Play/Pause
                if data['is_fist']:
                    now = time.time()
                    if now - gestures.last_fist_time > 0.5:
                        music_playing = not music_playing
                        gestures.last_fist_time = now
                        print("▶️ Play" if music_playing else "⏸️ Pause")

                # Pinch -> Volume
                if data['fingers'][0] and data['fingers'][1] and sum(data['fingers']) == 2:
                    target = np.clip(np.interp(data['pinch_dist'], [0.03, 0.12], [0, 100]), 0, 100)
                    current_volume = current_volume * 0.8 + target * 0.2
                    if abs(current_volume - prev_volume) > 2:
                        if player_id == 0: haptics.front_left(current_volume)
                        else: haptics.sleeve_pulse(current_volume)
                        prev_volume = current_volume

                # Wrist Height -> Speed
                wrist_y = hand_lms.landmark[0].y
                target_speed = np.interp(wrist_y, [0.8, 0.2], [0.6, 1.6])
                music_speed = music_speed * 0.85 + target_speed * 0.15
                if abs(music_speed - prev_speed) > 0.08:
                    rebuild_audio()
                    prev_speed = music_speed

                # Challenge logic
                if challenge_active and player_id == challenge_player:
                    f = data['fingers']
                    success = False
                    if current_challenge == "PITCH" and f == [0, 1, 0, 0, 0]:
                        music_pitch = np.clip(music_pitch + 2, -12, 12); success = True
                    elif current_challenge == "SPEED" and f == [0, 1, 1, 0, 0]:
                        music_speed = np.clip(music_speed + 0.25, 0.5, 2.0); success = True
                    elif current_challenge == "VOLUME" and f == [1, 0, 0, 0, 0]:
                        current_volume = np.clip(current_volume + 15, 0, 100); success = True
                    
                    if success:
                        scores[player_id] += 10; haptics.success_signal(player_id)
                        challenge_active = False; rebuild_audio()

        # UI Overlay
        cv2.putText(frame, f"P0 Score: {scores[0]} | P1 Score: {scores[1]}", (20, 40), 2, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Turn: Player {turns.current_player}", (20, 80), 2, 0.8, (0, 255, 255), 2)
        if challenge_active:
            cv2.putText(frame, f"DO: {current_challenge}!", (w//2-100, h//2), 2, 1.2, (0, 0, 255), 3)

        cv2.imshow("Maestro Full System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows(); audio_stream.stop()

if __name__ == "__main__":
    main()