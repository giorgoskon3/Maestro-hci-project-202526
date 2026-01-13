import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import sounddevice as sd
import librosa
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

#================================================================================================
# Audio setup
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

#================================================================================================
# Haptics setup with async support
class HapticManager:
    def __init__(self):
        self.connected = False
        self.player = None
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    async def initialize(self):
        """Initialize bHaptics connection asynchronously"""
        try:
            from bhaptics_python.haptic_player import HapticPlayer
            
            self.player = HapticPlayer()
            await asyncio.sleep(0.5)  # Give time to connect
            
            self.connected = True
            print("✅ bHaptics Connected Successfully")
            
            # Test connection
            devices = self.player.get_connected_devices()
            if devices:
                print(f"📱 Connected devices: {devices}")
            else:
                print("⚠️ No devices detected - make sure bHaptics Player is running")
                
        except ImportError:
            print("⚠️ bhaptics_python not installed")
            print("💡 Install with: pip install bhaptics-python")
        except Exception as e:
            print(f"⚠️ bHaptics error: {e}")
            self.connected = False
    
    def submit_haptic_sync(self, key, position, dot_points, duration=300):
        """Synchronous wrapper for haptic submission"""
        if not self.connected or not self.player:
            return
        try:
            self.player.submit_dot(
                key=key,
                position=position,
                dot_points=dot_points,
                duration_millis=duration
            )
        except Exception as e:
            print(f"Haptic error ({key}): {e}")

    def front_left(self, intensity):
        if self.connected:
            self.submit_haptic_sync(
                "FrontLeft",
                "VestFront",
                [{"index": 0, "intensity": int(intensity)}],
                200
            )

    def front_right(self, intensity):
        if self.connected:
            self.submit_haptic_sync(
                "FrontRight",
                "VestFront",
                [{"index": 4, "intensity": int(intensity)}],
                200
            )

    def back_left(self, intensity):
        if self.connected:
            self.submit_haptic_sync(
                "BackLeft",
                "VestBack",
                [{"index": 0, "intensity": int(intensity)}],
                200
            )

    def back_right(self, intensity):
        if self.connected:
            self.submit_haptic_sync(
                "BackRight",
                "VestBack",
                [{"index": 4, "intensity": int(intensity)}],
                200
            )

    def challenge_signal(self, intensity=100):
        if self.connected:
            # Pulse entire vest
            self.submit_haptic_sync(
                "ChallengeFront",
                "VestFront",
                [{"index": i, "intensity": intensity} for i in range(20)],
                500
            )
            time.sleep(0.05)
            self.submit_haptic_sync(
                "ChallengeBack",
                "VestBack",
                [{"index": i, "intensity": intensity} for i in range(20)],
                500
            )

    def glove_pulse(self, intensity=100):
        if self.connected:
            self.submit_haptic_sync(
                "GlovePulse",
                "GloveL",
                [{"index": i, "intensity": intensity} for i in range(6)],
                300
            )

    def sleeve_pulse(self, intensity=100):
        if self.connected:
            self.submit_haptic_sync(
                "SleevePulse",
                "ForearmR",
                [{"index": i, "intensity": intensity} for i in range(6)],
                300
            )

    def success_signal(self, player_id, intensity=100):
        if not self.connected:
            return
        if player_id == 0:
            self.glove_pulse(intensity)
            time.sleep(0.1)
            self.front_right(intensity)
        else:
            self.sleeve_pulse(intensity)

#===================================================================================================
# Change audio
def rebuild_audio():
    global processed_audio, original_audio, music_speed, music_pitch, audio_buffer, audio_position
    
    audio_out = librosa.effects.time_stretch(original_audio, rate=music_speed)
    
    if music_pitch != 0:
        audio_out = librosa.effects.pitch_shift(audio_out, sr=sr, n_steps=music_pitch)
    
    with audio_lock:
        processed_audio = audio_out
        audio_buffer = processed_audio.copy()
        audio_position = 0

#===================================================================================================
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
            chunk = np.concatenate([
                audio_buffer[audio_position:],
                audio_buffer[:end_pos - buffer_len]
            ])
            audio_position = end_pos - buffer_len
        
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')
        
        outdata[:, 0] = chunk * (current_volume / 100.0)

#===================================================================================================
def start_audio_stream():
    stream = sd.OutputStream(
        samplerate=sr,
        channels=1,
        callback=audio_callback,
        blocksize=1024,
        dtype='float32'
    )
    stream.start()
    return stream

#===================================================================================================
# Challenge loop
def challenge_loop(haptics, turns):
    global challenge_active, current_challenge, challenge_player, challenge_end_time, scores
    
    challenge_types = ["PITCH", "SPEED", "VOLUME"]
    
    time.sleep(10)
    
    while True:
        time.sleep(random.uniform(10, 20))
        
        challenge_active = True
        current_challenge = random.choice(challenge_types)
        challenge_player = turns.current_player
        duration = 5
        challenge_end_time = time.time() + duration
        
        print(f"⚡ Challenge started for Player {challenge_player} ({current_challenge})")
        
        if challenge_player == 0:
            haptics.challenge_signal()
        else:
            haptics.sleeve_pulse(100)
        
        if current_challenge == "PITCH":
            print("➡️ Do: INDEX UP (only index finger) to increase pitch!")
        elif current_challenge == "SPEED":
            print("➡️ Do: V SIGN (index+middle up) to increase speed!")
        else:
            print("➡️ Do: THUMBS UP (only thumb up) to boost volume!")
        
        while time.time() < challenge_end_time and challenge_active:
            time.sleep(0.05)
        
        if challenge_active:
            print(f"⏱️ Challenge timed out! Player {challenge_player} -5 points")
            scores[challenge_player] = max(0, scores[challenge_player] - 5)
            challenge_active = False
            current_challenge = None
            challenge_player = None

#=================================================================================================
# Player turns manager
class TurnManager:
    def __init__(self, haptics):
        self.current_player = 0
        self.haptics = haptics
        self.turn_duration = 30
        self.last_switch = time.time()
        self.signal_turn_start()

    def signal_turn_start(self):
        if self.current_player == 0:
            print("🎮 Player 0's Turn (Vest + Left Glove)")
            self.haptics.glove_pulse(100)
        else:
            print("🎮 Player 1's Turn (Sleeve + Right Glove)")
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

#=================================================================================================
# Gesture manager
class GestureManager:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_fist_time = 0
        self.fist_cooldown = 0.5

    def get_hand_data(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks and results.multi_handedness:
            return list(zip(results.multi_hand_landmarks, results.multi_handedness))
        return []

    def recognize_fingers(self, hand_lms, hand_label):
        fingers = []
        
        if hand_label == 'Right':
            fingers.append(1 if hand_lms.landmark[4].x < hand_lms.landmark[3].x else 0)
        else:
            fingers.append(1 if hand_lms.landmark[4].x > hand_lms.landmark[3].x else 0)
        
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(1 if hand_lms.landmark[tip].y < hand_lms.landmark[pip].y else 0)
        
        return fingers

    def analyze_gesture(self, hand_lms, hand_label):
        thumb_tip = hand_lms.landmark[4]
        index_tip = hand_lms.landmark[8]
        pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        
        fingers = self.recognize_fingers(hand_lms, hand_label)
        
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
            print("▶️ PLAY" if music_playing else "⏸️ PAUSE")

#=================================================================================================
# Main async function
async def async_main():
    global music_speed, current_volume, prev_volume, prev_speed, music_pitch
    global challenge_active, current_challenge, challenge_player, scores
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Initialize haptics asynchronously
    haptics = HapticManager()
    await haptics.initialize()
    
    turns = TurnManager(haptics)
    gestures = GestureManager()
    audio_stream = start_audio_stream()
    
    # Start challenge thread
    threading.Thread(target=challenge_loop, args=(haptics, turns), daemon=True).start()
    
    print("\n🎵 Maestro System Started!")
    print("Controls:")
    print("  👊 Fist = Play/Pause")
    print("  🤏 Pinch = Volume Control")
    print("  🖐️ Wrist Height = Speed Control")
    print("  Press 'q' to quit\n")
    
    while cap.isOpened():
        turns.update()
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        hands = gestures.get_hand_data(frame)
        
        if hands:
            for hand_lms, handedness in hands:
                hand_label = handedness.classification[0].label
                player_id = 0 if hand_label == 'Left' else 1
                
                if not turns.is_active(player_id):
                    continue
                
                gestures.mp_draw.draw_landmarks(
                    frame, hand_lms, gestures.mp_hands.HAND_CONNECTIONS
                )
                
                data = gestures.analyze_gesture(hand_lms, hand_label)
                
                # Fist → Play/Pause
                if data['is_fist']:
                    gestures.on_fist_detected()
                
                # Pinch → Volume
                if data['fingers'][0] and data['fingers'][1] and sum(data['fingers']) == 2:
                    target = np.clip(np.interp(data['pinch_dist'], [0.03, 0.12], [0, 100]), 0, 100)
                    current_volume = current_volume * 0.8 + target * 0.2
                    
                    if abs(current_volume - prev_volume) > 2:
                        intensity = int(current_volume)
                        if current_volume < prev_volume:
                            if player_id == 0:
                                haptics.front_left(intensity)
                            else:
                                haptics.sleeve_pulse(intensity)
                        else:
                            if player_id == 0:
                                haptics.front_right(intensity)
                            else:
                                haptics.sleeve_pulse(intensity)
                        prev_volume = current_volume
                
                # Wrist height → Speed
                wrist_y = hand_lms.landmark[0].y
                target_speed = np.interp(wrist_y, [0.8, 0.2], [0.6, 1.6])
                music_speed = music_speed * 0.85 + target_speed * 0.15
                music_speed = np.clip(music_speed, 0.5, 2.0)
                
                if abs(music_speed - prev_speed) > 0.08:
                    rebuild_audio()
                    intensity = int(np.interp(music_speed, [0.5, 2.0], [30, 100]))
                    
                    if music_speed < prev_speed:
                        if player_id == 0:
                            haptics.back_left(intensity)
                        else:
                            haptics.sleeve_pulse(intensity)
                    else:
                        if player_id == 0:
                            haptics.back_right(intensity)
                        else:
                            haptics.sleeve_pulse(intensity)
                    prev_speed = music_speed
                
                # Challenge handling
                if challenge_active and player_id == challenge_player:
                    fingers = data['fingers']
                    success = False
                    
                    if current_challenge == "PITCH" and fingers == [0, 1, 0, 0, 0]:
                        music_pitch += 2
                        music_pitch = np.clip(music_pitch, -12, 12)
                        rebuild_audio()
                        success = True
                        print(f"✅ PITCH Challenge completed! (+10 points)")
                    
                    elif current_challenge == "SPEED" and fingers == [0, 1, 1, 0, 0]:
                        music_speed = float(np.clip(music_speed + 0.25, 0.5, 2.0))
                        rebuild_audio()
                        success = True
                        print(f"✅ SPEED Challenge completed! (+10 points)")
                    
                    elif current_challenge == "VOLUME" and fingers == [1, 0, 0, 0, 0]:
                        current_volume = float(np.clip(current_volume + 15, 0, 100))
                        success = True
                        print(f"✅ VOLUME Challenge completed! (+10 points)")
                    
                    if success:
                        scores[player_id] += 10
                        haptics.success_signal(player_id, 100)
                        challenge_active = False
                        current_challenge = None
                        challenge_player = None
        
        # UI Overlay
        cv2.putText(frame, f"P0: {scores[0]}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(frame, f"P1: {scores[1]}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        turn_label = "P0 (Left)" if turns.current_player == 0 else "P1 (Right)"
        cv2.putText(frame, f"Turn: {turn_label}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
        
        status = "PLAYING" if music_playing else "PAUSED"
        status_color = (0, 255, 0) if music_playing else (0, 100, 255)
        cv2.putText(frame, status, (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 3)
        cv2.putText(frame, f"Vol: {int(current_volume)}%", (20, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 3)
        cv2.putText(frame, f"Speed: {music_speed:.2f}x", (20, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 3)
        
        if challenge_active and current_challenge and challenge_player is not None:
            remaining = max(0.0, challenge_end_time - time.time())
            if current_challenge == "PITCH":
                prompt = "CHALLENGE: INDEX UP"
            elif current_challenge == "SPEED":
                prompt = "CHALLENGE: V SIGN"
            else:
                prompt = "CHALLENGE: THUMBS UP"
            
            cv2.putText(frame, f"{prompt} ({remaining:.1f}s)", (20, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
        
        cv2.imshow("Maestro Gesture Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Small delay to allow async operations
        await asyncio.sleep(0.001)
    
    cap.release()
    cv2.destroyAllWindows()
    audio_stream.stop()
    audio_stream.close()

#=================================================================================================
def main():
    """Entry point that runs the async main function"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()