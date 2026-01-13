#===Game Description===#

#Player 0: vest + glove(left hand)
#Player 1: sleeve + glove(right hand)

#===Users take turns===#
#One player is active at a time
#When signal is sent to glove, it's their turn

#===For both players===##
#Fist: play-pause

#Pinch (thumb + index): volume (> distance --> higher)
#for player 0 --> feedback to front vest (left for low, right for high)
#for player 1 --> feedback to sleeve

#Wrist Height: speed (< height --> lower)
#for player 0 --> feedback to back vest (left for low, right for high)
#for player 1 --> feedback to sleeve

#Challenge 1:
#at random time intervals
#signal to front + back vest (player 0) or to sleeve (player 1) --> user points index up --> music plays at higher pitch

#================================================================================================

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
audio_lock = threading.Lock()           # Lock for thread safety
music_speed = 1.0
music_playing = False
current_volume = 50                     #0-100 scale
music_pitch = 0                         #0 = normal pitch
challenge_active = False                #true when challenge event happens
# --- Challenge system state ---
current_challenge = None          # "PITCH" | "SPEED" | "VOLUME"
challenge_player = None           # 0 or 1 (who must respond)
challenge_end_time = 0.0          # unix time when the window closes
challenge_start_time = 0.0        # unix time when the challenge was triggered

# --- Scoring ---
scores = [0, 0]                   # scores[0]=Player0, scores[1]=Player1

# --- Reaction time metrics (research-friendly) ---
rt_last = [None, None]            # last reaction time per player (seconds)
rt_sum = [0.0, 0.0]               # sum of reaction times per player
rt_count = [0, 0]                 # number of successful challenges per player
prev_volume = current_volume
prev_speed = music_speed
original_audio = audio.copy()
processed_audio = audio.copy()
audio_buffer = processed_audio.copy()   # Buffer used by audio callback

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

    def glove_pulse(self, intensity=100):
        if self.connected:
            self.player.submit_dot("Glove", intensity=intensity, device_name=self.vest_name)

    def sleeve_pulse(self, intensity=100):
        if self.connected:
            self.player.submit_dot("Arm", intensity=intensity, device_name=self.vest_name)


    # Success feedback (quick confirmation pulse)
    def success_signal(self, player_id, intensity=100):
        if not self.connected:
            return
        # Keep it simple & noticeable: pulse the device that belongs to the active player
        if player_id == 0:
            # Vest+glove player
            self.glove_pulse(intensity)
            self.front_right(intensity)
            self.back_right(intensity)
        else:
            # Sleeve player
            self.sleeve_pulse(intensity)

#===================================================================================================
#Change audio
def rebuild_audio():
    global processed_audio, original_audio, music_speed, music_pitch, challenge_active, audio_buffer

    # Apply time-stretch
    audio_out = librosa.effects.time_stretch(original_audio.astype(np.float32), music_speed)

    # Apply pitch shift when pitch has been modified by challenges
    if music_pitch != 0:
        audio_out = librosa.effects.pitch_shift(audio_out, sr=sr, n_steps=music_pitch)

    with audio_lock:
        processed_audio = audio_out
        audio_buffer = processed_audio.copy()  # update the buffer safely

#===================================================================================================
def audio_callback(outdata, frames, time_info, status):
    global audio_buffer, current_volume, music_playing

    # Safety: fill with silence if not playing
    if not music_playing:
        outdata[:] = np.zeros((frames, 1), dtype=np.float32)
        return

    with audio_lock:
        # Get a chunk of audio
        chunk = audio_buffer[:frames]
        if len(chunk) < frames:
            # Loop the music
            chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')
        # Apply volume
        outdata[:, 0] = chunk * (current_volume / 100.0)
        # Roll the buffer for looping
        audio_buffer = np.roll(audio_buffer, -frames)

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
    """Runs in a background thread and periodically triggers one random challenge.

    Challenges:
      - PITCH:    raise index finger only  -> increases pitch
      - SPEED:    raise index+middle (V)   -> increases speed slightly
      - VOLUME:   thumbs up only          -> boosts volume slightly
    """
    global challenge_active, current_challenge, challenge_player, challenge_end_time, challenge_start_time, scores

    challenge_types = ["PITCH", "SPEED", "VOLUME"]

    while True:
        time.sleep(random.uniform(5, 15))

        # Start a new challenge for the CURRENT active player
        challenge_active = True
        current_challenge = random.choice(challenge_types)
        challenge_player = turns.current_player
        duration = 5
        challenge_start_time = time.time()
        challenge_end_time = challenge_start_time + duration

        # Haptic alert (same intensity, different device based on player)
        print(f"⚡ Challenge started for Player {challenge_player} ({current_challenge})")
        if challenge_player == 0:
            haptics.challenge_signal()
        else:
            haptics.sleeve_pulse(100)

        # Console prompt (useful even if you don't show UI text)
        if current_challenge == "PITCH":
            print("➡️ Do: INDEX UP (only index finger) to increase pitch!")
        elif current_challenge == "SPEED":
            print("➡️ Do: V SIGN (index+middle up) to increase speed!")
        else:  # VOLUME
            print("➡️ Do: THUMBS UP (only thumb up) to boost volume!")

        # Wait until success or timeout
        while time.time() < challenge_end_time and challenge_active:
            time.sleep(0.05)

        # Timeout -> penalty
        if challenge_active:
            print("Challenge timed out! (-5 points)")
            scores[challenge_player] -= 5
            scores[challenge_player] = max(0, scores[challenge_player])
            challenge_active = False
            current_challenge = None
            challenge_player = None
#=================================================================================================
#Player turns manager
class TurnManager:
    def __init__(self, haptics):
        self.current_player = 0  # 0 = vest player, 1 = sleeve player
        self.haptics = haptics
        self.turn_duration = 20  # seconds
        self.last_switch = time.time()

        self.signal_turn_start()

    def signal_turn_start(self):
        if self.current_player == 0:
            print("🎮 Vest Player's Turn")
            self.haptics.glove_pulse(100)
        else:
            print("🎮 Sleeve Player's Turn")
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
# Gesture manager (gloves)
class GestureManager:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=2
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_fist_time = 0
        self.fist_cooldown = 1.0

    def get_hand_data(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
        # Return both landmarks and handedness
            return list(zip(results.multi_hand_landmarks, results.multi_handedness))
        else:
            return []

    def recognize_fingers(self, hand_lms, hand_label):
        fingers = []

        # Thumb
        if hand_label == 'Right':
            fingers.append(1 if hand_lms.landmark[4].x < hand_lms.landmark[3].x else 0)
        else:  # Left hand
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
            print("▶️ PLAY" if music_playing else "⏸ PAUSE")

#=================================================================================================
# Main
def main():
    global music_speed, current_volume, prev_volume, prev_speed, music_pitch
    global challenge_active, current_challenge, challenge_player, challenge_end_time, challenge_start_time
    global scores, rt_last, rt_sum, rt_count

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    haptics = HapticManager()
    turns = TurnManager(haptics)
    gestures = GestureManager()

    audio_stream = start_audio_stream()  # Non-blocking playback
    threading.Thread(target=challenge_loop, args=(haptics, turns), daemon=True).start()

    while cap.isOpened():
        turns.update()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hands = gestures.get_hand_data(frame)
        if hands:
            for i, (hand_lms, handedness) in enumerate(hands):
                hand_label = handedness.classification[0].label
                # Map left hand → player 0, right hand → player 1
                player_id = 0 if hand_label == 'Left' else 1
                if not turns.is_active(player_id):
                    continue  # Ignore gestures if not their turn
                gestures.mp_draw.draw_landmarks(frame, hand_lms, gestures.mp_hands.HAND_CONNECTIONS)
                data = gestures.analyze_gesture(hand_lms, hand_label)

                #👊 Fist → Play / Pause
                if data['is_fist']:
                    gestures.on_fist_detected()

                #🤏 Pinch → Volume (thumb+index)
                if data['fingers'][0] and data['fingers'][1]:
                    target = np.interp(data['pinch_dist'], [0.03, 0.12], [0, 100])
                    current_volume = current_volume * 0.9 + target * 0.1
                    target = np.clip(np.interp(data['pinch_dist'], [0.03, 0.12], [0, 100]), 0, 100)

                    if current_volume < prev_volume:
                        if player_id == 0:
                            haptics.front_left(int(current_volume))
                        elif player_id ==1:
                            haptics.sleeve_pulse(int(current_volume))
                    else:
                        if player_id ==0:
                            haptics.front_right(int(current_volume))
                        elif player_id ==1:
                            haptics.sleeve_pulse(int(current_volume))

                    prev_volume = current_volume

                #🖐 Wrist height → Speed
                wrist_y = hand_lms.landmark[0].y
                target_speed = np.interp(wrist_y, [0.8, 0.2], [0.6, 1.6])

                music_speed = music_speed * 0.9 + target_speed * 0.1
                music_speed = np.clip(music_speed, 0.5, 2.0)
                if abs(music_speed - prev_speed) > 0.05:
                    rebuild_audio()
                    prev_speed = music_speed

                intensity = int(np.interp(target_speed, [0.5, 2.0], [0, 100]))
                if target_speed < prev_speed:
                    if player_id ==0:
                        haptics.back_left(intensity)
                    elif player_id ==1:
                        haptics.sleeve_pulse(intensity)
                else:
                    if player_id ==0:
                        haptics.back_right(intensity)
                    elif player_id ==1:
                        haptics.sleeve_pulse(intensity)

                #⚡ Challenge handling (one of three types)
                if challenge_active and player_id == challenge_player:
                    # Finger patterns: [thumb, index, middle, ring, pinky]
                    fingers = data['fingers']

                    success = False
                    if current_challenge == "PITCH" and fingers == [0, 1, 0, 0, 0]:
                        # Index up only
                        music_pitch += 2
                        music_pitch = np.clip(music_pitch, -12, 12)
                        rebuild_audio()
                        success = True

                    elif current_challenge == "SPEED" and fingers == [0, 1, 1, 0, 0]:
                        # V sign (index + middle)
                        music_speed = float(np.clip(music_speed + 0.2, 0.5, 2.0))
                        rebuild_audio()
                        success = True

                    elif current_challenge == "VOLUME" and fingers == [1, 0, 0, 0, 0]:
                        # Thumbs up only
                        current_volume = float(np.clip(current_volume + 10, 0, 100))
                        success = True

                    if success:
                        # Reaction time (time from challenge trigger to correct gesture)
                        rt = max(0.0, time.time() - challenge_start_time)
                        rt_last[player_id] = rt
                        rt_sum[player_id] += rt
                        rt_count[player_id] += 1

                        scores[player_id] += 10
                        scores[player_id] = max(0, scores[player_id])
                        haptics.success_signal(player_id, 100)
                        challenge_active = False
                        current_challenge = None
                        challenge_player = None

        # =========================
        # UI Overlay (scores + turn + challenge)
        # =========================
        # Scores
        cv2.putText(frame, f"P0 Score: {scores[0]}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"P1 Score: {scores[1]}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Reaction times (last + avg)
        p0_last = "-" if rt_last[0] is None else f"{rt_last[0]:.2f}s"
        p1_last = "-" if rt_last[1] is None else f"{rt_last[1]:.2f}s"
        p0_avg = "-" if rt_count[0] == 0 else f"{(rt_sum[0]/rt_count[0]):.2f}s"
        p1_avg = "-" if rt_count[1] == 0 else f"{(rt_sum[1]/rt_count[1]):.2f}s"
        cv2.putText(frame, f"RT Last  P0:{p0_last}  P1:{p1_last}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"RT Avg   P0:{p0_avg}  P1:{p1_avg}", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Current turn
        turn_label = "P0 (Vest/Left)" if turns.current_player == 0 else "P1 (Sleeve/Right)"
        cv2.putText(frame, f"Turn: {turn_label}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Challenge prompt
        if challenge_active and current_challenge is not None and challenge_player is not None:
            remaining = max(0.0, challenge_end_time - time.time())
            if current_challenge == "PITCH":
                prompt = "CHALLENGE: INDEX UP -> +Pitch"
            elif current_challenge == "SPEED":
                prompt = "CHALLENGE: V SIGN -> +Speed"
            else:
                prompt = "CHALLENGE: THUMBS UP -> +Volume"

            cv2.putText(frame, f"{prompt}  ({remaining:.1f}s)", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Maestro Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

#=================================================================================================
if __name__ == "__main__":
    main()
