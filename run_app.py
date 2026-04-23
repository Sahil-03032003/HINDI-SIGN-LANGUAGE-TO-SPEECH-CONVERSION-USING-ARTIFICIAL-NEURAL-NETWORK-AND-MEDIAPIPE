import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from gtts import gTTS
import os
from PIL import ImageFont, ImageDraw, Image

# --- SETTINGS ---
MODEL_PATH = 'model_mediapipe.h5'
CLASSES_PATH = 'classes.npy'
AUDIO_FILE = "sentence_audio.mp3"
FONT_PATH = "NotoSansDevanagari-Regular.ttf" # Ensure this file exists

# --- LOAD MODEL ---
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# --- VARIABLES & LINGUISTIC MAPS ---
sentence_buffer = []
letter_buffer = []
current_letter = ""
letter_start_time = 0
CONFIRM_TIME = 1.0  # Seconds to hold sign to confirm

# NEW: Dictionary to convert full vowels into their attached Matra symbols
matra_map = {
    'आ': 'ा', 'इ': 'ि', 'ई': 'ी', 'उ': 'ु', 'ऊ': 'ू',
    'ए': 'े', 'ऐ': 'ै', 'ओ': 'ो', 'औ': 'ौ', 'ऋ': 'ृ'
}

def text_to_speech(text):
    if not text.strip(): return
    try:
        tts = gTTS(text=text, lang='hi')
        tts.save(AUDIO_FILE)
        os.system(f'start "" "{AUDIO_FILE}"')
    except Exception as e:
        print("TTS Error:", e)

def draw_text(img, text, pos, color=(0,0,0), size=32):
    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except:
        font = ImageFont.load_default()
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Mirror view
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    predicted_character = ""
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw Skeleton
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract Data
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Predict
            input_data = np.array([data_aux])
            prediction = model.predict(input_data, verbose=0)
            predicted_index = np.argmax(prediction)
            predicted_character = classes[predicted_index]
            confidence = np.max(prediction)

            # Draw Box & Label
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            if confidence > 0.7:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame = draw_text(frame, predicted_character, (x1, y1 - 40), (255, 0, 0))

    # --- LOGIC FOR SENTENCE FORMATION (UPDATED WITH MATRA LOGIC) ---
    if predicted_character:
        if predicted_character == current_letter:
            # If the user holds the sign for 1 second...
            if time.time() - letter_start_time > CONFIRM_TIME:
                
                # Check if the confirmed letter is a Vowel
                if predicted_character in matra_map:
                    # If there is already a letter in the buffer, attach the Matra symbol
                    if len(letter_buffer) > 0 and letter_buffer[-1] != " ":
                        letter_buffer.append(matra_map[predicted_character])
                    else:
                        # If starting a new word, use the full vowel (e.g., 'ओ')
                        letter_buffer.append(predicted_character)
                else:
                    # If it's a normal consonant, just append it
                    letter_buffer.append(predicted_character)
                
                print(f"Added to buffer: {letter_buffer[-1]}")
                current_letter = ""  # Reset timer so it doesn't spam the letter
                
        else:
            # The user changed their hand sign, restart the timer
            current_letter = predicted_character
            letter_start_time = time.time()
    else:
        current_letter = ""
    
    # UI Display
    curr_word = "".join(letter_buffer)
    curr_sentence = " ".join(sentence_buffer)
    
    # Display logic
    cv2.rectangle(frame, (0, 0), (W, 80), (245, 117, 16), -1) # Header
    frame = draw_text(frame, f"Word: {curr_word}", (10, 10), (255,255,255))
    frame = draw_text(frame, f"Sent: {curr_sentence}", (10, 45), (255,255,255), size=20)
    
    cv2.imshow('Real-Time HSL to Speech', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    
    # Controls
    if key == ord('n'): # Spacebar alternative ('n')
        letter_buffer.append(" ")
    if key == 8: # Backspace
        if letter_buffer: letter_buffer.pop()
    if key == ord('b'): # Add word to sentence
        if letter_buffer:
            sentence_buffer.append("".join(letter_buffer))
            letter_buffer = []
    if key == ord('s'): # Speak
        text_to_speech(" ".join(sentence_buffer))
        sentence_buffer = []

cap.release()
cv2.destroyAllWindows()