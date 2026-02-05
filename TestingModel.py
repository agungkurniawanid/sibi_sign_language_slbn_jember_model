# ==================== IMPORT ====================
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

# ==================== CONFIGURATION ====================
ACTIONS = np.array(['Saya', 'Buah']) # Pastikan urutan sama dengan Config.py
COLORS = [(245,117,16), (117,245,16)] 
THRESHOLD = 0.7 

MODEL_PATH = 'action.h5' 
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model '{MODEL_PATH}' tidak ditemukan!")
    exit()

print("🔄 Memuat Model AI...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Siap! Tekan 'M' untuk ubah Mirror, 'Q' untuk keluar.")

# ==================== MEDIAPIPE SETUP ====================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==================== FUNCTIONS ====================
def mediapipe_detection(image, model, is_mirrored):
    # --- FITUR MIRRORING (CONDITIONAL) ---
    if is_mirrored:
        image = cv2.flip(image, 1) # Flip Horizontal
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Wajah
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    # Pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    # Tangan Kiri
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    # Tangan Kanan
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color = colors[num] if num < len(colors) else (255, 255, 255)
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), color, -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# ==================== REALTIME DETECTION ====================
sequence = []
sentence = []
predictions = []
mirror_mode = True # Default Start: ON

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Deteksi dengan variabel 'mirror_mode'
        image, results = mediapipe_detection(frame, holistic, mirror_mode)
        draw_styled_landmarks(image, results)

        # 2. Logika Prediksi
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            best_class_index = np.argmax(res)
            confidence = res[best_class_index]
            predictions.append(best_class_index)

            if np.unique(predictions[-10:])[0] == best_class_index: 
                if confidence > THRESHOLD: 
                    if len(sentence) > 0: 
                        if ACTIONS[best_class_index] != sentence[-1]:
                            sentence.append(ACTIONS[best_class_index])
                    else:
                        sentence.append(ACTIONS[best_class_index])

            if len(sentence) > 5: sentence = sentence[-5:]
            image = prob_viz(res, ACTIONS, image, COLORS)
            
        # 3. Tampilan UI
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Indikator Mirror Mode (Pojok Kanan Atas)
        if mirror_mode:
            status_text = "MIRROR: ON"
            status_color = (0, 255, 0) # Hijau
        else:
            status_text = "MIRROR: OFF"
            status_color = (0, 0, 255) # Merah
            
        cv2.putText(image, status_text, (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(image, "Tekan 'M' untuk Toggle", (380, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow('SIBI Realtime Test', image)

        # 4. Handle Keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Keluar
            break
        if key == ord('m'): # Ganti Mode Mirror
            mirror_mode = not mirror_mode
            print(f"🔄 Mode Mirror diubah ke: {mirror_mode}")
            sequence = [] # Reset sequence agar model tidak bingung transisi

cap.release()
cv2.destroyAllWindows()