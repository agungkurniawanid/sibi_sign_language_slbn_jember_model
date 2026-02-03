import cv2
import numpy as np
import os
import mediapipe as mp

# IMPOR DARI CONFIG (Agar tetap satu sumber kebenaran)
from Config import DATA_PATH, actions, no_sequences, sequence_length

# ==========================================
# 1. SETUP MEDIAPIPE
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
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
    # Tubuh
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

# ==========================================
# 2. FUNGSI UTAMA (Dengan Menu Pilihan)
# ==========================================
def main():
    # --- STEP 1: PILIH KATA ---
    print("\n=== DAFTAR KATA YANG TERSEDIA ===")
    for idx, name in enumerate(actions):
        print(f"[{idx}] {name}")
    print("=================================")
    
    try:
        choice = int(input("Masukkan NOMOR kata yang ingin direkam (contoh: 0): "))
        if choice < 0 or choice >= len(actions):
            print("❌ Nomor tidak valid!")
            return
    except ValueError:
        print("❌ Masukkan angka!")
        return

    # Kata yang dipilih
    target_action = actions[choice] 
    print(f"\n🎥 Anda memilih untuk merekam: '{target_action}'")
    print(f"🎯 Target: {no_sequences} video.")

    # --- STEP 2: CEK DATA YANG SUDAH ADA (Smart Resume) ---
    # Kita cek folder target, video nomor berapa yang terakhir dibuat?
    action_path = os.path.join(DATA_PATH, target_action)
    
    # Ambil list folder yang isinya angka saja
    existing_sequences = [int(f) for f in os.listdir(action_path) if f.isdigit()]
    
    # Tentukan start sequence
    if len(existing_sequences) == 0:
        start_seq = 0
    else:
        # Cek folder terakhir yang FULLY recorded (biasanya kita ambil max + 1)
        # Tapi hati-hati, amannya kita cek folder kosong atau tidak. 
        # Simplifikasinya: Lanjutkan dari angka terbesar + 1
        start_seq = max(existing_sequences) + 1 # Jika ada folder 0,1,2 -> mulai dari 3
        
        # Cek apakah folder terakhir itu kosong (tadi kepotong)? 
        # Jika folder terakhir (misal folder '2') kosong, kita overwrite folder '2'
        last_folder_path = os.path.join(action_path, str(max(existing_sequences)))
        if len(os.listdir(last_folder_path)) == 0:
            start_seq = max(existing_sequences)
            print(f"⚠️ Folder {start_seq} terdeteksi kosong/rusak. Merekam ulang folder {start_seq}...")

    if start_seq >= no_sequences:
        print(f"✅ Kata '{target_action}' sudah lengkap ({no_sequences} video). Tidak perlu merekam lagi.")
        return

    print(f"🚀 Memulai rekaman dari video nomor: {start_seq} sampai {no_sequences-1}")
    input("Tekan ENTER untuk menyalakan kamera...")

    # --- STEP 3: BUKA KAMERA ---
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop Sequence (Hanya untuk kata yang dipilih)
        for sequence in range(start_seq, no_sequences):
            
            # Loop Frame
            for frame_num in range(sequence_length):

                ret, frame = cap.read()
                if not ret: break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                # --- TAMPILAN LAYAR ---
                if frame_num == 0: 
                    cv2.putText(image, 'BERSIAP...', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'MEREKAM: {target_action} | Video No: {sequence}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000) # Jeda 2 detik tiap awal video baru
                else: 
                    cv2.putText(image, f'REC: {target_action} | Video No: {sequence} | Frame: {frame_num}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                # --- SIMPAN DATA ---
                keypoints = extract_keypoints(results)
                # Folder path sudah dibuat oleh SetupFolders.py, jadi aman
                npy_path = os.path.join(DATA_PATH, target_action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Tombol Q untuk keluar paksa
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("\n🛑 Rekaman dihentikan paksa oleh pengguna.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Jeda sebentar antar video (opsional, agar tidak terlalu ngos-ngosan)
            # cv2.waitKey(500) 

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Selesai merekam '{target_action}' sampai target!")

if __name__ == "__main__":
    main()