import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ==========================================
# 1. IMPOR DARI SUMBER UTAMA (CONFIG)
# ==========================================
# Kita tidak perlu mengetik ulang actions atau path. Ambil dari config.
from Config import DATA_PATH, actions, sequence_length

def main():
    print("🔄 Memulai proses pemuatan dan pelabelan data...")

    # 1. Membuat Peta Label (Dictionary)
    # Contoh: {'Saya': 0, 'Makan': 1, ...}
    label_map = {label: num for num, label in enumerate(actions)}
    print(f"📝 Label Map: {label_map}")

    sequences, labels = [], []

    # 2. Loop Membaca Data
    for action in actions:
        print(f"   ↳ Memproses kata: {action}...")
        
        # Mendapatkan path folder untuk kata tersebut
        action_path = os.path.join(DATA_PATH, action)
        
        # Mendapatkan list folder sequence (video ke-1, ke-2, dst)
        # Filter: Pastikan yang dibaca hanya folder/file yang namanya angka (untuk menghindari file sistem tersembunyi)
        folder_list = [f for f in os.listdir(action_path) if f.isdigit()]
        
        # Loop setiap video (sequence)
        for sequence in np.array(folder_list).astype(int):
            window = []
            
            # Loop setiap frame (0 s/d 29)
            for frame_num in range(sequence_length):
                # Load file .npy
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            
            # Masukkan ke list utama
            sequences.append(window)
            labels.append(label_map[action])

    print("✅ Semua data berhasil dimuat!")

    # 3. Konversi ke Numpy Array
    X = np.array(sequences)
    y = tf.keras.utils.to_categorical(labels).astype(int) # One-Hot Encoding

    # 4. Membagi Data (Train & Test Split)
    # Test size 0.05 artinya 5% data disisihkan untuk ujian, 95% untuk latihan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # ==========================================
    # 5. MENAMPILKAN STATISTIK DATA (Output Visual)
    # ==========================================
    print("\n" + "="*40)
    print("📊 STATISTIK DATASET")
    print("="*40)
    print(f"Total Video Terkumpul : {len(sequences)}")
    print(f"Total Kelas (Kata)    : {len(actions)}")
    print("-" * 40)
    print(f"Bentuk X (Fitur)      : {X.shape}") 
    print(f"   👉 (Jumlah Video, Frame per Video, Keypoints)")
    print(f"Bentuk y (Label)      : {y.shape}")
    print("-" * 40)
    print(f"Data Latih (X_train)  : {X_train.shape}")
    print(f"Data Uji   (X_test)   : {X_test.shape}")
    print("="*40)

    # (Opsional) Simpan hasil olahan agar Training nanti lebih cepat
    # np.save('X_data.npy', X)
    # np.save('y_data.npy', y)
    # print("💾 Data X dan y telah disimpan (Opsional).")

if __name__ == "__main__":
    main()