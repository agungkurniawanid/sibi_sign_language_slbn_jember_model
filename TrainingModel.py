import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# IMPOR CONFIG
# Pastikan Config.py actions = np.array(['Saya', 'Buah'])
from Config import DATA_PATH, actions, no_sequences, sequence_length

# ==========================================
# 1. PERSIAPAN DATA (LOAD & SPLIT)
# ==========================================
print("🔄 Memuat dataset...")
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    # Filter hanya folder angka (untuk menghindari error file sistem/folder kosong)
    action_path = os.path.join(DATA_PATH, action)
    sequence_folders = [f for f in os.listdir(action_path) if f.isdigit()]
    
    for sequence in sequence_folders:
        window = []
        # Load 30 frame per video
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Membagi data Training (95%) dan Testing (5%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(f"✅ Data Training Siap: {X_train.shape}")
print(f"✅ Data Testing Siap: {X_test.shape}")
print(f"🎯 Target Kelas ({actions.shape[0]}): {actions}")

# ==========================================
# 2. DEFINISI MODEL (SESUAI REFERENSI ANDA)
# ==========================================
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()

# Layer 1: LSTM (Input Shape disesuaikan dengan data Anda: 30 frame, 1662 keypoints)
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))

# Layer 2: LSTM
model.add(LSTM(128, return_sequences=True, activation='relu'))

# Layer 3: LSTM (return_sequences=False karena masuk ke Dense layer)
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Layer 4 & 5: Dense (Berpikir)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Output Layer (Softmax untuk probabilitas kelas)
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ==========================================
# 3. TRAINING MODEL
# ==========================================
print("\n🚀 Memulai Training (2000 Epochs)...")
print("   (Tekan Ctrl+C di terminal jika ingin berhenti lebih awal)")

# Melatih model
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

# ==========================================
# 4. SIMPAN MODEL & RINGKASAN
# ==========================================
model.summary()

# Simpan hasil training agar bisa dipakai
model_filename = 'action.h5'
model.save(model_filename)
print(f"\n💾 Model BERHASIL disimpan sebagai: '{model_filename}'")

# Tes cepat akurasi pada data testing (yang tidak ikut training)
print("\n🧪 Evaluasi pada Data Test:")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Akurasi Final pada Data Baru: {accuracy*100:.2f}%")