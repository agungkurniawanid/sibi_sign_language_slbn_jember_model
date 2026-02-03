import os
# IMPOR DARI CONFIG (Agar data selalu sama)
from Config import DATA_PATH, actions, no_sequences

def main():
    print(f"📂 Memulai pembuatan folder di: {DATA_PATH}")
    print(f"📝 Total Kata: {len(actions)}")
    print(f"🎬 Jumlah Video per Kata: {no_sequences}")
    
    for action in actions:
        # Loop untuk membuat folder sequence (0 sampai 99)
        for sequence in range(no_sequences):
            try:
                # Membuat path: SIBI_Dataset_Keypoints/Kata/NomorSequence
                target_path = os.path.join(DATA_PATH, action, str(sequence))
                os.makedirs(target_path, exist_ok=True)
                
            except Exception as e:
                print(f"❌ Error membuat folder {action}/{sequence}: {e}")
                
    print("\n✅ SUKSES! Semua folder telah siap.")

if __name__ == "__main__":
    main()