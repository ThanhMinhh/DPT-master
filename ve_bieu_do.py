import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_all_6_features(audio_path1, audio_path2):
    print("Dang doc file va trich xuat tron bo 6 dac trung... Vui long doi...")
    # Tải 10 giây của 2 bài nhạc để vẽ
    y1, sr1 = librosa.load(audio_path1, sr=22050, duration=10) 
    y2, sr2 = librosa.load(audio_path2, sr=22050, duration=10)

    # --- TÍNH TOÁN 6 ĐẶC TRƯNG ---
    # 1. Average Energy (Là 1 đường thẳng hằng số)
    ave_energy1 = np.full_like(librosa.feature.rms(y=y1)[0], np.average(y1*y1))
    ave_energy2 = np.full_like(librosa.feature.rms(y=y2)[0], np.average(y2*y2))

    # 2. RMS (Root Mean Square)
    rms1 = librosa.feature.rms(y=y1)[0]
    rms2 = librosa.feature.rms(y=y2)[0]

    # 3. ZCR (Zero Crossing Rate)
    zcr1 = librosa.feature.zero_crossing_rate(y=y1)[0]
    zcr2 = librosa.feature.zero_crossing_rate(y=y2)[0]
    
    # 4. Spectral Centroid
    cent1 = librosa.feature.spectral_centroid(y=y1, sr=sr1)[0]
    cent2 = librosa.feature.spectral_centroid(y=y2, sr=sr2)[0]

    # 5. Spectral Bandwidth
    bw1 = librosa.feature.spectral_bandwidth(y=y1, sr=sr1)[0]
    bw2 = librosa.feature.spectral_bandwidth(y=y2, sr=sr2)[0]

    # 6. Spectral Rolloff
    rolloff1 = librosa.feature.spectral_rolloff(y=y1, sr=sr1)[0]
    rolloff2 = librosa.feature.spectral_rolloff(y=y2, sr=sr2)[0]

    # --- BẮT ĐẦU VẼ ĐỒ THỊ ---
    fig, axs = plt.subplots(6, 2, figsize=(16, 20)) # Tạo lưới 6 hàng x 2 cột
    fig.suptitle('SO SÁNH 6 ĐẶC TRƯNG ÂM THANH - KẾT QUẢ TRUNG GIAN', fontsize=20, fontweight='bold', y=0.98)

    features1 = [ave_energy1, rms1, zcr1, cent1, bw1, rolloff1]
    features2 = [ave_energy2, rms2, zcr2, cent2, bw2, rolloff2]
    titles = [
        '1. Average Energy (Năng lượng toàn cục)', 
        '2. RMS (Năng lượng cục bộ)', 
        '3. ZCR (Tốc độ đổi dấu)', 
        '4. Spectral Centroid (Trọng tâm phổ)', 
        '5. Spectral Bandwidth (Băng thông phổ)', 
        '6. Spectral Rolloff (Điểm cắt phổ)'
    ]
    colors = ['black', 'blue', 'red', 'green', 'purple', 'orange']

    for i in range(6):
        # Vẽ cột bên trái (File Truy vấn)
        axs[i, 0].plot(features1[i], color=colors[i])
        axs[i, 0].set_title(f'{titles[i]} - FILE TRUY VẤN')
        axs[i, 0].grid(True, linestyle='--', alpha=0.6)

        # Vẽ cột bên phải (File CSDL)
        axs[i, 1].plot(features2[i], color=colors[i])
        axs[i, 1].set_title(f'{titles[i]} - FILE KẾT QUẢ TRONG CSDL')
        axs[i, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("toan_bo_6_dac_trung.png", dpi=150) # Lưu ảnh chất lượng cao
    print("Ve thanh cong! Hay mo file 'toan_bo_6_dac_trung.png' ra xem nhe.")

# Chọn 2 bài bất kỳ trong CSDL để test
file_truy_van = "audio/track_10s_001_classical.00012.wav" 
file_csdl = "audio/track_10s_004_classical.00051.wav"

if os.path.exists(file_truy_van) and os.path.exists(file_csdl):
    plot_all_6_features(file_truy_van, file_csdl)
else:
    print("Không tìm thấy file nhạc để vẽ!")
