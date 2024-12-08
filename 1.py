import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Đường dẫn tới file âm thanh
audio_path = 'ffb86d3c_nohash_0.wav'

# 1. Load và cắt âm thanh để bỏ im lặng
y, sr = librosa.load(audio_path, sr=None)
y_trimmed, _ = librosa.effects.trim(y, top_db=30)

# 2. Tính STFT và chuyển đổi sang decibel
S = np.abs(librosa.stft(y_trimmed))
S_db = librosa.amplitude_to_db(S, ref=np.max)

# 3. Loại bỏ các cột không có giá trị
threshold = -50  # Ngưỡng giá trị dB
valid_columns = np.any(S_db > threshold, axis=0)  # Tìm các cột có giá trị > threshold
S_db_cropped = S_db[:, valid_columns]  # Giữ lại chỉ các cột hợp lệ

# 4. Vẽ biểu đồ của vùng giá trị
plt.figure(figsize=(12, 8))
librosa.display.specshow(
    S_db_cropped, sr=sr, x_axis='time', y_axis='hz', cmap='magma'
)

# 5. Lưu biểu đồ thành file ảnh
output_image = 'spectrogram_cropped.png'
plt.savefig(output_image)
plt.show()

print(f"Biểu đồ đã được lưu tại: {output_image}")
