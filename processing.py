import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import glob
import gc


train_data_path = 'train/'
test_data_path = 'test/'
y = []

os.makedirs(train_data_path, exist_ok=True)

#hàm tạo ảnh
def create_image(filename, name, file_path):
    """Convert audio to mel-spectrogram image and save as PNG."""
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    output_path = os.path.join(file_path, f"{name}.png")
    plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S


categories = {
    'bird': [1, 0, 0],
    'cat': [0, 1, 0],
    'dog': [0, 0, 1],
}
 


base_path = 'Animals/'  
for category, label in categories.items():
    wav_path = os.path.join(base_path, category, '*.wav') 
    for file in glob.glob(wav_path):
        name = os.path.basename(file).split('.')[0]
        create_image(file, name, train_data_path)
        y.append(label)
    gc.collect()


#lưu y vào file label.txt
with open("label.txt", "w") as f:
    for label in y:
        f.write(f"{label}\n")

print("Process done!")
