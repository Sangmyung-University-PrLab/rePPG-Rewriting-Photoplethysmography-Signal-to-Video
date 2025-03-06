import os
import numpy as np
import cv2
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def filter_bandpass(arr, srate, band):
    nyq = 60 * srate / 2
    coef_vector = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
    return signal.filtfilt(*coef_vector, arr)

def detrend_signal(arr, wsize):
    norm = np.convolve(np.ones(len(arr)), np.ones(wsize), mode='same')
    mean = np.convolve(arr, np.ones(wsize), mode='same') / norm
    return (arr - mean) / (mean + 1e-15)

class VideoSignalDataset(Dataset):
    def __init__(self, directory, transform=None, window_size=150, step_size=30, double=False):
        self.directory = directory
        self.transform = transform
        self.window_size = window_size
        self.double = double
        self.step_size = step_size
        self.video_files = [f for f in os.listdir(directory) if f.endswith('face.npy')]
        self.signal_files = [f for f in os.listdir(directory) if f.endswith('sig.npy')]
        self.paired_files = self._pair_files()
        self.data_indices = self._generate_indices()

    def _pair_files(self):
        paired_files = []
        for video_file in self.video_files:
            base_name = video_file.replace('face.npy', '')
            signal_file = base_name + 'sig.npy'
            if signal_file in self.signal_files:
                paired_files.append((video_file, signal_file))
        return paired_files

    def _generate_indices(self):
        indices = []
        for video_file, signal_file in self.paired_files:
            video_path = os.path.join(self.directory, video_file)
            video_data = np.load(video_path)
            num_windows = (len(video_data) - self.window_size) // self.step_size + 1
            for i in range(num_windows):
                start_idx = i * self.step_size
                end_idx = start_idx + self.window_size
                indices.append((video_file, signal_file, start_idx, end_idx))
        return indices

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        video_file, signal_file, start_idx, end_idx = self.data_indices[idx]
        video_path = os.path.join(self.directory, video_file)
        signal_path = os.path.join(self.directory, signal_file)

        video_data = np.load(video_path)[start_idx:end_idx] / 255.
        signal_data = np.load(signal_path, allow_pickle=True).item()

        cppg_signal = signal_data['cppg_signal'][start_idx:end_idx]

        # Normalize the signals
        scaler = MinMaxScaler()
        cppg_signal = scaler.fit_transform(cppg_signal.reshape(-1, 1)).flatten()
        
        # Detrend and filter the signals
        cppg_signal = detrend_signal(cppg_signal, 30)
        cppg_signal = filter_bandpass(cppg_signal, 30, (42, 180))
        cppg_signal = (cppg_signal - cppg_signal.min()) / (cppg_signal.max() - cppg_signal.min())

        if self.transform:
            video_data = self.transform(video_data)

        return video_data, cppg_signal

def get_dataloader(directory, batch_size=1, shuffle=True, transform=None, window_size=150, step_size=30):
    dataset = VideoSignalDataset(directory, transform=transform, window_size=window_size, step_size=step_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataloader_new(directory, batch_size=1, shuffle=True, transform=None, window_size=150, step_size=30):
    dataset = VideoSignalDataset(directory, transform=transform, window_size=window_size, step_size=step_size, double=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    train_loader = get_dataloader_new("H:\\PURE_TEST\\train")
    for batch_idx, (video_data, cppg_signal, rppg_signal, new_cppg_signal, new_rppg_signal) in enumerate(train_loader):
        print(video_data.shape, cppg_signal.shape, rppg_signal.shape, new_cppg_signal.shape, new_rppg_signal.shape)
        plt.plot(new_cppg_signal[0], label='New CPPG Signal')
        plt.plot(cppg_signal[0], label='CPPG Signal')
        plt.legend()
        plt.show()