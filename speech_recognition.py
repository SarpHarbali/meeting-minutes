import torchaudio
import torchaudio.transforms as transforms
import torch


waveform, sample_rate = torchaudio.load("train-clean-360/train-clean-360/14/208/14-208-0000.flac", backend="soundfile")

mel_transform = transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=128,
    n_fft=1024,
    hop_length=256
)

mel_spectrogram = mel_transform(waveform)

mfcc_transform = transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=13,
    melkwargs={"n_fft": 1024, "n_mels": 128, "hop_length": 256}
)

mfcc = mfcc_transform(waveform)