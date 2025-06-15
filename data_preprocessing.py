import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from scipy import signal

class EnhancedAudioPreprocessor:
    def __init__(self, 
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=128,
                 f_min=20,
                 f_max=8000,
                 high_pass_cutoff=100):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.high_pass_cutoff = high_pass_cutoff
        
        # Initialize transforms
        self.high_pass_filter = T.HighpassFilter(cutoff_freq=high_pass_cutoff)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        
    def apply_high_pass_filter(self, waveform):
        return self.high_pass_filter(waveform)
    
    def compute_mel_spectrogram(self, waveform):
        # Apply high-pass filter
        filtered_waveform = self.apply_high_pass_filter(waveform)
        
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(filtered_waveform)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-6)
        
        return log_mel_spec
    
    def normalize(self, spectrogram):
        # Normalize per frequency band
        mean = spectrogram.mean(dim=-1, keepdim=True)
        std = spectrogram.std(dim=-1, keepdim=True)
        return (spectrogram - mean) / (std + 1e-6)
    
    def preprocess(self, waveform):
        # Compute mel spectrogram
        mel_spec = self.compute_mel_spectrogram(waveform)
        
        # Normalize
        normalized_spec = self.normalize(mel_spec)
        
        return normalized_spec

class AudioAugmenter:
    def __init__(self, 
                 sample_rate=16000,
                 time_stretch_range=(0.8, 1.2),
                 pitch_shift_range=(-2, 2),
                 noise_level=0.01,
                 time_mask_ratio=0.1,
                 freq_mask_ratio=0.1):
        self.sample_rate = sample_rate
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_level = noise_level
        self.time_mask_ratio = time_mask_ratio
        self.freq_mask_ratio = freq_mask_ratio
        
    def time_stretch(self, waveform):
        stretch_factor = np.random.uniform(*self.time_stretch_range)
        return torchaudio.transforms.TimeStretch()(waveform, stretch_factor)
    
    def pitch_shift(self, waveform):
        n_steps = np.random.randint(*self.pitch_shift_range)
        return torchaudio.transforms.PitchShift(self.sample_rate, n_steps)(waveform)
    
    def add_noise(self, waveform):
        noise = torch.randn_like(waveform) * self.noise_level
        return waveform + noise
    
    def time_mask(self, spectrogram):
        time_steps = spectrogram.shape[-1]
        mask_length = int(time_steps * self.time_mask_ratio)
        mask_start = np.random.randint(0, time_steps - mask_length)
        masked = spectrogram.clone()
        masked[..., mask_start:mask_start + mask_length] = 0
        return masked
    
    def freq_mask(self, spectrogram):
        freq_bins = spectrogram.shape[-2]
        mask_length = int(freq_bins * self.freq_mask_ratio)
        mask_start = np.random.randint(0, freq_bins - mask_length)
        masked = spectrogram.clone()
        masked[..., mask_start:mask_start + mask_length, :] = 0
        return masked
    
    def augment(self, waveform, spectrogram):
        # Apply time domain augmentations
        if np.random.random() < 0.5:
            waveform = self.time_stretch(waveform)
        if np.random.random() < 0.5:
            waveform = self.pitch_shift(waveform)
        if np.random.random() < 0.5:
            waveform = self.add_noise(waveform)
            
        # Apply frequency domain augmentations
        if np.random.random() < 0.5:
            spectrogram = self.time_mask(spectrogram)
        if np.random.random() < 0.5:
            spectrogram = self.freq_mask(spectrogram)
            
        return waveform, spectrogram 