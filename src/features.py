import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

def extract_signal_features(signal, sampling_rate=25600):
    #Extracts time-domain and frequency-domain indicators from a 1D raw vibration array.
    
    # Time-Domain Features
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    p2p = np.ptp(signal)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    skew_val = skew(signal)
    kurt_val = kurtosis(signal) # Excess kurtosis
    crest_factor = peak / rms if rms != 0 else 0
    shape_factor = rms / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0
    impulse_factor = peak / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0
    
    # Frequency-Domain Features (FFT)
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freqs = np.fft.rfftfreq(len(signal), d=1/sampling_rate)
    
    fft_mean = np.mean(fft_vals)
    fft_std = np.std(fft_vals)
    fft_max = np.max(fft_vals)
    spectral_centroid = np.sum(fft_freqs * fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) != 0 else 0
    
    return {
        "RMS": rms,
        "Peak": peak,
        "PeakToPeak": p2p,
        "Mean": mean_val,
        "Std": std_val,
        "Skewness": skew_val,
        "Kurtosis": kurt_val,
        "CrestFactor": crest_factor,
        "ShapeFactor": shape_factor,
        "ImpulseFactor": impulse_factor,
        "FFT_Mean": fft_mean,
        "FFT_Std": fft_std,
        "FFT_Max": fft_max,
        "SpectralCentroid": spectral_centroid
    }