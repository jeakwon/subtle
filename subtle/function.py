import numpy as np
from scipy import signal

def morlet_cwt(x, fs, omega, n_channels):
    f_nyquist = fs/2
    freq = np.linspace(f_nyquist/10, f_nyquist, n_channels)
    widths = omega*fs/(2*freq*np.pi)
    return np.abs(signal.cwt(x, signal.morlet2, widths, w=omega))