from scipy import signal

def morlet_cwt(x, fs, omega, n_channels):
    freq = np.linspace(fs/20, fs/2, n_channels)
    widths = omega*fs/(2*freq*np.pi)
    return np.abs(signal.cwt(x, signal.morlet2, widths, w=omega)