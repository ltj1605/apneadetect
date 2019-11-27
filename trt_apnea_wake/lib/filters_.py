from scipy import signal

def smoothing(data):
    data = signal.savgol_filter(data, polyorder=3, window_length=33)
    data = signal.savgol_filter(data, polyorder=3, window_length=33)
    data = signal.savgol_filter(data, polyorder=3, window_length=33)
    return data

def butter_lowpass(cutOff, fs, order=5):
    '''
    low pass filter coefficient extractor using IIR filter
    see detail --> https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html
    :param cutOff: bandwidth of lowpass filer, type : int or float
    :param fs: sampling frequency of input dawta, type : int
    :param order: filter order, type : int
    :return: numerator (b) and denominator (a) polynomials of the IIR filter.
    '''
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = signal.butter(order, normalCutoff, btype='low')
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    """

    :param data: input signal
    :param cutOff: bandwidth of lowpass filer, type : int or float
    :param fs: sampling frequency of input dawta, type : int
    :param order: filter order, type : int
    :return: filtered signal
    """
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    """

    :param lowcut:
    :param highcut:
    :param fs:
    :param order:
    :return:
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band',output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    :param data:
    :param lowcut:
    :param highcut:
    :param fs:
    :param order:
    :return:
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfilt(sos, data)
    return y

if __name__ == '__main__':
    print('filters main function')