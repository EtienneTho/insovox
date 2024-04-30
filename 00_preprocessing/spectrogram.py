import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


def raised_cosine(x, mu, s):
    return 1 / 2 / s * (1 + np.cos((x - mu) / s * math.pi)) * s

def compute_complex_spectrogram(waveform, window_size, frame_step):
    # % Figure out the fft_size (twice the window size because we are doing
    # % circular convolution).  We'll place the windowed time-domain signal into
    # % the middle of the buffer (zeros before and after the signal in the array.)
    fft_size = 2 * window_size
    fftB = math.floor(window_size / 2)
    fftE = fftB + window_size
    fftp_buffer = np.zeros((fft_size))

    r = len(waveform)
    frame_count = math.floor((r - window_size) / frame_step) + 1
    spectrogram = np.multiply(np.zeros((fft_size, frame_count)),np.exp(1j*np.zeros((fft_size, frame_count))))
    h = 0.54 - 0.46 * np.cos(2 * np.double(math.pi) * np.arange(window_size) / (window_size - 1))

    # % Note: This code loads the waveform data (times hamming) into the center
    # % of the fft_size buffer.  Then uses fftshift to rearrange things so that
    # % the 0-time is Matlab sample 1.  This means that the center of the window
    # % defines 0 phase.  After ifft, zero time will be at the same place.
    for frame_number in range(frame_count):
        waveB = frame_number * frame_step
        waveE = waveB + window_size
        fftp_buffer = 0.0 * fftp_buffer  # make sure the buffer is empty
        fftp_buffer[fftB:fftE] = waveform[waveB:waveE] * h
        fftp_buffer = np.fft.fftshift(fftp_buffer)
        spectrogram[:, frame_number] = np.transpose((np.fft.fft(fftp_buffer)))
    return spectrogram


def InvertSpectrogram(originalSpectrogram = None,frameStep = None, iterationCount = 100): 
    fftSize,__ = originalSpectrogram.shape
    windowSize = int(np.floor(fftSize / 2))
    currentSpectrogram = originalSpectrogram
    magOrigSpectrogram = np.abs(originalSpectrogram)
    regularization = np.amax(np.amax(magOrigSpectrogram)) / 100000000.0

    for iteration in np.arange(1,iterationCount+1).reshape(-1):
        # Invert the spectrogram by summing and adding      
        print(iteration)  
        waveform = InvertOneSpectrogram(currentSpectrogram,frameStep,iteration)
        # Compute the resulting (complex) spectrogram
        newSpectrogram = compute_complex_spectrogram(waveform,windowSize,frameStep)
        # Keep the original magnitude, but use the new phase (make sure we don't divide by zero.)
        newPhase = np.exp(1j * np.angle(newSpectrogram)) # / np.amax([np.amax(regularization),np.abs(newSpectrogram)])
        currentSpectrogram = np.multiply(magOrigSpectrogram,newPhase)

    waveform = InvertOneSpectrogram(currentSpectrogram,frameStep,iterationCount + 1)
    waveform[np.isnan(waveform)] = 0
    waveform[np.isinf(waveform)] = 0
    waveform = waveform / np.max(abs(waveform))

    return waveform
    
def InvertOneSpectrogram(originalSpectrogram = None,frameStep = None,iterationNumber = None): 
    fftSize,frameCount = originalSpectrogram.shape
    windowSize = int(np.floor(fftSize / 2))
    waveform = np.zeros((1,frameCount * frameStep + windowSize - 1))
    totalWindowingSum = np.zeros((1,frameCount * frameStep + windowSize - 1))

    h = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(windowSize) / (windowSize - 1))    
    fftB = int(np.floor(windowSize / 2))
    fftE = fftB + windowSize - 1

    for frameNumber in np.arange(1,frameCount+1).reshape(-1):
        waveB = 1 + (frameNumber - 1) * frameStep
        waveE = waveB + windowSize - 1
        spectralSlice = originalSpectrogram[:,frameNumber-1]
        newFrame = np.transpose(np.fft.ifft(spectralSlice))
        newFrame = np.real(np.fft.fftshift(newFrame))
        waveform[0,np.arange(waveB,waveE+1)-1] += newFrame[np.arange(fftB,fftE+1)-1]
        totalWindowingSum[0,np.arange(waveB,waveE+1)-1] +=  h 

    totalWindowingSum[0,totalWindowingSum[0,:]==0] = 1
    waveform = np.real(waveform[0,:]) / totalWindowingSum[0,:]

    return waveform