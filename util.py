import cv2
import numpy as np

def get_added_frequencies(N):
    num_freqs = np.random.randint(0,N) # will never set freq 0 to anything, so only up to N-1

    freq_pos = np.random.choice(np.arange(1,N), size=num_freqs, replace=False) # never set freq 0 to anything

    print(f'num_freqs:\t\t{num_freqs}')

    # TODO: figure out best distribution
    rand_freqs_real = np.random.uniform(-100,100,(num_freqs,1))
    rand_freqs_imag = np.random.uniform(-100,100,(num_freqs,1))
    rand_freqs = rand_freqs_real + 1j*rand_freqs_imag

    X = np.zeros((N,1)).astype(np.complex64)
    X[freq_pos] = rand_freqs

    return X

def get_frame_shift(X):
    x = np.fft.ifft(X, axis=0)
    w_shift = np.real(x).astype(np.int32)
    h_shift = np.imag(x).astype(np.int32)
    return w_shift, h_shift

def get_video_info(cap):
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    return num_frames, fps, height, width

# NOTE: potential issue: might want to have some classifier to find regions of video with multiple colors (i.e. not monotone)
# maybe just check for number of unique colors in the region
def get_crop_pos(width, frame_width, height, frame_height):
    return np.random.randint(width-frame_width), np.random.randint(height-frame_height)