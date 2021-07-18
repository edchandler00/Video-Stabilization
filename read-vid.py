import cv2
import numpy as np
from util import get_added_frequencies, get_frame_shift, get_video_info, get_crop_pos
import matplotlib.pyplot as plt

N = 120
frame_width, frame_height = 64,64 # TODO: figure out best size (hyperparameter)

cap = cv2.VideoCapture('Videos/shred.mov')

num_frames, fps, height, width = get_video_info(cap)

print(fps)
print(height)
print(width)

X = get_added_frequencies(N)
w_shift, h_shift = get_frame_shift(X)

# w_shift_neg, h_shift_neg = get_frame_shift(-X)

# TODO: only return if pixel values are different enough
w_start,h_start = get_crop_pos(width, frame_width, height, frame_height)

count = 0
for i in range(N):
#while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("cannot recieve frame")
        break

    w, h = w_start + w_shift[i,0], h_start + h_shift[i,0]
    # w, h = w_start + w_shift[i,0] + w_shift_neg[i,0], h_start + h_shift[i,0] + h_shift_neg[i,0]
    cv2.imshow('frame', frame[h:h+frame_height,w:w+frame_width])

    count+=1
    if cv2.waitKey(33) == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)

ax1.stem(np.arange(N), np.real(X))
ax1.set_title("Real Components")

ax2.stem(np.arange(N), np.imag(X))
ax2.set_title("Imag Components")

ax3.plot(w_shift)
ax3.set_title("w_shift")

ax4.plot(h_shift)
ax4.set_title("h_shift");

plt.show();