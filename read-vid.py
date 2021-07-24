import cv2
import numpy as np
from util import get_added_frequencies, get_frame_shift, get_video_info, get_crop_pos
import matplotlib.pyplot as plt



frame_width, frame_height = 500,500 # TODO: figure out best size (hyperparameter)
N = 120

cap = cv2.VideoCapture('/Users/eddiechandler/Documents/Personal_Projects/Videos/shred.mov')

num_frames, fps, height, width = get_video_info(cap)

print(fps)
print(f'height {height}')
print(f'width {width}')

X = get_added_frequencies(N)
w_shift, h_shift = get_frame_shift(X)

# w_shift_neg, h_shift_neg = get_frame_shift(-X)

# TODO: only return if pixel values are different enough
# w_start,h_start = get_crop_pos(width, frame_width, height, frame_height)

w_start = int((width - frame_width) / 2)
h_start = int((height - frame_height) / 2)

reconstructed_t_pos = []
reconstructed_l_pos = []
reconstructed_b_pos = []
reconstructed_r_pos = []

the_frames = []

for i in range(N):
#while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("cannot recieve frame")
        break

    amount = 10

    orig_with_trap = frame[h_start-amount:h_start+frame_height+amount, w_start-amount:w_start+frame_width+amount]
    
    pt_a = np.array([amount,amount])
    pt_b = np.array([amount,frame_height+amount])
    pt_c = np.array([frame_width+amount,frame_height+amount])
    pt_d = np.array([frame_width+amount,amount])

    # pt_a += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]
    # pt_b += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]
    # pt_c += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]
    # pt_d += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]

    pt_a += np.array([np.random.normal(0,amount),np.random.normal(0,amount)],dtype=np.int32)
    pt_b += np.array([np.random.normal(0,amount),np.random.normal(0,amount)],dtype=np.int32)
    pt_c += np.array([np.random.normal(0,amount),np.random.normal(0,amount)],dtype=np.int32)
    pt_d += np.array([np.random.normal(0,amount),np.random.normal(0,amount)],dtype=np.int32)

    # TODO: figure out this!!! (pretty much copied)
    orig_with_trap = cv2.rectangle(orig_with_trap, (amount,amount), (frame_width+amount, frame_height+amount), (255, 0, 0), 1)
    pts = np.array([pt_a, pt_b, pt_c, pt_d], np.int32)
    # print(pts)
    pts = pts.reshape((-1, 1, 2))
    # print(pts)

    orig_with_trap = cv2.polylines(img=orig_with_trap, pts=[pts], isClosed=True, color=(0,0,255), thickness=1)
    orig_with_trap = cv2.rectangle(orig_with_trap, (2*amount,2*amount), (frame_width, frame_height), (0,255,0), 1)
    
    M_input_pts = np.float32([pt_a, pt_b, pt_c, pt_d])
    M_output_pts = np.float32([[0, 0],
                            [0, frame_height - 1],
                            [frame_width - 1, frame_height - 1],
                            [frame_width - 1, 0]])

    M = cv2.getPerspectiveTransform(M_input_pts, M_output_pts)
    
    added_disturbance = cv2.warpPerspective(orig_with_trap, M, (frame_width, frame_height))

    reconstruction_M = np.linalg.pinv(M)

    # TODO: try cv2.WARP_INVERSE_MAP
    reconstructed = cv2.warpPerspective(added_disturbance, reconstruction_M, (frame_width+2*amount, frame_height+2*amount))
    # print(cv2.perspectiveTransform())
    temp_t_l = np.matmul(reconstruction_M, [0,0,1])
    temp_t_l /= temp_t_l[2]
    temp_t_l = temp_t_l[:2]

    temp_b_l = np.matmul(reconstruction_M, [0,frame_height-1,1])
    temp_b_l /= temp_b_l[2]
    temp_b_l = temp_b_l[:2]

    temp_b_r = np.matmul(reconstruction_M, [frame_width-1,frame_height-1,1])
    temp_b_r /= temp_b_r[2]
    temp_b_r = temp_b_r[:2]

    temp_t_r = np.matmul(reconstruction_M, [frame_width-1,0,1])
    temp_t_r /= temp_t_r[2]
    temp_t_r = temp_t_r[:2]

    reconstructed_t_pos.append(np.max([temp_t_l[1], temp_t_r[1]]))
    reconstructed_l_pos.append(np.max([temp_t_l[0], temp_b_l[0]]))
    reconstructed_b_pos.append(np.min([temp_b_l[1], temp_b_r[1]]))
    reconstructed_r_pos.append(np.min([temp_b_r[0], temp_t_r[0]]))

    padded_added_disturbance = np.pad(added_disturbance, ((amount,amount), (amount,amount), (0,0)), mode='constant', constant_values=0)
    # cv2.imshow('asdf', np.hstack((orig_with_trap, padded_added_disturbance, reconstructed)))

    # if cv2.waitKey(33) == ord('q'):
    #     break
    # input('enter')
    the_frames.append(reconstructed)
    
    
    
final_t_pos = np.max(reconstructed_t_pos).astype(np.int32)
final_l_pos = np.max(reconstructed_l_pos).astype(np.int32)
final_b_pos = np.min(reconstructed_b_pos).astype(np.int32)
final_r_pos = np.min(reconstructed_r_pos).astype(np.int32)

print(f'final pos: {final_t_pos}, {final_l_pos}, {final_b_pos}, {final_r_pos}')
print(len(the_frames))
print(the_frames[0].shape)
for frame in the_frames:
    cv2.imshow('reconstructed', frame[ final_t_pos:final_b_pos, final_l_pos:final_r_pos])
    if cv2.waitKey(33) == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()


# plt.scatter(np.arange(len(reconstructed_t_pos)),reconstructed_t_pos)
plt.boxplot(reconstructed_t_pos)
plt.show()


#TODO: fix M that cut off too much















 # image_center = (width/2, height/2)
    # print(image_center)
    # A = cv2.getRotationMatrix2D(center=image_center, angle=i*.2, scale=1)
    
    # A rotation and then a translation
    # rot_matrix = cv2.getRotationMatrix2D(center=image_center, angle=w_shift[i,0]/3, scale=1)
    # frame_rot = cv2.warpAffine(src=frame, M=rot_matrix, dsize=(width, height))
    # w, h = w_start + w_shift[i,0], h_start + h_shift[i,0]
    # frame_rot = frame_rot[h:h+frame_height,w:w+frame_width]

    # w, h = w_start, h_start
    # frame_rot = frame[h:h+frame_height,w:w+frame_width]

    # cv2.imshow('frame', frame_rot)



    # pt_a = np.array([10,10])
    # pt_b = np.array([10,522])
    # pt_c = np.array([522,522])
    # pt_d = np.array([522,10])



"""
N = 120
frame_width, frame_height = 512,512 # TODO: figure out best size (hyperparameter)

cap = cv2.VideoCapture('/Users/eddiechandler/Documents/Personal_Projects/Videos/shred.mov')

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
"""