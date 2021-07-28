import cv2
import numpy as np
from util import get_added_frequencies, get_frame_shift, get_video_info, get_crop_pos
import matplotlib.pyplot as plt
import pandas as pd


# frame_width, frame_height = 500,500 # TODO: figure out best size (hyperparameter)
frame_width, frame_height = 128,128 # TODO: figure out best size (hyperparameter)
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
# the_small_frames = []

small_disturbance_frames = []
large_disturbance_frames = []

print("\n\nstarting\n\n")

the_Ms = []

ewm_span = 3

# TODO: add more lateral and rotational
amount = 3

for i in range(N+ewm_span):
#while cap.isOpened():
    # ret,frame = cap.read()
    # if not ret:
    #     print("cannot recieve frame")
    #     break

    # amount = 10
    # amount = 4
    # amount = 3
    
    # orig_with_trap = frame[h_start-amount:h_start+frame_height+amount, w_start-amount:w_start+frame_width+amount].copy() #https://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
    
    # print(type(orig_with_trap))
    pt_a = np.array([amount,amount])
    pt_b = np.array([amount,frame_height+amount])
    pt_c = np.array([frame_width+amount,frame_height+amount])
    pt_d = np.array([frame_width+amount,amount])

    # FIXME: truncate if out of bounds!
    pt_a += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]
    pt_b += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]
    pt_c += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]
    pt_d += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]

    # # TODO: figure out this!!! (pretty much copied)
    # orig_with_trap = cv2.rectangle(orig_with_trap, (amount,amount), (frame_width+amount, frame_height+amount), (255, 0, 0), 2) # create rectangle that is starting point for the polygon that determines morphing
    # pts = np.array([pt_a, pt_b, pt_c, pt_d], np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # orig_with_trap = cv2.polylines(img=orig_with_trap, pts=[pts], isClosed=True, color=(0,0,255), thickness=2) # The polygon that will lead to morphing
    
    # inside_poly_pts = []
    # for pt,amt in zip(pts, [[10,10], [10, -10], [-10,-10], [-10,10]]):
    #     inside_poly_pts.append(pt+amt)
    # inside_poly_pts = np.array(inside_poly_pts, np.int32)
    # inside_poly_pts = inside_poly_pts.reshape((-1,1,2))
    # orig_with_trap = cv2.polylines(img=orig_with_trap, pts=[inside_poly_pts], isClosed=True, color=(23,230,120), thickness=2) # The polygon that will lead to morphing
    

    M_input_pts = np.float32([pt_a, pt_b, pt_c, pt_d])
    M_output_pts = np.float32([[0, 0],
                            [0, frame_height - 1],
                            [frame_width - 1, frame_height - 1],
                            [frame_width - 1, 0]])

    M = cv2.getPerspectiveTransform(M_input_pts, M_output_pts)

    the_Ms.append(M)

the_Ms = np.array(the_Ms).reshape(N+ewm_span, 9)

the_Ms_final = np.ones((N, 9))

for i in range(8): # TODO: make sure (2,2) is always 1
    temp = the_Ms[:,i]
    s_temp = pd.Series(temp)
    the_Ms_final[:,i] = s_temp.ewm(span=ewm_span).mean()[ewm_span:]

the_Ms_final = the_Ms_final.reshape(N,3,3)
print(the_Ms_final.shape)
print(the_Ms_final)
# quit()
    
for i in range(N):
#while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("cannot recieve frame")
        break
    M = the_Ms_final[i]

    orig_with_trap = frame[h_start-amount:h_start+frame_height+amount, w_start-amount:w_start+frame_width+amount].copy() #https://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python


    added_disturbance = cv2.warpPerspective(orig_with_trap, M, (frame_width, frame_height))
    cv2.imshow('added_disturbance small', added_disturbance)
    small_disturbance_frames.append(added_disturbance)

    padded_added_disturbance = np.pad(added_disturbance, ((amount,amount), (amount,amount), (0,0)), mode='constant', constant_values=0) # This is for using imshow

    reconstruction_M = np.linalg.pinv(M)

    # TODO: try cv2.WARP_INVERSE_MAP
    reconstructed = cv2.warpPerspective(added_disturbance, reconstruction_M, (frame_width+2*amount, frame_height+2*amount))

    cv2.imshow('asdf', np.hstack((orig_with_trap, padded_added_disturbance, reconstructed)))
    the_frames.append(reconstructed)



    # # shape (3,4); last axis in order TL, BL, BR, TR
    # temp_corners = np.matmul(reconstruction_M, np.array([[0,0,1], [0,frame_height-1,1], [frame_width-1,frame_height-1,1], [frame_width-1,0,1]]).T)
    # temp_corners /= temp_corners[-1]

    # reconstructed_t_pos.append(np.max([temp_corners[1,0], temp_corners[1,3]]))
    # reconstructed_l_pos.append(np.max([temp_corners[0,0], temp_corners[0,1]]))
    # reconstructed_b_pos.append(np.min([temp_corners[1,1], temp_corners[1,2]]))
    # reconstructed_r_pos.append(np.min([temp_corners[0,2], temp_corners[0,3]]))

    #
    h_large_start = int((height - 512) / 2)
    w_large_start = int((width - 512) / 2)

    large_orig_with_trap = frame[h_large_start-amount:h_large_start+512+amount, w_large_start-amount:w_large_start+512+amount]

    poly_w_shift = int((512-frame_width)/2)
    poly_h_shift = int((512-frame_height)/2)

    pts = np.array([pt_a, pt_b, pt_c, pt_d], np.int32)
    for i, pt in enumerate(pts):
        pts[i] = pt + [poly_w_shift, poly_h_shift]
    pts = pts.reshape((-1, 1, 2))

    # # draw rectangle that is starting point for the polygon that determines morphing
    # large_orig_with_trap = cv2.rectangle(large_orig_with_trap, (amount+poly_w_shift,amount+poly_h_shift), (frame_width+amount+poly_w_shift, frame_height+amount+poly_h_shift), (255, 0, 0), 2)
    # # draw polygon that determines morphing
    # large_orig_with_trap = cv2.polylines(img=large_orig_with_trap, pts=[pts], isClosed=True, color=(0,0,255), thickness=2)
    
    cv2.imshow('larger', large_orig_with_trap)

    # Create the morphed frame
    h = 512
    w = 512
    indy, indx = np.indices((h, w), dtype=np.float32)
    indy -= poly_h_shift
    indx -= poly_w_shift

    # https://stackoverflow.com/questions/44457064/displaying-stitched-images-together-without-cutoff-using-warpaffine/44459869#44459869
    # and https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function/46524544#46524544
    indices = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    # TODO: figure out why I must take pinv
    temp_M = np.linalg.pinv(M)
    mappings = temp_M.dot(indices)
    map_x, map_y = mappings[:-1]/mappings[-1] 
    map_x = map_x.reshape(h, w).astype(np.float32) + poly_w_shift
    map_y = map_y.reshape(h, w).astype(np.float32) + poly_h_shift

    dst = cv2.remap(large_orig_with_trap, map_x, map_y, cv2.INTER_LINEAR)
    cv2.imshow('Large Disturbance', dst)

    large_disturbance_frames.append(dst[70:-70,70:-70])

    if cv2.waitKey(33) == ord('q'):
        break
    


cap.release()
cv2.destroyAllWindows()

# quit()

for small_disturbance_frame, large_disturbance_frame in zip(small_disturbance_frames, large_disturbance_frames):
    cv2.imshow('diturb small', small_disturbance_frame)
    cv2.imshow('disturb large', large_disturbance_frame)

    if cv2.waitKey(33) == ord('q'):
        break

cv2.destroyAllWindows()





"""

import cv2
import numpy as np
from util import get_added_frequencies, get_frame_shift, get_video_info, get_crop_pos
import matplotlib.pyplot as plt


frame_width, frame_height = 128,128 # TODO: figure out best size (hyperparameter)
N = 120

cap = cv2.VideoCapture('/Users/eddiechandler/Documents/Personal_Projects/Videos/shred.mov')
num_frames, fps, height, width = get_video_info(cap)

# print(fps)
# print(f'height {height}')
# print(f'width {width}')

X = get_added_frequencies(N)
w_shift, h_shift = get_frame_shift(X)

w_start = int((width - frame_width) / 2)
h_start = int((height - frame_height) / 2)

the_frames = []

the_small_frames = []

print("\n\nstarting\n\n")
for i in range(N):
#while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("cannot recieve frame")
        break

    amount = 10
    # amount = 4

    orig_with_trap = frame[h_start-amount:h_start+frame_height+amount, w_start-amount:w_start+frame_width+amount].copy() #https://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
    
    pt_a = np.array([amount,amount])
    pt_b = np.array([amount,frame_height+amount])
    pt_c = np.array([frame_width+amount,frame_height+amount])
    pt_d = np.array([frame_width+amount,amount])

    pt_a += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]
    pt_b += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]
    pt_c += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]
    pt_d += [np.random.randint(-amount,amount),np.random.randint(-amount,amount)]

    # TODO: figure out this!!! (pretty much copied)
    orig_with_trap = cv2.rectangle(orig_with_trap, (amount,amount), (frame_width+amount, frame_height+amount), (255, 0, 0), 1) # create rectangle that is starting point for the polygon that determines morphing
    pts = np.array([pt_a, pt_b, pt_c, pt_d], np.int32)
    pts = pts.reshape((-1, 1, 2))
    orig_with_trap = cv2.polylines(img=orig_with_trap, pts=[pts], isClosed=True, color=(0,0,255), thickness=1) # The polygon that will lead to morphing
    
    M_input_pts = np.float32([pt_a, pt_b, pt_c, pt_d])
    M_output_pts = np.float32([[0, 0],
                            [0, frame_height - 1],
                            [frame_width - 1, frame_height - 1],
                            [frame_width - 1, 0]])

    M = cv2.getPerspectiveTransform(M_input_pts, M_output_pts)
    
    added_disturbance = cv2.warpPerspective(orig_with_trap, M, (frame_width, frame_height))

    # cv2.imshow('added_disturbance small', added_disturbance)

    padded_added_disturbance = np.pad(added_disturbance, ((amount,amount), (amount,amount), (0,0)), mode='constant', constant_values=0) # This is for using imshow

    reconstruction_M = np.linalg.pinv(M)

    # TODO: try cv2.WARP_INVERSE_MAP
    reconstructed = cv2.warpPerspective(added_disturbance, reconstruction_M, (frame_width+2*amount, frame_height+2*amount))

    cv2.imshow('Small Frames', np.hstack((orig_with_trap, padded_added_disturbance, reconstructed)))
    the_frames.append(reconstructed)


    ############################################
    # Creating small frame with alt way
    ############################################


    small_orig_with_trap = orig_with_trap



    pts = np.array([pt_a, pt_b, pt_c, pt_d], np.int32)
    print(f'pts {pts}')

    h = 128
    w = 128

    indy, indx = np.indices((h,w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])
    temp_M = np.linalg.pinv(M)
    map_ind = temp_M.dot(lin_homg_ind)

    # print(map_ind.shape)
    # quit()
    map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    print(map_x[0,0])
    print(map_y[0,0])


    dst = cv2.remap(small_orig_with_trap, map_x, map_y, cv2.INTER_LINEAR)

    # quit()

    cv2.imshow('2', dst)


    ############################################
    ############################################




    h_large_start = int((height - 512) / 2)
    w_large_start = int((width - 512) / 2)

    large_orig_with_trap = frame[h_large_start-amount:h_large_start+512+amount, w_large_start-amount:w_large_start+512+amount]


    poly_w_shift = int((512-frame_width)/2)
    poly_h_shift = int((512-frame_height)/2)

    large_orig_with_trap = cv2.rectangle(large_orig_with_trap, (amount+poly_w_shift,amount+poly_h_shift), (frame_width+amount+poly_w_shift, frame_height+amount+poly_h_shift), (255, 0, 0), 1) # create rectangle that is starting point for the polygon that determines morphing

    pts = np.array([pt_a, pt_b, pt_c, pt_d], np.int32)
    # print(pts)
    for i, pt in enumerate(pts):
        pts[i] = pt + [poly_w_shift, poly_h_shift]
    # print(pts)
    # quit()
    pts = pts.reshape((-1, 1, 2))
    large_orig_with_trap = cv2.polylines(img=large_orig_with_trap, pts=[pts], isClosed=True, color=(0,0,255), thickness=1)
    



    h = 512
    w = 512
    indy, indx = np.indices((h, w), dtype=np.float32)
    indy -= (poly_h_shift)
    indx -= (poly_w_shift)

    large_orig_with_trap= cv2.circle(large_orig_with_trap, tuple(pt_a+poly_h_shift), radius=3, color=(0,0,255), thickness=-1)
    large_orig_with_trap= cv2.circle(large_orig_with_trap, tuple(pt_b+poly_h_shift), radius=3, color=(0,0,255), thickness=-1)
    large_orig_with_trap= cv2.circle(large_orig_with_trap, tuple(pt_c+poly_h_shift), radius=3, color=(0,0,255), thickness=-1)
    large_orig_with_trap= cv2.circle(large_orig_with_trap, tuple(pt_d+poly_h_shift), radius=3, color=(0,0,255), thickness=-1)

    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    # warp the coordinates of src to those of true_dst
    temp_M = np.linalg.pinv(M)
    map_ind = temp_M.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
    map_x = map_x.reshape(h, w).astype(np.float32) + poly_w_shift + amount
    map_y = map_y.reshape(h, w).astype(np.float32) + poly_h_shift + amount

    # # remap!
    print()
    print(pt_a)
    print(pt_a+poly_h_shift)
    print()
    print('hi')
    print(map_x[amount+poly_w_shift, amount+poly_h_shift])
    print(map_y[amount+poly_w_shift, amount+poly_h_shift])
    dst = cv2.remap(large_orig_with_trap, map_x, map_y, cv2.INTER_LINEAR)
    print('hi')
    print()

    print(dst[amount+poly_w_shift, amount+poly_h_shift])

    cv2.imshow('asdfas;ldkfjas;ldfj', dst)

    if cv2.waitKey(33) == ord('q'):
        break
    input('enter')



"""