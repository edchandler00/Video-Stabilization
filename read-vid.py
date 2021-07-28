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
for i in range(N):
#while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("cannot recieve frame")
        break

    # amount = 10
    # amount = 4
    amount = 3
    orig_with_trap = frame[h_start-amount:h_start+frame_height+amount, w_start-amount:w_start+frame_width+amount].copy() #https://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
    
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

    # TODO: figure out this!!! (pretty much copied)
    orig_with_trap = cv2.rectangle(orig_with_trap, (amount,amount), (frame_width+amount, frame_height+amount), (255, 0, 0), 2) # create rectangle that is starting point for the polygon that determines morphing
    pts = np.array([pt_a, pt_b, pt_c, pt_d], np.int32)
    pts = pts.reshape((-1, 1, 2))
    orig_with_trap = cv2.polylines(img=orig_with_trap, pts=[pts], isClosed=True, color=(0,0,255), thickness=2) # The polygon that will lead to morphing
    
    inside_poly_pts = []
    for pt,amt in zip(pts, [[10,10], [10, -10], [-10,-10], [-10,10]]):
        inside_poly_pts.append(pt+amt)
    inside_poly_pts = np.array(inside_poly_pts, np.int32)
    inside_poly_pts = inside_poly_pts.reshape((-1,1,2))
    orig_with_trap = cv2.polylines(img=orig_with_trap, pts=[inside_poly_pts], isClosed=True, color=(23,230,120), thickness=2) # The polygon that will lead to morphing
    

    M_input_pts = np.float32([pt_a, pt_b, pt_c, pt_d])
    M_output_pts = np.float32([[0, 0],
                            [0, frame_height - 1],
                            [frame_width - 1, frame_height - 1],
                            [frame_width - 1, 0]])

    M = cv2.getPerspectiveTransform(M_input_pts, M_output_pts)
    
    added_disturbance = cv2.warpPerspective(orig_with_trap, M, (frame_width, frame_height))
    cv2.imshow('added_disturbance small', added_disturbance)
    small_disturbance_frames.append(added_disturbance)

    padded_added_disturbance = np.pad(added_disturbance, ((amount,amount), (amount,amount), (0,0)), mode='constant', constant_values=0) # This is for using imshow

    reconstruction_M = np.linalg.pinv(M)

    # TODO: try cv2.WARP_INVERSE_MAP
    reconstructed = cv2.warpPerspective(added_disturbance, reconstruction_M, (frame_width+2*amount, frame_height+2*amount))

    cv2.imshow('asdf', np.hstack((orig_with_trap, padded_added_disturbance, reconstructed)))
    the_frames.append(reconstructed)



    # shape (3,4); last axis in order TL, BL, BR, TR
    temp_corners = np.matmul(reconstruction_M, np.array([[0,0,1], [0,frame_height-1,1], [frame_width-1,frame_height-1,1], [frame_width-1,0,1]]).T)
    temp_corners /= temp_corners[-1]

    reconstructed_t_pos.append(np.max([temp_corners[1,0], temp_corners[1,3]]))
    reconstructed_l_pos.append(np.max([temp_corners[0,0], temp_corners[0,1]]))
    reconstructed_b_pos.append(np.min([temp_corners[1,1], temp_corners[1,2]]))
    reconstructed_r_pos.append(np.min([temp_corners[0,2], temp_corners[0,3]]))

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

    # draw rectangle that is starting point for the polygon that determines morphing
    large_orig_with_trap = cv2.rectangle(large_orig_with_trap, (amount+poly_w_shift,amount+poly_h_shift), (frame_width+amount+poly_w_shift, frame_height+amount+poly_h_shift), (255, 0, 0), 2)
    # draw polygon that determines morphing
    large_orig_with_trap = cv2.polylines(img=large_orig_with_trap, pts=[pts], isClosed=True, color=(0,0,255), thickness=2)
    
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
    






for small_disturbance_frame, large_disturbance_frame in zip(small_disturbance_frames, large_disturbance_frames):
    cv2.imshow('diturb small', small_disturbance_frame)
    cv2.imshow('disturb large', large_disturbance_frame)

    if cv2.waitKey(33) == ord('q'):
        break


    
# FIXME: should i use rint instead???
final_t_pos = np.max(reconstructed_t_pos).astype(np.int32)
final_l_pos = np.max(reconstructed_l_pos).astype(np.int32)
final_b_pos = np.min(reconstructed_b_pos).astype(np.int32)
final_r_pos = np.min(reconstructed_r_pos).astype(np.int32)

print(f'final pos: {final_t_pos}, {final_l_pos}, {final_b_pos}, {final_r_pos}')
print(len(the_frames))
print(the_frames[0].shape)
# for frame, small_frame in zip(the_frames, the_small_frames):
for frame in the_frames:
    cv2.imshow('reconstructed', frame[ final_t_pos:final_b_pos, final_l_pos:final_r_pos])
    # cv2.imshow('small reconstructed', small_frame[ final_t_pos+125:final_b_pos-125, final_l_pos+125:final_r_pos-125])
    if cv2.waitKey(33) == ord('q'):
        break
    # input('enters')
    
cap.release()
# out.release()
cv2.destroyAllWindows()


fig, (ax1, ax2) = plt.subplots(2,1)
ax1.boxplot([reconstructed_t_pos, reconstructed_l_pos])
# ax1.set_xticks(['a','b'])
ax2.boxplot([reconstructed_b_pos, reconstructed_r_pos])
# ax2.xticks([1,2],['bottom','right'])

plt.show()


#TODO: fix M that cut off too much








# for pt in [pt_a, pt_b, pt_c, pt_d]:
#     print(pt)
#     temp_M = M
#     temp = np.matmul(temp_M, np.array([pt[0], pt[1], 1]).T)
#     temp /= temp[-1]
#     print(f'Diff between {pt} and {np.rint(temp[:-1]).astype(np.int32)}: {np.rint(pt-temp[:2])}')
#     print()

# print('\nNext\n')
# for pt in pts:
#     pt = pt[0]
#     print(pt)
#     # temp_M = np.linalg.pinv(M)
#     temp_M = M
#     # temp = np.matmul(temp_M, np.array([pt[0], pt[1], 1]).T-poly_h_shift) 
#     the_index = np.array([pt[0]-poly_h_shift, pt[1]-poly_h_shift, 1])
#     print(f'the index: {the_index}')
#     temp = temp_M.dot(the_index)
#     temp /= temp[-1]
#     temp += poly_h_shift
#     print(f'Diff between {pt} and {np.rint(temp[:-1]).astype(np.int32)}: {np.rint(pt-temp[:2])}')
#     print()









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








'''
# This is to see what happens when using same M on smaller frame w/same center
# make this 270x270 (1/2) w/padding
# orig_with_trap is 520x520 (b/c of amount padding)
h_crop_start = int((orig_with_trap.shape[0] - 2*amount - 250)/2)
w_crop_start = int((orig_with_trap.shape[1] - 2*amount - 250)/2)

small_orig_with_trap = orig_with_trap[h_crop_start-amount:h_crop_start+250+amount, w_crop_start-amount:w_crop_start+250+amount]
small_orig_with_trap = cv2.rectangle(small_orig_with_trap, (amount,amount), (250+amount, 250+amount), (255,0,0),1)
small_orig_with_trap = cv2.rectangle(small_orig_with_trap, (2*amount,2*amount), (250, 250), (0,255,0),1)
small_orig_with_trap = np.pad(small_orig_with_trap, ((125,125), (125,125),(0,0)), mode="constant", constant_values=0)


small_added_disturbance = added_disturbance[h_crop_start:h_crop_start+250, w_crop_start:w_crop_start+250]
small_added_disturbance = np.pad(small_added_disturbance, ((125,125), (125,125),(0,0)), mode="constant", constant_values=0) # need to add padding because of how matrix works

# small_reconstructed = cv2.warpPerspective(small_added_disturbance, reconstruction_M, (250+2*amount, 250+2*amount))
small_reconstructed = cv2.warpPerspective(small_added_disturbance, reconstruction_M, (frame_width+2*amount, frame_height+2*amount))
the_small_frames.append(small_reconstructed)
'''





'''
    # input(f'{pts[0,0,0]}')
    indices = np.indices((512,512)) - poly_w_shift# TODO: make so works for rectangle
    indices = np.concatenate([indices, np.ones((1,512,512))])
    # indices = np.swapaxes(np.swapaxes(indices, 0,2), 0,1)
    # print(f'indices shape: {indices.shape}')
    # print(f'pts[0,0]: {pts[0,0]}')
    # print(f'before reshape {indices[:,pts[0,0,0], pts[0,0,1]]}')
    # indices = indices.reshape((512*512,3))
    indices = indices.reshape((3,512*512))

    # print(f'indices shape {indices.shape}')
    # print(f'after reshape {indices[:,pts[0,0,0]*pts[0,0,1]]}')


    print(indices.reshape((3,512,512))[:,pts[0,0,0], pts[0,0,1]])


    # new_indices = np.matmul(M, indices)
    new_indices = M.dot(indices)
    # new_indices = np.matmul(M, indices.T)
    new_indices /= new_indices[-1]
    # new_indices = new_indices[:-1]
    new_indices = new_indices.reshape((3,512,512))
    new_indices += poly_w_shift # TODO:make work for diff sized h,w

    # print(f'new_indices shape {new_indices.shape}')
    print(f'new_indices {new_indices[:,pts[0,0,0],pts[0,0,1]]} {new_indices[:,pts[1,0,0],pts[1,0,1]]}')
    print(f'new_indices {new_indices[:,pts[2,0,0],pts[2,0,1]]} {new_indices[:,pts[3,0,0],pts[3,0,1]]}')


    # TODO: fix this
    map_x, map_y = new_indices[:-1]
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    # print(map_x.shape)
    # print(map_x.dtype)
    # print()
    # print(map_y.shape)
    # temp = cv2.remap(large_orig_with_trap, map_x, map_y, cv2.INTER_LINEAR)
    # temp = cv2.remap(large_orig_with_trap, map_x, map_y, cv2.INTER_LINEAR)
    temp = cv2.remap(large_orig_with_trap, map_y, map_x, cv2.INTER_LINEAR)

    cv2.imshow('???', temp)
    # print(f'{new_indices[pts[0,0,0],pts[0,0,1]]}')
    # print(indices[:,pts[0,0,0]*pts[0,0,1]])
    print()
    '''









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