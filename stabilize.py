import cv2
import numpy as np
from util import get_frame_shift, get_video_info, get_crop_pos

cap = cv2.VideoCapture('Videos/shred.mov')

num_frames, fps, full_height, full_width = get_video_info(cap)

save_vid = True
show_vid = True
num_crops = 1
N = 120
model_frame_width, model_frame_height = 64,64 # TODO: figure out best size (hyperparameter)
output_w_pad, output_h_pad = 40, 40
output_frame_width, output_frame_height = full_width - 2*output_w_pad, full_height - 2*output_h_pad

# TODO: only return if pixel values are different enough
model_w_starts,model_h_starts = np.empty(num_crops), np.empty(num_crops) # TODO: check np empty syntax
for i in range(num_crops):
    model_w_starts[i], model_h_starts[i] = get_crop_pos(full_width, model_frame_width, full_height, model_frame_height)

frames_array = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("cannot receive frame")
        break
    frames_array.append(frame)
frames_array = np.array(frames_array)

# Split vid_frames into patches of N
split_locs = [N*i for i in range(1, int(num_frames/N))]
vid_patches = np.split(frames_array, split_locs)

# Check and fix the last patch (overlap with prev) and save where patch actually starts
final_patch_start = 0
if len(vid_patches[-1]) < N:
    final_patch_start = -len(vid_patches[-1])
    vid_patches[-1] = frames_array[-N:]

vid_patches = np.array(vid_patches)


# Run through model
corrected_patches = []
for vid_patch in vid_patches:
    all_X = []
    # Get correction function for all crops
    for model_w_start, model_h_start in zip(model_w_starts, model_h_starts):
        model_vid_patch = vid_patch[model_h_start:model_h_start+model_frame_height, model_w_start:model_w_start+model_frame_width]

        all_X.append([0]) # TODO: run vid_patch through model!!
    
    # Get average correction function
    X = np.avg(all_X) # TODO: check correctly done

    w_shift, h_shift = get_frame_shift(X)

    corrected_patch = []

    for i,frame in enumerate(vid_patch):
        w, h = output_w_pad + w_shift[i,0], output_h_pad + h_shift[i,0]

        # TODO: figure out if a better way to cut off!
        # Cut off if predicted too far left or up
        w = 0 if w < 0 else w
        h = 0 if w < 0 else h
        # Cut off if predicted too far right or down
        w = 2*output_w_pad if w > 2*output_w_pad else w
        h = 2*output_h_pad if w > 2*output_h_pad else h

        corrected_frame = frame[h:h+output_frame_height,w:w+output_frame_width]
        corrected_patch.append(corrected_frame)

    corrected_patches.append(np.array(corrected_patch))
    
# Start final patch at correct frame
corrected_patches[-1] = corrected_patches[-1][final_patch_start:]

# Stitch together patches
corrected_vid = np.concatenate(corrected_patches, axis=0) # TODO: check with real video that this is correct

if save_vid:
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (output_frame_width, output_frame_height))

for frame in corrected_vid:
    if save_vid:
        out.write(corrected_frame)

    if show_vid:
        cv2.imshow('corrected frame', frame)
        if cv2.waitKey(33) == ord('q'):
            break

cap.release()
if save_vid:
    out.release()
cv2.destroyAllWindows()