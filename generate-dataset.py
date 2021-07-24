import cv2
import numpy as np
from util import get_added_frequencies, get_frame_shift, get_video_info, get_crop_pos
import matplotlib.pyplot as plt
import argparse
import json
import glob
from tqdm import tqdm
import os

print(cv2.__version__)
# quit()

FILE_TYPES = ("mov", "mp4", "avi")

def generate_dataset(N, frame_height, frame_width, num_data_points, input_directory, output_directory):

    # os.chdir(input_directory) #TODO: should i use this or not

    # TODO: parallelize
    for file_type in FILE_TYPES:
        print(f'Working on all {file_type} files')
        
        print(glob.glob(f'{input_directory}*.{file_type}'))
        for f in tqdm(glob.iglob(f'{input_directory}*.{file_type}')):
        # for f in glob.iglob(f'{input_directory}*.{file_type}'):
            # print(f)

            cap = cv2.VideoCapture(f)
            # TODO: num_frames is known not to be accurate, figure out best way!!
            num_frames, fps, height, width = get_video_info(cap) #TODO: check frame size order

            for data_point_num in range(num_data_points):
                # TODO: deal w/ num_frames thing above to fix
                # start_frame_idx = np.random.randint(num_frames-N)
                start_frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame_idx)

                file_name = "output"
                # TODO: check frame size order
                # FIXME: set 30 to a variable
                out = cv2.VideoWriter(f'{output_directory}{file_name}_{data_point_num}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))

                X = get_added_frequencies(N)

                w_shift, h_shift = get_frame_shift(X)
                # TODO: only return if pixel values are different enough
                w_start,h_start = get_crop_pos(width, frame_width, height, frame_height)

                # start_frame_idx = np.random.randint((num_frames-N))
                
                for i in range(N):
                    ret,frame = cap.read()
                    if not ret:
                        print("cannot recieve frame")
                        break

                    w, h = w_start + w_shift[i,0], h_start + h_shift[i,0]
                    corrected_frame = frame[h:h+frame_height,w:w+frame_width]
                    # cv2.imshow('frame', frame[h:h+frame_height,w:w+frame_width])
                    # cv2.imshow('frame', corrected_frame)
                    out.write(corrected_frame)
                    if cv2.waitKey(33) == ord('q'):
                        break

            cap.release()
            out.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--FPS", help="")
    parser.add_argument("-i", "--Input", required=True, help="<Required> Input directory of video samples.")
    parser.add_argument("-o", "--Output", required=True, help="<Required> Output directory of generated data points.")
    parser.add_argument("-n", "--NumFrames", type=int, help="The number of frames of the video to generate. Default is 120.")
    parser.add_argument("-s", "--FrameSize", nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'), help="The size of the generated frame. Input two positive integers as height width. Default is 64 64")
    parser.add_argument("-e", "--NumDataPoints", type=int, help="The number of generated added motion data points for each video. Default is 10.")
    # parser.add_argument("")

    args = parser.parse_args()

    input_directory = args.Input
    output_directory = args.Output
    N = args.NumFrames if args.NumFrames else 120
    frame_height, frame_width = args.FrameSize if args.FrameSize else [64,64]
    num_data_points = args.NumDataPoints if args.NumDataPoints else 10

    print("hi")
    print(args)
    print(N)
    print(frame_height, frame_width)
    print(num_data_points)

    # quit()

    generate_dataset(N=N, frame_height=frame_height, frame_width=frame_width, num_data_points=num_data_points, input_directory=input_directory, output_directory=output_directory)

    # TODO: save arguments as json



# fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)

# ax1.stem(np.arange(N), np.real(X))
# ax1.set_title("Real Components")

# ax2.stem(np.arange(N), np.imag(X))
# ax2.set_title("Imag Components")

# ax3.plot(w_shift)
# ax3.set_title("w_shift")

# ax4.plot(h_shift)
# ax4.set_title("h_shift");

# plt.show();