#get relevant parts
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

ffmpeg_extract_subclip("./videos/trump_original.mp4", 360, 475, targetname="./videos/trump_cut/trump_360_475.mp4")
ffmpeg_extract_subclip("./videos/trump_original.mp4", 482, 750, targetname="./videos/trump_cut/trump_482_750.mp4")
ffmpeg_extract_subclip("./videos/trump_original.mp4", 960, 1040, targetname="./videos/trump_cut/trump_960_1040.mp4")


#transform video to frames
import cv2
import os
rootdir = './videos/trump_cut'
count = 0

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)        
        if path.endswith('.mp4'):
            print path
            vidcap = cv2.VideoCapture(path)
            success = true
            while success:
              success,image = vidcap.read()
              print('Read a new frame: ', success)
              cv2.imwrite("frame%d.png" % count, image)     # save frame as JPEG file
              count += 1
              
count = 0     
vidcap = cv2.VideoCapture('./videos/trump_cut/trump_960_1040.mp4')


success,image = vidcap.read()
print('Read a new frame: ', success)
if success:
  cv2.imwrite("./videos/trump_imgs/frame%d.png" % count, image)     # save frame as JPEG file
  count += 1
