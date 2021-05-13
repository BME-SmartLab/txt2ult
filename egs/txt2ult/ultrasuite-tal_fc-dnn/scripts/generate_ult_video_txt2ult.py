## txt2ult

import sys
import os

if len(sys.argv) != 2:
    print('usage: python3 generate_ult_video.py <speaker>')
    print('Number of arguments:', len(sys.argv))
    print('Argument List:', str(sys.argv))
    speaker = '01fi'
else:
    speaker = sys.argv[1]

import matplotlib.pyplot as plt
import numpy as np

import pickle

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

from subprocess import run

# requirement: ultrasuite-tools, https://github.com/UltraSuite/ultrasuite-tools
sys.path.append('~/ultrasuite-tools/')
from ustools.read_core_files import parse_parameter_file,read_ultrasound_file
from ustools.transform_ultrasound import transform_ultrasound

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, IncrementalPCA

import skimage


# ult2vid converts raw ultrasound data to .mp4 video
# the video is extracted in the conventional 'ultrasound wedge' orientation
def ult2vid_wedge_ustools(ult_data, dir_file, filename_no_ext, NumVectors, PixPerVector, ZeroOffset, Angle, FramesPerSec, pixels_per_mm=2):
    print(filename_no_ext + ' - ultrasound video started')
    
    output_file_no_ext = dir_file + filename_no_ext + '_ultrasound_ustools'
    
    # sample from https://github.com/UltraSuite/ultrasuite-tools/blob/master/ultrasound_process.ipynb
    ult_data_wedge = transform_ultrasound(ult_data, \
        spline_interpolation_order=2, background_colour=255, \
        num_scanlines=NumVectors, size_scanline=PixPerVector, \
        angle=Angle, zero_offset=ZeroOffset, pixels_per_mm=pixels_per_mm)
    
    print(filename_no_ext + ' - ultrasound video to wedge conversion (ustools) finished', ult_data_wedge.shape)
    
    (n_frames, n_width, n_height) = ult_data_wedge.shape
    
    # compressed
    # fourcc = VideoWriter_fourcc(*'MP4V')
    fourcc = VideoWriter_fourcc(*'mp4v')
    
    # uncompressed 8-bit
    # fourcc = VideoWriter_fourcc(*'Y800')
    
    # video = VideoWriter(output_file_no_ext + '.avi', fourcc, float(FramesPerSec), (n_width, n_height), 0)
    video = VideoWriter(output_file_no_ext + '.mp4', fourcc, float(FramesPerSec), (n_width, n_height), 0)
    
    for n in range(n_frames):
        # print('starting frame ', n)
        # print(filename_no_ext, 'frame ', n, ' ', 'minmax', np.min(ult_data_wedge[n]), np.max(ult_data_wedge[n]), end='\n')
        frame = np.uint8(ult_data_wedge[n]).reshape(n_width, n_height)
        frame = np.rot90(frame).reshape(n_height, n_width, 1)
        
        video.write(frame)
        print('frame ', n, ' done', end='\r')
        # print('frame ', n, ' done')
    
    video.release()
    
    print(filename_no_ext + ' - ultrasound video finished')


# ult2vid converts raw ULT data to .mp4 video
def ult2vid(ult_data, dir_file, filename_no_ext, n_width, n_height, FramesPerSec):
    
    print(filename_no_ext + ' - ULT video started')
    
    output_file_no_ext = dir_file + filename_no_ext
    n_frames = len(ult_data)
    
    # compressed
    # fourcc = VideoWriter_fourcc(*'MP4V')
    
    # uncompressed 8-bit
    fourcc = VideoWriter_fourcc(*'Y800')
    video = VideoWriter(output_file_no_ext + '.avi', fourcc, float(FramesPerSec), (n_width, n_height), 0)
    # video = VideoWriter(output_file_no_ext + '.mp4', fourcc, float(FramesPerSec), (n_width, n_height), 0)
    
    for n in range(n_frames):
        frame = np.rot90(np.uint8(ult_data[n]).reshape(n_width, n_height, 1))
        video.write(frame)
        print('frame ', n, ' done', end='\r')
        
    video.release()
    
    print(filename_no_ext + ' - ULT video finished, n_frames: ', n_frames)


# vidwav2demo adds ultrasound video (left & right) and speech together
# resulting in synchronized ultrasound + speech
# requirement: ffmpeg
def vidwav2demo_combined(in_mp4_file_left, in_mp4_file_right, in_wav_file, out_mp4_file, fps): 
    # original video: 948 x 578
    # -filter:v "crop=out_w:out_h:x:y", x and y specify the top left corner of the output rectangle
    command = 'ffmpeg ' + \
        '-i ' + in_mp4_file_left + ' ' + \
        '-i ' + in_mp4_file_right + ' ' + \
        '-i ' + in_wav_file + ' ' + \
        '-filter_complex "' + \
        'nullsrc=size=1896x578 [base]; ' + \
        '[0:v] crop=948:578:0:0, setpts=PTS-STARTPTS, scale=948x578 [left]; ' + \
        '[1:v] crop=948:578:0:0, setpts=PTS-STARTPTS, scale=948x578 [right]; ' + \
        '[base][left] overlay=shortest=1 [tmp1]; ' + \
        '[tmp1][right] overlay=shortest=1:x=948:y=0' + '" ' + \
        '-c:v libx264 -r ' + str(fps) + ' ' + \
        '-strict -2 ' + \
        '-y ' + out_mp4_file
        
    # print(command)
    run(command, shell=True)

### main part

# parameters of ultrasound images, from .param file
n_lines = 64
n_pixels = 842
zero_offset = 210
angle = 0.025
fps = 200 # for 5 ms hop size
n_pixels_reduced = 128
n_pca = 128

# speaker from command line
# speaker = '02fe'

dir_gen = 'experiments/' + speaker + '/test_synthesis/wav/'

ult_pca = pickle.load(open(speaker + '_data/UTI_to_PCA_' + speaker + '.sav', 'rb'))
# ult_pca = pickle.load(open('01fi_data/UTI_to_PCA_01fi.sav', 'rb'))

ult_files_gen = []
if os.path.isdir(dir_gen):
    for file in sorted(os.listdir(dir_gen)):
        if file.endswith('.ultpca128'):
            ult_files_gen += [dir_gen + file[:-10]]


for basefilename in ult_files_gen:

    ultpca128 = np.fromfile(basefilename + '.ultpca128', dtype=np.float32).reshape(-1, n_pca)

    ult_data_reduced = ult_pca.inverse_transform(ultpca128).reshape(-1, n_lines, n_pixels_reduced)

    print('before normalization', np.min(ult_data_reduced), np.max(ult_data_reduced))
    ult_data_reduced -= np.min(ult_data_reduced)
    ult_data_reduced /= np.max(ult_data_reduced)
    ult_data_reduced *= 255
    print('after normalization', np.min(ult_data_reduced), np.max(ult_data_reduced))

    ult2vid(ult_data_reduced, '', \
        basefilename, n_lines, n_pixels_reduced, fps)

    # resize
    ult_data_full = np.zeros((len(ult_data_reduced), n_lines, n_pixels))
    for i in range(len(ult_data_full)):
        ult_data_full[i] = skimage.transform.resize(ult_data_reduced[i], (n_lines, n_pixels), preserve_range=True)

    ult2vid_wedge_ustools(ult_data_full, '', \
        basefilename, n_lines, n_pixels, zero_offset, angle, fps, 2)

for i in range(len(ult_files_gen)):
    
    vidfile_ref = ult_files_gen[i] + '_ultrasound_ustools.mp4'
    vidfile_gen = ult_files_gen[i] + '_ultrasound_ustools.mp4'
    wavfile_gen = ult_files_gen[i] + '.wav'
    vidfile_out = ult_files_gen[i] + '_double_ultrasound_ustools.mp4'
    
    print('generating double video for', vidfile_gen)
    
    vidwav2demo_combined(vidfile_ref, vidfile_gen, wavfile_gen, vidfile_out, fps)
