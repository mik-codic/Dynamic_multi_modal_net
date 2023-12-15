import os
import subprocess
import time
from spectrogram_2 import extract
from data_stats import extract_statistics
import cv2
from PIL import Image

def extract_spect(video_path, output_dir, fps=None):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Extract audio
    #print("\nvideo path \n",video_path)
    #audio_command = ['ffmpeg', '-i', video_path, '-vn', '-acodec','copy', f'{output_dir}/audio.mp3']
    audio_command = ["ffmpeg", "-i", video_path, output_dir+"/output.mp3"]
    subprocess.run(audio_command)
    extract(output_dir)
def extract_single_frame(video_path,time_sec,frame_indx,output_dir,use_seconds = False):
    os.makedirs(output_dir, exist_ok=True)
    print(time_sec)
    #convert seconds to format
    ss = time.strftime('%H:%M:%S', time.gmtime(float(time_sec)))
    print(ss)
    if use_seconds:
        frame_command = ['ffmpeg', '-ss',ss,"-i", video_path, "-frames:v",str(1)]
        frame_command += [f'{output_dir}/frame%04d.png']
    else:
        frame_command = ['ffmpeg', '-i', video_path, '-vf', f"select=eq(n\,{frame_indx})", '-vframes', '1']
        frame_command += [f'{output_dir}/frame%04d.png']
    subprocess.run(frame_command)
def cut_spectro(path,frame_indx,total_frames):
    im = Image.open(path)
    
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size
    #print("\nwidth of the spectrogram\n",width)
    bins_per_frame = width/total_frames
    #print("\nbins per frame\n",bins_per_frame)
    center = width/2#(width*frame_indx)/total_frames
    #bins_per_frame_convers = bins_per_frame*width/total_frames
    # Setting the points for cropped image
    left = center-320#bins_per_frame_convers*center-320*bins_per_frame_convers
    top = 0
    right = center+320#bins_per_frame_convers*center+320*bins_per_frame_convers
    bottom = height
    #print("\nehy ecco a te left right b t\n",left,right,bottom,top)
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    
    # Shows the image in image viewer
    im1.show()
    return im1
# Example usage
dir = "/Users/michelepresutto/Desktop/Intership Folder/data_script/data_set/train/"
for file in os.listdir("/Users/michelepresutto/Desktop/Intership Folder/data_script/data_set/train"):
    #print(file)
    #extract_frames_and_audio(dir+file, 'data_set/single_frames_for_video/frame-'+file, fps=30)
    sr,fps,vid_len,resolution = extract_statistics(dir+file, folder=False)
    cap = cv2.VideoCapture(dir+file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(f'Total number of frames: {total_frames}')

    #extract the middle frame from the video 
    
    extract_single_frame(dir+file,vid_len/3,int(round(total_frames/3)),"unpacked_single_frame/"+file,use_seconds = False)
    extract_spect(dir+file, 'unpacked_single_frame/'+file)
    #extract_window_from_spect(dir+file,vid_len/2,2)
    cutted_spect = cut_spectro('unpacked_single_frame/'+file+"/full_spectrogram.png",total_frames/3,total_frames)
    print("\n\nsize of the cutted spectrogram\n\n",(cutted_spect.size))
    cutted_spect.save("unpacked_single_frame/"+file+"/c_spectrogram.png")


#ffmpeg -ss 00:00:04 -i input.mp4 -frames:v 1 screenshot.png
