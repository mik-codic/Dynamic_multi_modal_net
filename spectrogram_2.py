import torch
import torch.nn as nn
import torchaudio 
import torchlibrosa as tl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_stats import extract_statistics
        
def extract(path):
    # Load the image
    audio, sr = torchaudio.load(path+"/output.mp3")
    print("audio and sample_rate",audio,sr)
    #sr,fps,vid_len,resolution = extract_statistics(path,folder=False)
    scale = 0.5
    sample_rate = int(44100 * scale)
    n_fft = 640
    hop_size = int(320 * scale)
    mel_bins = 320
    window_size = int(1024 * scale)
    fmin = 50
    fmax = 14000
    duration = 5
    crop_len = 5 * sample_rate
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    # Compute the spectrogram
    spec = tl.Spectrogram(n_fft=n_fft, hop_length=hop_size, 
        win_length=window_size, center=True, pad_mode='reflect', 
        power=2.0)


    logmel_extractor = tl.LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
        n_mels=mel_bins, fmin=fmin, fmax=fmax,ref=ref, amin=amin, top_db=top_db,  
        freeze_parameters=True)

    bn0 = nn.BatchNorm2d(mel_bins)
    feature_extractor = nn.Sequential(
        spec
        #logmel_extractor
    )
    a = feature_extractor(audio)
    a = a.log()
    #a = a.detach().numpy()
    
    
    # a = a.transpose(1, 3)
    # a = bn0(a)
    # a = a.transpose(1, 3)
    # print(a.size())
    
    # Display the spectrogram    
    print("\nspectrum shape:\n",a.shape)
    #plt.imsave(path+"/spectrogram.png",a)
    print("\nfirst image spect\n",len(a[0]))
    a = np.squeeze(a.detach().numpy())
    a = a.transpose(2,1,0)
    pad = np.zeros_like(a)
    a = np.concatenate((a, pad[...,0:1]),axis=2)
    print(a.shape)
    print(np.max(a),np.min(a))
    a_norm = (a-np.min(a))/(np.max(a)-np.min(a))
    a_norm = (a_norm*255).astype(np.uint8)
    a_norm = a_norm[:-1]
    im = Image.fromarray(a_norm)
    
    im.save(path+"/spectrogram.png")
    #exit()
    # fig = plt.figure(figsize=(5, 5),frameon=False)

    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)   
    # ax.imshow(a[0].T, aspect='auto')
    # fig.savefig(path+"/spectrogram.png")

    # plt.imshow(a[0].T, origin='lower', aspect='auto')
    # print("\nshape of a\n",a.shape)
    # plt.savefig(path+"/spectrogram.png")
    # #plt.colorbar(format='%+2.0f dB')
    # #plt.title('Spectrogram')
    # plt.show()

# Example usage data_set/train/A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A.mp4
#extract("/Users/michelepresutto/Desktop/Intership Folder/data_script/data_set/train/A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A.mp4")
path="/Users/michelepresutto/Desktop/Intership Folder/data_script/data_set/unpacked/frames-A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A.mp4/audio.wav"
path3 = "/Users/michelepresutto/Desktop/Intership Folder/data_script/code/Audio-Bus256.wav"
pat_2 = "/Users/michelepresutto/Desktop/Intership Folder/data_script/data_set/unpacked/frames-v=cO1UefhG7AY__#1_label_G-0-0.mp4/audio.wav"
#audio, sr = torchaudio.load(path)
#extract(path3)
