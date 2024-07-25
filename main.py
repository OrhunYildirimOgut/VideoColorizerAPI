#main.py
import torch
import logging
from eccv16 import eccv16
from siggraph17 import siggraph17
from util import process_frame
from moviepy.editor import VideoFileClip

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU usage
use_gpu = torch.cuda.is_available()

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if use_gpu:
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

def colorize_and_add_audio(video_path, output_path, model_name):
    model = colorizer_siggraph17 if model_name == 'siggraph17' else colorizer_eccv16

    original_clip = VideoFileClip(video_path)
    audio = original_clip.audio
    processed_clip = original_clip.fl_image(lambda frame: process_frame(frame, model, use_gpu))
    processed_clip = processed_clip.set_audio(audio)
    processed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')


