import json
import base64
import tempfile
from moviepy.editor import VideoFileClip
from skimage import color
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

def load_video_from_json(json_data):
    try:
        logging.info("Received JSON data: %s", json_data[:100])  # İlk 100 karakteri loglayalım
        video_base64 = json_data
        logging.info("Extracted base64 video data: %s", video_base64[:50])  # İlk 50 karakteri loglayalım
        video_data = base64.b64decode(video_base64)
    except Exception as e:
        logging.error("Failed to decode video data: %s", str(e))
        raise ValueError(f"Failed to decode video data: {str(e)}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(video_data)
            temp_video_path = temp_video_file.name
    except Exception as e:
        logging.error("Failed to write video data to temp file: %s", str(e))
        raise IOError(f"Failed to write video data to temp file: {str(e)}")

    return temp_video_path, 'mp4'

def resize_img(img, HW=(256, 256), resample=Image.BICUBIC):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256, 256), resample=Image.BICUBIC):
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)
    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]
    tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]
    return tens_orig_l, tens_rs_l

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]
    if HW_orig != HW:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
    else:
        out_ab_orig = out_ab
    out_ab_orig = torch.clamp(out_ab_orig, -128, 128)
    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    out_lab_orig_np = out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0))
    out_lab_orig_np[..., 0] = np.clip(out_lab_orig_np[..., 0], 0, 100)
    out_lab_orig_np[..., 1:] = np.clip(out_lab_orig_np[..., 1:], -127, 127)
    return color.lab2rgb(out_lab_orig_np)

def process_frame(frame, model, use_gpu):
    img = np.array(frame)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    if use_gpu:
        tens_l_rs = tens_l_rs.cuda()
    out_ab = model(tens_l_rs).cpu()
    out_img = postprocess_tens(tens_l_orig, out_ab)
    return (out_img * 255).astype(np.uint8)

def video_to_base64(video_path, format='mp4'):
    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()
    return base64.b64encode(video_data).decode('utf-8')
