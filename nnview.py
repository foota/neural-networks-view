#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nnview.py - View from the Neural Networks

This program generates video files visualized by the AlexNet/VGG19 architectures.

  * https://github.com/foota/neural-networks-view

Usage:
  * $ python nnview.py input-video-file output-video-file

References:
  * https://github.com/kkroening/ffmpeg-python
  * https://pytorch.org/hub/pytorch_vision_alexnet/
  * https://pytorch.org/hub/pytorch_vision_vgg/

© 2020 foota
"""

import sys
import subprocess
import argparse
import time
import numpy as np
from PIL import Image

import torch
import torchvision

import ffmpeg

parser = argparse.ArgumentParser(description="View from the Neural Networks")
parser.add_argument("infile", help="Input video filename")
parser.add_argument("outfile", help="Output video filename")
parser.add_argument(
    "-d",
    "--device",
    default="cuda",
    action="store",
    help="Device: [cuda | cpu | auto] (default: auto)",
)
parser.add_argument(
    "-a",
    "--architecture",
    default="alexnet",
    action="store",
    help="Network architecture: [AlexNet | VGG19] (default: AlexNet)",
)
parser.add_argument(
    "-b",
    "--blend",
    action="store",
    type=float,
    default=0.3,
    help="Alpha value of an input image to blend into output streams [0.0-1.0] (default: 0.3)",
)


def process_frame(model, frame):
    im0 = Image.fromarray(frame)
    h, w, _ = frame.shape  # (H, W, C)
    frame = torch.tensor(frame, dtype=torch.float32, device=DEVICE)  # host -> device
    frame = torch.transpose(frame, 1, 2)  # (H, C, W)
    frame = torch.transpose(frame, 0, 1)  # (C, H, W)
    frame = torch.unsqueeze(frame, 0)  # (1, C, H, W)
    frame = model.features(frame)  # (1, NROWS * NCOLS, Ho, Wo)
    # frame = model.avgpool(frame)
    _, _, h_, w_ = frame.shape
    frame = frame.to("cpu").detach().numpy()  # device -> host
    frame = (
        np.squeeze(frame).reshape((NROWS * NCOLS, -1)).transpose((1, 0))
    )  # (Ho * Wo, NROWS * NCOLS)
    frame = (
        (frame - np.min(frame, axis=0))
        / (np.max(frame, axis=0) - np.min(frame, axis=0))
    ) * 255  # min-max scaling (0-255)
    frame = (
        frame.reshape((h_, w_, NROWS, NCOLS))
        .transpose(2, 0, 3, 1)
        .reshape((h_ * NROWS, w_ * NCOLS))
    )  # (NROWS, NCOLS, Ho, Wo) -> (NROWS, Ho, NCOLS, Wo) -> (Ho * NROWS, Wo * NCOLS)
    im = Image.fromarray(frame.astype(np.uint8)).convert("RGB").resize((w, h))
    im = Image.blend(im, im0, ALPHA)
    frame = np.asarray(im)
    return frame


def run(model, infile, outfile):
    probe = ffmpeg.probe(infile)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width, height = int(video_info["width"]), int(video_info["height"])
    proc_in = subprocess.Popen(
        ffmpeg.input(infile)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .compile(),
        stdout=subprocess.PIPE,
    )
    proc_out = subprocess.Popen(
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(width, height)
        )
        .output(outfile, pix_fmt="yuv420p")
        .overwrite_output()
        .compile(),
        stdin=subprocess.PIPE,
    )
    while True:
        inbytes = proc_in.stdout.read(width * height * 3)
        if len(inbytes) == 0:
            break
        inframe = np.frombuffer(inbytes, np.uint8).reshape((height, width, 3))
        outframe = process_frame(model, inframe)
        proc_out.stdin.write(outframe.astype(np.uint8).tobytes())
    proc_in.wait()
    proc_out.stdin.close()
    proc_out.wait()


def main(args):
    global DEVICE, ALPHA, NROWS, NCOLS
    start_time = time.time()
    args = parser.parse_args()
    DEVICE = (
        args.device.lower()
        if args.device.lower() in ("cuda", "cpu")
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ALPHA = (
        args.blend
        if 0.0 <= args.blend <= 1.0
        else print("{} is not range [0.0-1.0], so use ALPHA=0.3".format(args.blend))
        or 0.3
    )
    if args.architecture.lower() == "vgg19":
        NROWS, NCOLS = 16, 32
        model = torchvision.models.vgg19(pretrained=True).to(DEVICE).eval()  # VGG19
    else:  # AlexNet
        NROWS, NCOLS = 16, 16
        model = torchvision.models.alexnet(pretrained=True).to(DEVICE).eval()  # AlexNet
    torch.set_grad_enabled(False)
    run(model, args.infile, args.outfile)
    print(model)
    print("Time (s): {:.3f}".format(time.time() - start_time))


if __name__ == "__main__":
    main(sys.argv)
