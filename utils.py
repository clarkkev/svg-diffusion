"""Utility functions."""
import io
import json
import math
import os
import pickle

import cairosvg
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


def show_images(images, titles=None, cols=4, figsize=(16, 8)):
  if isinstance(images, torch.Tensor):
    images = tensor_to_image(images)
  if not isinstance(images, list):
    images = [images]
  if isinstance(images[0], str):
    images = [svg_to_image(image) for image in images]

  plt.figure(figsize=figsize)
  rows = math.ceil(len(images) / cols)
  for i, image in enumerate(images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(image)
    if titles and i < len(titles):
        plt.title(titles[i])
    plt.axis("off")
  plt.tight_layout()
  plt.show()


def svg_to_image(svg):
  if isinstance(svg, str):
    svg = svg.encode('utf-8')
  if isinstance(svg, bytes):
    png_bytes = cairosvg.svg2png(bytestring=svg)
    return Image.open(io.BytesIO(png_bytes)).convert('RGB')
  else:
    return [svg_to_image(s) for s in svg]


def svg_to_tensor(svg, dtype=torch.bfloat16):
  if isinstance(svg, str) or isinstance(svg, bytes):
    return transforms.ToTensor()(svg_to_image(svg)).to("cuda", dtype=dtype)
  else:
    return torch.stack([svg_to_tensor(s) for s in svg])


def tensor_to_image(tensor):
  if tensor.ndim == 4:
    return [to_pil_image(img_tensor.to(torch.float32)) for img_tensor in tensor]
  return to_pil_image(tensor.to(torch.float32))


def image_to_tensor(image, dtype=torch.bfloat16):
  transform = transforms.ToTensor()
  if isinstance(image, list):
    tensor = torch.stack([transform(img) for img in image])
  else:
    tensor = transform(image)
  return tensor.to(device="cuda", dtype=dtype)


def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


def load_pickle(path):
  with open(path, "rb") as f:
    return pickle.load(f)


def write_pickle(o, path):
  if "/" in path:
    mkdir(path.rsplit("/", 1)[0])
  with open(path, "wb") as f:
    pickle.dump(o, f, -1)


def load_json(path):
  with open(path, "r") as f:
    return json.load(f)


def write_json(o, path):
  if "/" in path:
    mkdir(path.rsplit("/", 1)[0])
  with open(path, "w") as f:
    json.dump(o, f, indent=2)


def print_max_memory():
  print(f"Peak allocated memory: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")


def print_allocated_memory():
  print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")


def print_cached_memory():
  print(f"Cached memory: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
