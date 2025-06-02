"""Reward functions that score generated images."""
import functools

import clip
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel

import utils


def clip_preprocess_torch(image, n_px):
  image = F.interpolate(image, size=(n_px, n_px), mode='bicubic',
                        align_corners=False, antialias=True)
  mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda')
  std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda')
  return (image - mean[:, None, None]) / std[:, None, None]


class AestheticPredictor(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.input_size = input_size
    self.layers = nn.Sequential(
        nn.Linear(self.input_size, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    )

  def forward(self, x):
    return self.layers(x)


def aesthetic_reward(aesthetic_evaluator=None):
  if aesthetic_evaluator is not None:
    return lambda _, image: aesthetic_evaluator.score_differentiable(image)

  model, _ = clip.load('ViT-L/14', device='cuda')
  model.requires_grad_(False)
  state_dict = torch.load('sac+logos+ava1-l14-linearMSE.pth')
  predictor = AestheticPredictor(768)
  predictor.load_state_dict(state_dict)
  predictor.eval()
  predictor.requires_grad_(False)
  predictor.to('cuda', dtype=torch.float16)

  def reward_fn(_, image):
    image = clip_preprocess_torch(image, model.visual.input_resolution)
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return predictor(image_features).mean()

  return reward_fn


def clip_like_reward(model, processor):
  def reward_fn(prompts, images):
    image_embs = model.get_image_features(pixel_values=clip_preprocess_torch(images, 224).to("cuda"))
    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

    text_inputs = processor(
          text=prompts,
          padding=True,
          truncation=True,
          max_length=77,
          return_tensors="pt").to("cuda")
    text_embs = model.get_text_features(**text_inputs)
    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    return (image_embs * text_embs).sum(-1).mean()
  return reward_fn


def hpsv2_reward():
  hpsv2_model = CLIPModel.from_pretrained("adams-story/HPSv2-hf").eval().requires_grad_(False).to("cuda")
  hpsv2_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  return clip_like_reward(hpsv2_model, hpsv2_processor)


def pickscore_reward():
  pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().requires_grad_(False).to("cuda")
  pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
  return clip_like_reward(pickscore_model, pickscore_processor)


@functools.lru_cache(maxsize=1)
def load_siglip():
  print("Loading SigLIP2...")
  model = AutoModel.from_pretrained("google/siglip2-large-patch16-384")
  processor = AutoProcessor.from_pretrained("google/siglip2-large-patch16-384")
  model = model.to("cuda", dtype=torch.bfloat16)
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  return model, processor


def siglip_reward(model=None, processor=None):
  if model is None or processor is None:
    model, processor = load_siglip()

  def reward_fn(prompts, images, no_sigmoid_weight=0):
    tokens = processor.tokenizer(
      prompts, return_tensors='pt', padding='max_length',
      max_length=64)['input_ids'].to('cuda')
    outputs = model(
        input_ids=tokens, pixel_values=(2 * images - 1))
    score = torch.sigmoid(torch.diagonal(outputs.logits_per_image)).mean()
    if no_sigmoid_weight > 0:
      score += no_sigmoid_weight * torch.diagdonal(outputs.logits_per_image).mean()
    return score

  return reward_fn


def tv_reward(diag_weight=0):
  def reward_fn(_, x):
    dx = torch.abs(x[..., 1:] - x[..., :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    loss = dx.mean() + dy.mean()

    if diag_weight > 0:
      diag_dx = torch.abs(x[..., 1:, 1:] - x[..., :-1, :-1])
      diag_dy = torch.abs(x[..., 1:, :-1] - x[..., :-1, 1:])
      loss += diag_weight * (diag_dx.mean() + diag_dy.mean())

    return -loss
  return reward_fn


def lv_reward(k=3):
  def reward_fn(_, x):
    assert k % 2 != 0 and k >= 3
    pad = k // 2
    mean = F.avg_pool2d(x, k, stride=1, padding=pad)
    mean_sq = F.avg_pool2d(x * x, k, stride=1, padding=pad)
    var = mean_sq - mean * mean
    return -var.mean()
  return reward_fn


def preprocess_paligemma(x):
  x = 2 * x - 1
  return F.interpolate(
      x, size=(448, 448), mode='bicubic', align_corners=False, antialias=True)


def paligemma_captioning_reward(vqa):
  def reward_fn(prompts, images):
    inputs = vqa.processor(
      images=utils.tensor_to_image(images),
      text="<image>caption en\nAn abstract image of ",
      suffix=[p + '.' for p in prompts],
      return_tensors="pt",
      padding="longest",
    ).to("cuda")
    inputs["pixel_values"] = preprocess_paligemma(images)
    return -vqa.model(**inputs).loss / len(prompts[0].split())
  return reward_fn


def paligemma_reward(vqa, return_prob=False):
  A_id = vqa.processor.tokenizer.encode('A', add_special_tokens=False)[0]
  B_id = vqa.processor.tokenizer.encode('B', add_special_tokens=False)[0]
  keep_ids = torch.tensor([A_id, B_id], dtype=torch.long)
  answer_id = torch.tensor(B_id, dtype=torch.long)

  def reward_fn(prompts, images):
    questions = [f'Does this image show {p}?' for p in prompts]
    choices_list = [["no", "yes"]] * len(questions)
    vqa_prompts = [
      vqa.format_prompt(question, choices)
      for question, choices in zip(questions, choices_list, strict=True)
    ]

    inputs = vqa.processor(
      images=utils.tensor_to_image(images),
      text=vqa_prompts,
      return_tensors='pt',
      padding='longest',
    ).to('cuda')
    inputs["pixel_values"] = preprocess_paligemma(images)

    outputs = vqa.model(**inputs)
    logits = outputs.logits[:, -1, :]

    masked_logits = torch.full_like(logits, float('-inf'))
    masked_logits[:, keep_ids] = logits[:, keep_ids]
    if return_prob:
      return F.softmax(masked_logits, -1)[:, answer_id].sum()
    else:
      log_prob = F.log_softmax(masked_logits, -1)[:, answer_id]
      return log_prob.sum()
  return reward_fn


def l1_reward(target, image):
  return -torch.abs(target - image).mean()


def lpips_reward(net='vgg'):
  lpips_loss = lpips.LPIPS(net=net).to("cuda")
  def reward_fn(target, image):
    target = (2 * target) - 1
    image = (2 * image) - 1
    return -lpips_loss(target, image).mean()
  return reward_fn
