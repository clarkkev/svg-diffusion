import collections
import copy
import glob
import os
import random
import time

import huggingface_hub
import torch
import torch.nn as nn

import competition_metrics
import diffusion
import global_stopwatch
import prompt_dataset
import reward_factory
import utils
import vectorization


def reward_grad(image, svg_image, prompt, reward_spec):
    name, reward_fn, weight = reward_spec
    image = image.detach().clone().requires_grad_(True)

    if 'svg' in name:
        if svg_image is None:
            return torch.zeros_like(image), {}
        reward = reward_fn(svg_image, image)
    else:
        reward = reward_fn(prompt, image)

    grad = torch.autograd.grad(reward, image,
                               retain_graph=False,
                               create_graph=False)[0]
    grad *= -weight

    metrics = {
        name: reward.item(),
        f'{name}_norm': grad.norm().item(),
        'total_reward': weight * reward.item()
    }
    return grad.detach(), metrics


def batched_reward_grad(images, svg_images, prompts, reward_functions):
    global hpsv2_model

    B = len(prompts)
    batch_grads = torch.zeros_like(images)
    metrics = collections.defaultdict(float)

    for spec in reward_functions:
        for i in range(B):
            grad, mb_metrics = reward_grad(
                images[i].unsqueeze(0),
                None if svg_images is None else svg_images[i].unsqueeze(0),
                [prompts[i]],
                spec
            )
            batch_grads[i] += grad.squeeze(0)
            for k, v in mb_metrics.items():
                metrics[k] += v / B

    return batch_grads, dict(metrics)


class LoRAOnly(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.params = torch.nn.ParameterList(
            [p for p in base.parameters() if p.requires_grad])


def train(pipe,
          scheduler,
          aesthetic_evaluator,
          vqa_evaluator,
          reward_functions,
          base_name="",
          save_dir="/content/drive/MyDrive/DRaFT/training_runs",
          dataset_path="/content/drive/MyDrive/DRaFT/generated_data/dataset.pkl",
          prompt_prefix="",
          prompt_suffix="",
          batch_size=2,
          lv_steps=3,
          step_every=2,
          lr=1.5e-4,
          seed=0,
          max_steps=5000,
          checkpoint_every=100):

  transformer = pipe.transformer
  optimizer = torch.optim.Adam(
      [p for p in transformer.parameters() if p.requires_grad], lr=lr)
  generator = torch.Generator().manual_seed(seed)

  model_name = (base_name + "_" +
                "_".join(f"{n}={w}" for n, _, w in reward_functions))
  model_dir = os.path.join(save_dir, model_name)
  image_dir = os.path.join(model_dir, "images")
  os.makedirs(image_dir, exist_ok=True)

  history_path = os.path.join(model_dir, "history.pkl")
  ckpt_files = sorted(glob.glob(os.path.join(model_dir, "checkpoint_*.pt")))
  if ckpt_files:
      ckpt_path = ckpt_files[-1]
      ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
      step = ckpt["step"]
      history = ckpt.get("hist", [])
      optimizer.load_state_dict(ckpt["opt"])
      generator.set_state(ckpt["gen"])
      LoRAOnly(transformer).load_state_dict(ckpt["lora"], strict=False)
      print(f"Resumed from {ckpt_path} (step {step})")
  else:
      step, history = 0, []
      print("Starting fresh run")

  batch = []
  for example_ind, rows in enumerate(
      prompt_dataset.dataset_iterator(dataset_path, n_epochs=1000)):
    if example_ind // batch_size < step:
      continue
    batch.append(rows)
    if len(batch) < batch_size:
      continue
    if step >= max_steps:
      break
    if step <= 1:
      global_stopwatch.clear()

    torch.cuda.reset_peak_memory_stats()

    descriptions = [rows['description'].iloc[0] for rows in batch]
    prompts = [prompt_prefix + p + prompt_suffix for p in descriptions]

    with torch.no_grad():
      global_stopwatch.start("generate")
      images = diffusion.generate(
          pipe, prompts, height=384, width=384, generator=generator)
      latents = diffusion.encode_image(pipe.vae, utils.image_to_tensor(images))
      global_stopwatch.stop()

    metrics = collections.defaultdict(float)
    svg_image = None
    if step % 2 == 0:
      svg_images = []
      for description, rows, image in zip(descriptions, batch, images):
        global_stopwatch.start("vectorize")
        svg_image = utils.svg_to_image(
            vectorization.make_svg(image, refine_iters=40)).resize((384, 384))
        svg_images.append(utils.image_to_tensor(svg_image))
        global_stopwatch.stop()

        global_stopwatch.start("evaluate")
        defended_image = competition_metrics.prep_for_score(svg_image)
        aesthetic_score = aesthetic_evaluator.score(defended_image)

        vqa_self = vqa_evaluator.score([
              f'Does this image show {description}?'],
                [['no', 'yes']], ['yes'], defended_image)
        vqa_score = vqa_evaluator.score(
            rows['question'].tolist(), rows['choices'].tolist(),
            rows['answer'].tolist(), defended_image)
        instance_score = competition_metrics.harmonic_mean(
            vqa_score, aesthetic_score, beta=0.5)
        metrics['aesthetic_score'] += aesthetic_score
        metrics['vqa_score'] += vqa_score
        metrics['vqa_self'] += vqa_self
        metrics['instance_score'] += instance_score
        global_stopwatch.stop()
      svg_images = torch.stack(svg_images)
      metrics = {k: v / len(images) for k, v in metrics.items()}

    global_stopwatch.start("draft")
    draft_metrics = collections.defaultdict(float)
    for _ in range(lv_steps):
      t, sigma = diffusion.look_up_timestep(scheduler, random.randint(30, 300))
      z_t = diffusion.add_noise(latents, sigma)
      z0 = diffusion.denoise(pipe, prompts, z_t, t, sigma,
                             transformer=transformer)
      reconstructed = diffusion.decode_image(pipe.vae, z0)

      drdimage, step_metrics = batched_reward_grad(
        reconstructed, svg_images, descriptions, reward_functions)
      for k, v in step_metrics.items():
        draft_metrics[k] = draft_metrics.get(k, 0) + v
      (drdimage * reconstructed / step_every).sum().backward()
    global_stopwatch.stop()

    if step % step_every == 0:
      nn.utils.clip_grad_norm_([p for p in transformer.parameters()
                                if p.requires_grad], max_norm=10.0)
      optimizer.step()
      optimizer.zero_grad()

    metrics.update({k: v / lv_steps for k, v in draft_metrics.items()})
    metrics['time'] = time.time()
    history.append(metrics)

    if step % 2 == 0:
      print(f"Step {step}, Reward: {metrics['total_reward']:.2f}, Model: {model_name}")
      print(", ".join([f"{name}: {r:0.3f}" for name, r in metrics.items()
                      if name != 'time']))
    if step % 5 == 0:
      images[0].save(f'{image_dir}/image_{step}.png')
    if step % 10 == 0:
      utils.write_pickle(history, history_path)
      print("Times:")
      global_stopwatch.print_times()
    if step % checkpoint_every == 0:
      global_stopwatch.start("checkpoint")
      ckpt_path = os.path.join(model_dir, f"checkpoint_{step:06d}.pt")
      torch.save(
          {
              "step": step,
              "opt":  optimizer.state_dict(),
              "gen":  generator.get_state(),
              "hist": history,
              "lora": LoRAOnly(transformer).state_dict(),
          },
          ckpt_path,
      )
      print(f"Saved checkpoint → {ckpt_path}")
      global_stopwatch.stop()

    step += 1
    batch = []


def main():
    # load stable diffusion
    huggingface_hub.login()
    pipe = diffusion.load_stable_diffusion(use_t5=False)
    pipe.vae.enable_gradient_checkpointing()
    scheduler = copy.deepcopy(pipe.scheduler)
    diffusion.add_lora(pipe.transformer)
    pipe.transformer = torch.compile(pipe.transformer)
    pipe.vae = torch.compile(pipe.vae)

    # load evaluation metrics
    aesthetic_evaluator = competition_metrics.AestheticEvaluator('ViT-L/14')
    vqa_evaluator = competition_metrics.VQAEvaluator()
    vqa_evaluator.model.requires_grad_(False)
    vqa_evaluator.model.config.text_config.use_cache = False

    # load reward functions
    aesthetic_reward = reward_factory.aesthetic_reward(aesthetic_evaluator)
    siglip_reward = reward_factory.siglip_reward()
    hpsv2_reward = reward_factory.hpsv2_reward()
    pickscore_reward = reward_factory.pickscore_reward()
    lpips_reward = reward_factory.lpips_reward()

    reward_functions = [
        ('svg_lpips', lpips_reward, 20),
        ('aesthetic', aesthetic_reward, 0.2),
        ('hpsv2', hpsv2_reward, 25),
        ('pickscore', pickscore_reward, 40),
        ('siglip', siglip_reward, 2),
        ('vqa_self_train',
        reward_factory.paligemma_reward(vqa_evaluator, True), 100),
    ]

    train(pipe, scheduler, aesthetic_evaluator, vqa_evaluator, reward_functions, 'draft-finetune')
