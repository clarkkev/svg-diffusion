"""Utility functions for loading and running SD3.5 Medium."""
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig
import torch

from transformers import T5EncoderModel, BitsAndBytesConfig


def load_stable_diffusion(use_t5=False, quantize_t5=True):
  if use_t5:
    if quantize_t5:
      quantization_config = BitsAndBytesConfig(load_in_8bit=True)
      text_encoder = T5EncoderModel.from_pretrained(
          "stabilityai/stable-diffusion-3.5-medium",
          subfolder="text_encoder_3",
          quantization_config=quantization_config,
      )
      pipe = StableDiffusion3Pipeline.from_pretrained(
          "stabilityai/stable-diffusion-3.5-medium",
          text_encoder_3=text_encoder,
          torch_dtype=torch.float16
      ).to("cuda")
    else:
      pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium"
      ).to("cuda", dtype=torch.bfloat16)
  else:
    pipe = StableDiffusion3Pipeline.from_pretrained(
      "stabilityai/stable-diffusion-3.5-medium",
      text_encoder_3=None,
      tokenizer_3=None,
    ).to("cuda", dtype=torch.bfloat16)

  pipe.transformer.enable_gradient_checkpointing()

  pipe.transformer.requires_grad_(False)
  pipe.vae.requires_grad_(False)
  pipe.text_encoder.requires_grad_(False)
  pipe.text_encoder_2.requires_grad_(False)
  if pipe.text_encoder_3 is not None:
    pipe.text_encoder_3.requires_grad_(False)

  return pipe


def add_lora(transformer, name="default"):
  lora_rank = 32
  target_modules = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
  ]
  transformer_lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank,
    init_lora_weights="gaussian",
    target_modules=target_modules,
  )
  transformer.add_adapter(transformer_lora_config, adapter_name=name)


def generate(pipe, prompt, height=384, width=384, **kwargs):
  with torch.no_grad():
    return pipe(prompt, guidance_scale=7.0, num_inference_steps=25,
                height=height, width=width, **kwargs).images


def encode_image(vae, image):
  image = 2 * image - 1
  latents = vae.encode(image).latent_dist.mean
  return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def decode_image(vae, latents):
  latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
  image = vae.decode(latents).sample
  return ((image + 1) / 2).clamp(0, 1)


def look_up_timestep(scheduler, t):
  timestep = scheduler.timesteps[-t - 1].to("cuda", dtype=torch.bfloat16)
  sigma = scheduler.sigmas[-t - 1].to("cuda", dtype=torch.bfloat16)
  return timestep, sigma


def add_noise(latents, sigma):
  noise = torch.randn_like(latents)
  return (1.0 - sigma) * latents + sigma * noise


def compute_embeddings(pipe, prompts):
  with torch.no_grad():
    return pipe.encode_prompt(prompts, None, None, max_sequence_length=77)


def denoise(pipe, prompts, z_t, timestep, sigma, transformer=None, embeddings=None):
  if transformer is None:
    transformer = pipe.transformer
  if embeddings is None:
    embeddings = compute_embeddings(pipe, prompts)

  (prompt_embeds,
  negative_prompt_embeds,
  pooled_prompt_embeds,
  negative_pooled_prompt_embeds) = embeddings
  combined_prompt_embeds = torch.cat(
      [negative_prompt_embeds, prompt_embeds], dim=0)
  combined_pooled_prompt_embeds = torch.cat(
      [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

  vel_pred = transformer(
      hidden_states=torch.cat([z_t] * 2),
      timestep=timestep.expand(2 * z_t.shape[0]),
      encoder_hidden_states=combined_prompt_embeds,
      pooled_projections=combined_pooled_prompt_embeds,
      return_dict=False,
  )[0]

  vel_pred_uncond, vel_pred_text = vel_pred.chunk(2)
  vel_pred = vel_pred_uncond + pipe.guidance_scale * (
      vel_pred_text - vel_pred_uncond)

  return z_t - sigma * vel_pred
