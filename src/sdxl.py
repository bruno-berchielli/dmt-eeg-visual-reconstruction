"""SDXL-turbo + IP-Adapter (ViT-H/14) conditioned only on a CLIP image
embedding, the same generator ATM uses for EEG reconstruction."""
import torch
from diffusers import AutoPipelineForText2Image
from transformers import CLIPVisionModelWithProjection

from common import device


class EmbeddingToImage:
    def __init__(self, steps=4, size=512):
        self.dev = device()
        self.steps = steps
        self.size = size
        encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", image_encoder=encoder,
            torch_dtype=torch.float16, variant="fp16").to(self.dev)
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                                  weight_name="ip-adapter_sdxl_vit-h.safetensors")
        self.pipe.set_ip_adapter_scale(1.0)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def clip_embed(self, images):
        fe = self.pipe.feature_extractor(images, return_tensors="pt").pixel_values
        return self.pipe.image_encoder(fe.to(self.dev, torch.float16)).image_embeds.float().cpu()

    @torch.no_grad()
    def __call__(self, embed, seed=0, latents=None):
        embed = embed.reshape(1, 1, -1).to(self.dev, torch.float16)
        g = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            prompt="", ip_adapter_image_embeds=[embed], num_inference_steps=self.steps,
            guidance_scale=0.0, height=self.size, width=self.size,
            generator=g, latents=latents).images[0]
