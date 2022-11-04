# custom_diffusers_wrapper (Work in Progress)

* Pipeline of txt2img in diffusers format implementing FrozenCLIPEmbedderWithCustomWords and CLIP stop at last layers
* Based on https://github.com/AUTOMATIC1111/stable-diffusion-webui
* This differs from the CompVis-based txt2img results due to sampler differences

## HOW TO USE
```python
!git clone https://github.com/kawaiiprompter/custom_diffusers_wrapper
!pip install -qq git+https://github.com/huggingface/diffusers
!pip install -qq transformers scipy ftfy lark

import sys
sys.path.append("custom_diffusers_wrapper")
from pipeline_stable_diffusion import StableDiffusionPipeline

from pathlib import Path
import torch
from torch import autocast

from diffusers.schedulers import DDIMScheduler
scheduler = DDIMScheduler(
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
    beta_schedule="scaled_linear",
)

model_path = "hakurei/waifu-diffusion" # when download waifu diffusion
# model_path = "./models/waifu-diffusion" when　use local model files
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float32,
    scheduler=scheduler
).to("cuda")

prompt = "((masterpiece)), 1girl, aqua eyes, baseball cap, (((blonde hair))), closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
negative_prompt = "((ugly))"

# use FrozenCLIPEmbedderWithCustomWords(True) or original diffusers embedder(False)
use_custom_encoder = True

# select last layer of CLIP model (Valid only use_custom_encoder=True)
CLIP_stop_at_last_layers = 2 

seed = 1
generator = torch.Generator("cuda").manual_seed(seed)
with autocast("cuda"):
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
        use_custom_encoder=use_custom_encoder,
        CLIP_stop_at_last_layers=CLIP_stop_at_last_layers,
        ).images[0]

image.save("test.png")
```

## Credits
* Stable Diffusion - https://github.com/CompVis/stable-diffusion
* AUTOMATIC1111 - https://github.com/AUTOMATIC1111/stable-diffusion-webui
