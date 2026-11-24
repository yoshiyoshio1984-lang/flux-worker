"""
RunPod Serverless Worker - 画像生成 (Flux.1-dev BF16)
推奨GPU: L40S 48GB (Serverless Flex $1.89/h)
  - 1024px / 20-28ステップで 1枚 15-20秒
  - 1枚あたり 約 $0.008-0.011 (L40S 48GB。BF16フルモデル約24GB+エンコーダのため4090 24GBではギリギリ)
  - fp8版を使いたい場合は MODEL_ID=black-forest-labs/FLUX.1-dev-FP8 + torch.float8_e4m3fn が必要で、このコードのままではfp8にはならない
環境変数:
  HF_TOKEN - Hugging Faceトークン (Flux.1-devのライセンス承認済みのもの)
  MODEL_ID - デフォルト black-forest-labs/FLUX.1-dev
入力 (job['input']):
  prompt: str (必須)
  negative_prompt: str (任意, Fluxでは基本使わない)
  width: int (デフォルト 1024)
  height: int (デフォルト 1024)
  num_inference_steps: int (デフォルト 24)
  guidance_scale: float (デフォルト 3.5)
  seed: int (任意, -1でランダム)
出力:
  {"image_base64": "...", "width": 1024, "height": 1024, "seed": 123, "model": "..."}
"""
import os
import base64
import io
import random

import runpod
import torch
from diffusers import FluxPipeline

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-dev")

print(f"Loading {MODEL_ID} ...", flush=True)
pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=os.environ.get("HF_TOKEN"),
)
# Safety CheckerはFluxPipelineには標準で入っていない。
# オープンウェイトを自分でホストする構成なので、ここでフィルタはかけない。
pipe.enable_model_cpu_offload()
print("Model loaded.", flush=True)


def handler(job):
    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}

    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))
    steps = int(inp.get("num_inference_steps", 24))
    guidance = float(inp.get("guidance_scale", 3.5))
    seed = inp.get("seed", -1)
    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "width": width,
        "height": height,
        "seed": seed,
        "model": MODEL_ID,
    }


runpod.serverless.start({"handler": handler})
