"""
RunPod Serverless Worker - 画像生成 (Flux.1-dev FP8)
推奨GPU: L40S 48GB (Serverless Flex $1.89/h)
  - FP8版 (black-forest-labs/FLUX.1-dev-FP8) を使用。transformer が
    BF16 24GB → FP8 約12GB になり、BF16版ではワーカーのディスクが
    足りずモデルダウンロード中に "No space left on device" で落ちた
    ため (2026-11-25)、FP8版に切り替えた。
  - 注意: FLUX.1-dev-FP8 リポジトリは Diffusers形式ではない。
    model_index.json が無く、flux1-dev-fp8.safetensors (12.3GB)
    1ファイルだけのため、FluxPipeline.from_pretrained では
    404 になる。そのため transformer だけを from_single_file で
    読み込み、テキストエンコーダとVAEは FLUX.1-dev (BF16版、
    同じくライセンス承認済み) から from_pretrained で読む。
    ダウンロード総量は約24GB。VRAM使用量は約20-24GB。
    画質はBF16とほぼ同等。
  - 1024px / 20-28ステップで 1枚 15-20秒
  - 1枚あたり 約 $0.008-0.011 (L40S 48GB)
  - BF16版に戻したい場合は MODEL_ID=black-forest-labs/FLUX.1-dev
    にすれば from_pretrained 経路になる。
環境変数:
  HF_TOKEN - Hugging Faceトークン (FLUX.1-dev と FLUX.1-dev-FP8
    両方のライセンス承認済みのもの)
  MODEL_ID - デフォルト black-forest-labs/FLUX.1-dev-FP8
入力 (job['input']):
  prompt: str (必須)
  negative_prompt: str (任意, Fluxでは基本使わない)
  image_base64: str (任意, 指定すると img2img で描き直す)
  strength: float (任意, img2imgの変更の強さ 0-1, デフォルト 0.75)
  width: int (デフォルト 1024。img2img時は元画像サイズ)
  height: int (デフォルト 1024。img2img時は元画像サイズ)
  num_inference_steps: int (デフォルト 24)
  guidance_scale: float (デフォルト 3.5)
  seed: int (任意, -1でランダム)
出力:
  {"image_base64": "...", "width": 1024, "height": 1024, "seed": 123,
   "model": "...", "mode": "txt2img" | "img2img"}
"""
import os
import base64
import io
import random

import runpod
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-dev-FP8")
HF_TOKEN = os.environ.get("HF_TOKEN")

if "fp8" in MODEL_ID.lower():
    # FP8版は単体ファイル形式 (model_index.json が無い) なので、
    # transformer だけ from_single_file で読み、残りは BF16版から読む。
    # diffusers 0.32 の from_single_file は URL の扱いが不安定
    # (リポジトリIDだと ValueError、resolve URLだとファイル名の
    # 解決に失敗する) ため、hf_hub_download でローカルに落としてから
    # ローカルパスを渡す。これが一番確実。
    from huggingface_hub import hf_hub_download

    print("Downloading FP8 transformer file ...", flush=True)
    fp8_path = hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev-FP8",
        filename="flux1-dev-fp8.safetensors",
        token=HF_TOKEN,
    )
    print(f"Loading FP8 transformer from {fp8_path} ...", flush=True)
    transformer = FluxTransformer2DModel.from_single_file(
        fp8_path,
        torch_dtype=torch.float8_e4m3fn,
    )
    print("Loading text encoders + VAE from black-forest-labs/FLUX.1-dev ...", flush=True)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
else:
    print(f"Loading {MODEL_ID} ...", flush=True)
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
# Safety CheckerはFluxPipelineには標準で入っていない。
# オープンウェイトを自分でホストする構成なので、ここでフィルタはかけない。
pipe.enable_model_cpu_offload()
print("Model loaded.", flush=True)


def _round16(n):
    return max(16, (int(n) + 8) // 16 * 16)


def handler(job):
    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}

    steps = int(inp.get("num_inference_steps", 24))
    guidance = float(inp.get("guidance_scale", 3.5))
    seed = inp.get("seed", -1)
    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    image_b64 = inp.get("image_base64")
    if image_b64:
        # img2img: 貼られた画像を描き直す
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        width = _round16(inp.get("width") or img.width)
        height = _round16(inp.get("height") or img.height)
        strength = float(inp.get("strength", 0.75))
        strength = min(1.0, max(0.0, strength))
        image = pipe(
            prompt=prompt,
            image=img,
            strength=strength,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
        mode = "img2img"
    else:
        width = _round16(inp.get("width", 1024))
        height = _round16(inp.get("height", 1024))
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
        mode = "txt2img"

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "width": width,
        "height": height,
        "seed": seed,
        "model": MODEL_ID,
        "mode": mode,
    }


runpod.serverless.start({"handler": handler})
