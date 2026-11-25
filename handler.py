"""
RunPod Serverless Worker - 画像生成 (Flux.1-dev BF16 フルモデル)
推奨GPU: L40S 48GB (Serverless Flex $1.89/h)
  - BF16フルモデル (black-forest-labs/FLUX.1-dev) を from_pretrained
    でそのまま読む。標準的で確実な読み込み方。
  - かつて FP8版 (FLUX.1-dev-FP8) を使っていたが、FP8ファイルを
    from_single_file で BF16 に変換して読むやり方では、生成は
    成功するものの出力がノイズだけになった (2026-11-25 13:17 JST、
    夕日の海岸のプロンプトでカラーの砂嵐だけが出た)。
    diffusers 公式も「FP8 inference can be brittle」とし、直接の
    FP8読み込みではなく optimum-quanto を推奨している。
    そのため FP8経路は廃止し、BF16フルモデルに戻した。
  - BF16版はモデル約24GB。以前はコンテナディスクが5GBしかなく
    ダウンロード中に "No space left on device" で落ちたが、
    ディスクは50GBに増やしてあるので入る。
  - 1024px / 20-28ステップで 1枚 15-20秒
  - 1枚あたり 約 $0.008-0.011 (L40S 48GB)
環境変数:
  HF_TOKEN - Hugging Faceトークン (FLUX.1-dev のライセンス承認済みのもの)
  MODEL_ID - デフォルト black-forest-labs/FLUX.1-dev
入力 (job['input']):
  prompt: str (必須)
  negative_prompt: str (任意, 避けたいものを英語で指定。
    アプリ側が日本語プロンプトの英語化と一緒に送る。
    例: "clothed, wearing clothes, dressed, underwear, bikini")
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
from diffusers import FluxPipeline

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-dev")
HF_TOKEN = os.environ.get("HF_TOKEN")

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

# 描き直し (img2img) 用パイプライン。FluxPipeline (テキスト→画像) には
# image 引数が無いため、別クラスの FluxImg2ImgPipeline を使う。
# 部品 (transformer / text encoder / VAE) は上の pipe と共有するので
# モデルの再ダウンロードは発生しない。初回の描き直し時にだけ作る。
_img2img_pipe = None


def _get_img2img_pipe():
    global _img2img_pipe
    if _img2img_pipe is None:
        from diffusers import FluxImg2ImgPipeline

        print("Loading FluxImg2ImgPipeline from FluxPipeline components ...", flush=True)
        # FluxImg2ImgPipeline には image_encoder を渡してはいけない。
        # FluxPipeline の components には image_encoder が含まれるが、
        # diffusers 0.32.2 の FluxImg2ImgPipeline.__init__ には
        # image_encoder 引数が無いため、**pipe.components だと
        # "got an unexpected keyword argument 'image_encoder'" で
        # 失敗する (2026-11-25 16:57 JST)。必要な部品だけ明示して渡す。
        c = pipe.components
        _img2img_pipe = FluxImg2ImgPipeline(
            scheduler=c["scheduler"],
            text_encoder=c["text_encoder"],
            tokenizer=c["tokenizer"],
            text_encoder_2=c["text_encoder_2"],
            tokenizer_2=c["tokenizer_2"],
            vae=c["vae"],
            transformer=c["transformer"],
        )
    return _img2img_pipe


def _round16(n):
    return max(16, (int(n) + 8) // 16 * 16)


def handler(job):
    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}

    steps = int(inp.get("num_inference_steps", 24))
    guidance = float(inp.get("guidance_scale", 3.5))
    # ネガティブプロンプト (任意): アプリ側が英語化と一緒に送る。
    # 例: "clothed, wearing clothes, dressed, underwear, bikini"
    # FluxPipeline にそのまま渡す。空なら指定なしと同じ。
    negative_prompt = inp.get("negative_prompt", "") or ""
    # ネガティブプロンプトは true_cfg_scale > 1 のときだけ効く
    # (diffusers の Flux は true CFG でのみネガティブを使う)。
    # true_cfg_scale を渡さない (既定 1.0) とネガティブは無視される。
    # ネガティブがあるときだけ 2.0 にする。
    # 注意: true CFG は計算が約2倍になる (生成時間・料金も約2倍)。
    true_cfg_scale = 2.0 if negative_prompt else 1.0
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
        # diffusers 0.35.1 の FluxImg2ImgPipeline.__call__ は
        # negative_prompt / true_cfg_scale を受け付ける。
        # (0.32.2 では受け付けず TypeError になったため、一時的に
        # 渡さない修正を入れていたが、0.35.1 への更新で元に戻す。)
        image = _get_img2img_pipe()(
            prompt=prompt,
            image=img,
            strength=strength,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
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
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
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
