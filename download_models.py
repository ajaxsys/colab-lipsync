import os
import sys
import time
import urllib.request
from huggingface_hub import hf_hub_download

MODELS_DIR = os.environ.get("MODELS_DIR", "/content/models")
os.makedirs(f"{MODELS_DIR}/base/google/umt5-xxl", exist_ok=True)
os.makedirs(f"{MODELS_DIR}/base/xlm-roberta-large", exist_ok=True)
os.makedirs(f"{MODELS_DIR}/dit", exist_ok=True)
os.makedirs(f"{MODELS_DIR}/infinitetalk", exist_ok=True)
os.makedirs(f"{MODELS_DIR}/lora", exist_ok=True)
os.makedirs(f"{MODELS_DIR}/wav2vec", exist_ok=True)
os.makedirs(f"{MODELS_DIR}/t5", exist_ok=True)
os.makedirs(f"{MODELS_DIR}/demucs", exist_ok=True)

print("🚀 开始高速下载 InfiniteTalk / Wan2.1 模型权重...")

# 1. Demucs 模型
demucs_path = f"{MODELS_DIR}/demucs/htdemucs.pth"
if not os.path.exists(demucs_path):
    print("📥 [1/5] 下载 Demucs 人声分离权重...")
    demucs_url = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
    urllib.request.urlretrieve(demucs_url, demucs_path)
    print("✅ Demucs 权重下载完成")
else:
    print("⚡ Demucs 权重已存在")

# 2. Chinese Wav2Vec2 权重
print("📥 [2/5] 下载 Chinese Wav2Vec2 音频特征提取权重...")
for fn in ["config.json", "preprocessor_config.json", "pytorch_model.bin"]:
    target = f"{MODELS_DIR}/wav2vec/{fn}"
    if not os.path.exists(target):
        print(f"  下载 wav2vec/{fn}...")
        hf_hub_download(repo_id="TencentGameMate/chinese-wav2vec2-base", filename=fn, local_dir=f"{MODELS_DIR}/wav2vec")
print("✅ Wav2Vec2 权重准备就绪")

# 3. MeiGen-AI/InfiniteTalk 单人口型模型与量化 T5
print("📥 [3/5] 下载 InfiniteTalk 单人口型模块与量化 T5...")
inf_target = f"{MODELS_DIR}/infinitetalk/infinitetalk_single.safetensors"
if not os.path.exists(inf_target):
    print("  下载 infinitetalk_single.safetensors...")
    p = hf_hub_download(repo_id="MeiGen-AI/InfiniteTalk", filename="single/infinitetalk.safetensors", local_dir=f"{MODELS_DIR}/infinitetalk")
    if p != inf_target:
        if os.path.exists(p):
            os.rename(p, inf_target)

t5_fp8 = f"{MODELS_DIR}/t5/t5_fp8.safetensors"
if not os.path.exists(t5_fp8):
    print("  下载 t5_fp8.safetensors...")
    p = hf_hub_download(repo_id="MeiGen-AI/InfiniteTalk", filename="quant_models/t5_fp8.safetensors", local_dir=f"{MODELS_DIR}/t5")
    if p != t5_fp8:
        if os.path.exists(p):
            os.rename(p, t5_fp8)

t5_map = f"{MODELS_DIR}/t5/t5_map_fp8.json"
if not os.path.exists(t5_map):
    print("  下载 t5_map_fp8.json...")
    p = hf_hub_download(repo_id="MeiGen-AI/InfiniteTalk", filename="quant_models/t5_map_fp8.json", local_dir=f"{MODELS_DIR}/t5")
    if p != t5_map:
        if os.path.exists(p):
            os.rename(p, t5_map)
print("✅ InfiniteTalk 模块与 T5 准备就绪")

# 4. Wan-AI/Wan2.1-I2V-14B-480P 基础组件 (VAE, CLIP, Tokenizer, Config)
print("📥 [4/5] 下载 Wan2.1 基础组件 (VAE, CLIP, 提示词分词器)...")
base_files = [
    "config.json",
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
    "xlm-roberta-large/sentencepiece.bpe.model",
    "xlm-roberta-large/special_tokens_map.json",
    "xlm-roberta-large/tokenizer.json",
    "xlm-roberta-large/tokenizer_config.json",
]
for bf in base_files:
    target = f"{MODELS_DIR}/base/{bf}"
    if not os.path.exists(target):
        print(f"  下载 base/{bf}...")
        hf_hub_download(repo_id="Wan-AI/Wan2.1-I2V-14B-480P", filename=bf, local_dir=f"{MODELS_DIR}/base")
print("✅ Wan2.1 基础组件准备就绪")

# 5. Kijai/WanVideo_comfy (Wan2.1 480P fp8 主模型 + LightX2V 蒸馏加速 LoRA)
print("📥 [5/5] 下载 Wan2.1 480P fp8 主模型及 LightX2V 加速 LoRA...")
dit_target = f"{MODELS_DIR}/dit/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
if not os.path.exists(dit_target):
    print("  下载 Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors (~15GB)...")
    hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors", local_dir=f"{MODELS_DIR}/dit")

lora_target = f"{MODELS_DIR}/lora/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
if not os.path.exists(lora_target):
    print("  下载 lightx2v 加速 LoRA (~450MB)...")
    p = hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", local_dir=f"{MODELS_DIR}/lora")
    if p != lora_target:
        if os.path.exists(p):
            os.rename(p, lora_target)
print("✅ 主模型与加速 LoRA 准备就绪")

print("\n🎉🎉 所有核心模型与权重已全部下载完毕！")
