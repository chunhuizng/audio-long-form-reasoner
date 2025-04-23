from modelling_audio_only_thinker import AudioOnlyThinker
from transformers import AutoTokenizer
import os
import json

# 1. 设置模型 repo（HuggingFace 上的地址）
remote_model_name = "chunhuizng/AudioOnlyThinker"
save_path = "/mnt/ssd3/chunhui/cleaned_audioonly_ckpt"

# 2. 加载模型（本地类 + 远程 ckpt）
print("📥 Loading model...")
model = AudioOnlyThinker.from_pretrained(
    remote_model_name,
    trust_remote_code=True
)

# 3. 删除视觉模块 + 更新 config
print("🧹 Removing visual module...")
model.visual = None
if hasattr(model.config, "vision_config"):
    del model.config.vision_config

# Optional: 强制 bfloat16 精度保存
# model = model.to(dtype="bfloat16")

# 4. 创建保存路径
os.makedirs(save_path, exist_ok=True)

# 5. 保存 config.json（修改后去除视觉配置）
cleaned_config = model.config.to_dict()
if "vision_config" in cleaned_config:
    del cleaned_config["vision_config"]

# Save updated config
with open(os.path.join(save_path, "config.json"), "w") as f:
    json.dump(cleaned_config, f, indent=2)

# 6. 保存 safetensors 格式 checkpoint
print("💾 Saving safetensors model...")
model.save_pretrained(save_path, safe_serialization=True)

# 7. 保存 tokenizer（从远程下载对应 tokenizer）
print("📦 Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(remote_model_name, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

print(f"✅ Cleaned model saved to: {save_path}")