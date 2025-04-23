from transformers import Qwen2AudioForConditionalGeneration, AutoConfig
from modelling_audio_only_thinker import AudioOnlyThinker
import torch
import os

# Step 1: Load source model and state_dict
source_model = AudioOnlyThinker.from_pretrained(
    "/mnt/ssd3/chunhui/cleaned_audioonly_ckpt/",
    trust_remote_code=True
)

print("source_model:", source_model)


source_sd = source_model.state_dict()

# Step 2: Remap keys to match Qwen2Audio naming
remapped_sd = {}
for key in source_sd:
    new_key = key
    if key.startswith("audio_tower.proj."):
        new_key = key.replace("audio_tower.proj", "multi_modal_projector.linear")
    elif key.startswith("model."):
        new_key = key.replace("model.", "language_model.model.")
    elif key.startswith("lm_head."):
        new_key = key.replace("lm_head.", "language_model.lm_head.")
    elif "rotary_emb" in key:
        new_key = key.replace("rotary_emb", "language_model.model.rotary_emb")
    remapped_sd[new_key] = source_sd[key]

# Step 3: Prepare save path
save_path = "/mnt/ssd3/chunhui/converted_qwen2audio_from_audioonly"
os.makedirs(save_path, exist_ok=True)

# Step 4: load qwen 2 audio with aligned config
config = AutoConfig.from_pretrained(
    "/mnt/ssd3/chunhui/aligned_qwen2_5_audio/",
    trust_remote_code=True
)
config.save_pretrained(save_path)

# Step 5: Save model with injected weights (no random init at all)
# Qwen2AudioForConditionalGeneration.save_pretrained(
#     pretrained_model_name_or_path=None,
#     config=config,
#     state_dict=remapped_sd,
#     save_directory=save_path,
#     safe_serialization=True
# )

# Step 5: create model from state_dict
# model = Qwen2AudioForConditionalGeneration.from_pretrained(
#     pretrained_model_name_or_path=None,
#     config=config,
#     state_dict=remapped_sd
# )

model = Qwen2AudioForConditionalGeneration(config)
model.init_weights()
print("model:", model)
# Step 6: Save to disk
# model.save_pretrained(save_path, safe_serialization=True)

print(f"✅ Converted + saved model to: {save_path}")