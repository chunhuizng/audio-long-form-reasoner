from transformers import (
    Qwen2AudioForConditionalGeneration, Qwen2_5OmniModel, Qwen2_5OmniThinkerForConditionalGeneration
)
model = Qwen2_5OmniModel.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
# display model clearly
print("omni model is like this:")
print(model)
# model = Qwen2_5OmniModel.from_pretrained( #Qwen2_5OmniThinkerForConditionalGeneration
#     "Qwen/Qwen2.5-Omni-7B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )
# print("omni loaded")
# audio = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
# print("audio loaded")

# HF_HOME=/mnt/ssd3/chunhui/hf_cache TRANSFORMERS_CACHE=/mnt/ssd3/chunhui/hf_cache/transformers HF_DATASETS_CACHE=/mnt/ssd3/chunhui/hf_cache/datasets HF_METRICS_CACHE=/mnt/ssd3/chunhui/hf_cache/metrics CUDA_VISIBLE_DEVICES=7 python play_ckpt.py

# # 拷贝 audio encoder
# audio.model.audio_encoder.load_state_dict(
#     omni.model.audio_encoder.state_dict()
# )

# # 如果有 audio_adapter
# if hasattr(omni.model, "audio_adapter") and hasattr(audio.model, "audio_adapter"):
#     audio.model.audio_adapter.load_state_dict(
#         omni.model.audio_adapter.state_dict()
#     )

# # 拷贝 Thinker (LLM)
# audio.model.llm.load_state_dict(
#     omni.model.llm.state_dict()
# )

# # 拷贝文本生成头
# audio.lm_head.load_state_dict(
#     omni.lm_head.state_dict()
# )
# audio.save_pretrained("my-qwen2-audio-upgraded")
