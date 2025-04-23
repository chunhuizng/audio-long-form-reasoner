import os
from transformers import (
    Qwen2_5OmniModel,
    Qwen2_5OmniThinkerForConditionalGeneration, Qwen2AudioForConditionalGeneration
)

def extract_and_save_thinker(full_model_ckpt: str, save_path: str):
    print(f"🔄 Loading full Qwen2.5-Omni model from: {full_model_ckpt}")
    full_model = Qwen2_5OmniModel.from_pretrained(
        full_model_ckpt,
        device_map="cpu",  # 放 CPU 节省显存
        torch_dtype="auto",
    )

    print("✅ Extracting Thinker weights...")
    thinker_state_dict = full_model.thinker.state_dict()

    print("📦 Initializing Thinker-only model...")
    thinker_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        full_model_ckpt,
        device_map="cpu",
        torch_dtype="auto",
    )

    print("🔁 Loading extracted weights into Thinker model...")
    missing, unexpected = thinker_model.load_state_dict(thinker_state_dict, strict=False)
    print("⚠️ Missing keys:", missing)
    print("⚠️ Unexpected keys:", unexpected)

    print(f"💾 Saving Thinker-only model to: {save_path}")
    thinker_model.save_pretrained(save_path)
    print("✅ Done.")

class AudioOnlyThinker(Qwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = None  # 显式删掉
        if hasattr(self.config, "vision_config"):
            del self.config.vision_config  # 清掉 config 里残留

    def forward(self, *args, pixel_values=None, pixel_values_videos=None, **kwargs):
        return super().forward(*args, pixel_values=None, pixel_values_videos=None, **kwargs)
    
if __name__ == "__main__":
    # 👉 你可以替换为本地路径或者其它 HuggingFace repo
    full_model_repo = "Qwen/Qwen2.5-Omni-7B"
    output_dir = "/mnt/ssd3/chunhui/hf_cache/transformers/qwen2.5-audio-7b"

    # os.makedirs(output_dir, exist_ok=True)
    # extract_and_save_thinker(full_model_repo, output_dir)
    thinker_model = AudioOnlyThinker.from_pretrained(
    "/mnt/ssd3/chunhui/hf_cache/transformers/qwen2.5-audio-7b",
    device_map="auto",         # or "cpu" / "cuda"
    torch_dtype="auto",        # optional: bfloat16 / float16 if your GPU supports it
    attn_implementation="flash_attention_2",
)
    # thinker_model.push_to_hub("chunhuizng/AudioOnlyThinker")

    # print('thinker_model:', thinker_model)
    # audio = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
    # # Optional: print model summary
    # print("audio model is like this:", audio)

