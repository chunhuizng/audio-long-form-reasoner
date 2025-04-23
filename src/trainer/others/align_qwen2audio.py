
from transformers import Qwen2AudioForConditionalGeneration
from modelling_audio_only_thinker import AudioOnlyThinker

def compare_state_dicts(target_sd, source_sd):
    mismatches = []
    matched = []
    target_keys = set(target_sd.keys())
    source_keys = set(source_sd.keys())

    for key in sorted(target_keys.union(source_keys)):
        in_target = key in target_sd
        in_source = key in source_sd

        if in_target and in_source:
            if target_sd[key].shape != source_sd[key].shape:
                mismatches.append((key, source_sd[key].shape, target_sd[key].shape))
            else:
                matched.append(key)
        elif in_target:
            mismatches.append((key, None, target_sd[key].shape))
        elif in_source:
            mismatches.append((key, source_sd[key].shape, None))

    return matched, mismatches


if __name__ == "__main__":
    print("📥 Loading Qwen2AudioForConditionalGeneration...")
    target_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True
    )
    target_sd = target_model.state_dict()
    print("✅ Loaded target model.")

    print("📥 Loading AudioOnlyThinker from /mnt/ssd3/chunhui/cleaned_audioonly_ckpt ...")
    audioonly_model = AudioOnlyThinker.from_pretrained(
        "/mnt/ssd3/chunhui/cleaned_audioonly_ckpt", trust_remote_code=True
    )
    source_sd = audioonly_model.state_dict()
    print("✅ Loaded source model.")

    matched_keys, mismatched = compare_state_dicts(target_sd, source_sd)

    print(f"🔍 Matched parameters: {len(matched_keys)}")
    print(f"❗Mismatched or missing parameters: {len(mismatched)}")
    
    
    print("target_model:", target_model)
    print("audioonly_model:", audioonly_model)    
