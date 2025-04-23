import json
import logging
# Remove torchaudio imports if no longer needed elsewhere
# import torchaudio # <-- REMOVE or comment out
import librosa # <-- ADD
import numpy as np # <-- ADD

from torch.utils.data import Dataset

import os
import json
import re
# import torchaudio # <-- REMOVE or comment out
from datasets import load_dataset
from torch.utils.data import Dataset

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. "
    "The Assistant first reasons internally, then answers. "
    "Use tags: <think> reasoning </think><answer> response </answer>."
)

class kk_Dataset(Dataset):
    def __init__(self, data_file, split="train", sample_rate=16000):
        dataset = load_dataset('parquet', data_files=data_file, split=split)

        def make_avqa_format(example, target_sample_rate):
            # ... (Sections 1-3 remain the same) ...
             # 1️⃣ Extract raw prompt string from list[dict]
            raw_prompt_data = example["prompt"]
            assert isinstance(raw_prompt_data, list) and "content" in raw_prompt_data[0], "prompt should be a list of dicts with 'content' key"
            raw_prompt = raw_prompt_data[0]["content"]

            # 2️⃣ Extract system + user blocks from flattened prompt string
            system_match = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", raw_prompt, re.DOTALL)
            user_match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", raw_prompt, re.DOTALL)

            system_text = system_match.group(1).strip() if system_match else "You are a thinking assistant."
            new_user_text = "<|AUDIO|> Listen carefully, then logically deduce who is knight or knave."

            # 3️⃣ Build Qwen-style multi-turn structure
            prompt_data = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": new_user_text}
            ]


            # 4️⃣ Audio path logic
            audio_path = os.path.join(
                os.path.dirname(data_file),
                "tts_train/merged" if "train" in split else "tts_test/merged",
                f"{example['index']:04d}_merged.wav"
            )
            if not os.path.exists(audio_path):
                print(f"⚠️ Audio file not found: {audio_path}")
                # Handle missing file
                return {
                    "audio": np.array([], dtype=np.float32), # Empty array
                    "prompt": prompt_data,
                    "index": example["index"],
                    "sample_rate": target_sample_rate,
                    "reward_model": example.get("reward_model", None)
                }

            # --- Replace torchaudio load and resample using librosa ---
            try:
                # librosa.load:
                # - loads audio into a numpy array (float32)
                # - `sr=target_sample_rate` resamples during loading
                # - `mono=False` keeps original channels (if >1), returns shape (channels, samples)
                # - `mono=True` mixes down to mono, returns shape (samples,)
                # We want the first channel, similar to original code's waveform[0].
                # Let's load as mono directly for simplicity if only the first channel is needed.
                waveform_np, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
                # waveform_np is now a 1D numpy array (samples,) at the target sample rate

                # If you strictly need to load stereo first and then take the first channel:
                # waveform_stereo, sr = librosa.load(audio_path, sr=target_sample_rate, mono=False)
                # if waveform_stereo.ndim > 1:
                #      waveform_np = np.ascontiguousarray(waveform_stereo[0, :]) # Take first channel
                # else:
                #      waveform_np = waveform_stereo # It was already mono

            except Exception as e:
                 print(f"Error loading/resampling audio file {audio_path} with librosa: {e}")
                 # Handle loading/resampling error
                 return {
                    "audio": np.array([], dtype=np.float32), # Empty array
                    "prompt": prompt_data,
                    "index": example["index"],
                    "sample_rate": target_sample_rate,
                    "reward_model": example.get("reward_model", None)
                 }
            # --- End replacement ---


            # 5️⃣ Return
            # waveform_np is already the desired 1D numpy array (first channel / mono)
            # from librosa.load(mono=True) or manual selection above.
            
            # waveform_np have to be numpy array, not other types like list or tensor
            assert isinstance(waveform_np, np.ndarray), "waveform_np should be a numpy array"
            return {
                "audio": waveform_np, # 1D numpy array
                "prompt": prompt_data,
                "index": example["index"],
                "sample_rate": target_sample_rate, # librosa already loaded at this rate
                "reward_model": example.get("reward_model", None)
            }

        # Pass the sample_rate to the map function using fn_kwargs
        self.dataset = dataset.map(make_avqa_format, fn_kwargs={'target_sample_rate': sample_rate})


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]