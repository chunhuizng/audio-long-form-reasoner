import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import Parameter

# Assuming original imports are available...
from transformers.models.llama.modeling_llama import rotate_half
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
# from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig # Not needed
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    # Qwen2_5_VisionTransformerPretrainedModel, # Not needed
    Qwen2_5_VLAttention,
    Qwen2_5_VLMLP,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    # Qwen2_5_VLVisionBlock, # Not needed
    Qwen2RMSNorm,
)
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoderLayer # Make sure Qwen2_5OmniAudioEncoder is importable
# from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding # Check if needed by text model

from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel # Import base PreTrainedModel
from transformers.utils import logging # Assuming logging is handled by parent

# Import necessary components from the original file/library if not automatically inherited
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel, # Need the Text Model definition
    Qwen2_5OmniPreTrainedModelForConditionalGeneration, # Need the base pretrained model
    Qwen2_5OmniThinkerCausalLMOutputWithPast, # Need the output class definition
    Qwen2_5OmniAudioEncoder,
)

# Placeholder for QWEN2_5OMNITHINKER_INPUTS_DOCSTRING if needed, or remove decorator if not critical
QWEN2_5OMNITHINKER_INPUTS_DOCSTRING = "..."
# Placeholder for replace_return_docstrings if needed
def replace_return_docstrings(**kwargs):
    def decorator(func):
        return func
    return decorator
# Placeholder for add_start_docstrings_to_model_forward if needed
def add_start_docstrings_to_model_forward(*args, **kwargs):
     def decorator(func):
        return func
     return decorator


logger = logging.get_logger(__name__)

class AudioOnlyThinker(Qwen2_5OmniThinkerForConditionalGeneration):
    # Keep the config class, potentially modify later if config causes issues
    # config_class = Qwen2_5OmniThinkerConfig
    # Adjust base_model_prefix if needed, likely keep "thinker"
    # base_model_prefix = "thinker"
    # Remove "Qwen2_5OmniVisionEncoder" from no_split_modules
    _no_split_modules = ["Qwen2_5OmniAudioEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        # Call the grandparent's __init__ to set up basic config and potentially weights
        # Note: Qwen2_5OmniThinkerForConditionalGeneration inherits from Qwen2_5OmniPreTrainedModelForConditionalGeneration
        # We might need to call PreTrainedModel.__init__ if the intermediate one also loads vision.
        # Let's try the immediate parent's base class first.
        Qwen2_5OmniPreTrainedModelForConditionalGeneration.__init__(self, config)
        # super(Qwen2_5OmniThinkerForConditionalGeneration, self).__init__(config) # Alternate way to call grandparent

        # --- Replicate necessary initializations from Qwen2_5OmniThinkerForConditionalGeneration.__init__ ---
        self.spatial_merge_size = 2
        # Initialize Audio Tower
        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(
            config.audio_config, attn_implementation=config._attn_implementation
        )

        # Initialize Text Model
        self.vocab_size = config.text_config.vocab_size
        # self.language_model = Qwen2_5OmniThinkerTextModel._from_config(
        self.model = Qwen2_5OmniThinkerTextModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Set visual tower explicitly to None
        self.visual = None

        # Other attributes
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        # spatial_merge_size is vision related, set to None or default if needed elsewhere, otherwise remove.
        # self.spatial_merge_size = config.vision_config.spatial_merge_size # Remove this line
        self.rope_deltas = None

        # --- End Replication ---

        # Call post_init at the end
        self.post_init()

    # get_input_embeddings and set_input_embeddings are usually inherited correctly,
    # but we define them explicitly if self.language_model structure matches the parent's intent.
    def get_input_embeddings(self):
        # return self.language_model.get_input_embeddings()
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # self.language_model.set_input_embeddings(value)
        self.model.set_input_embeddings(value)

    # No need for vision-related config patching if self.visual is never used.

    # Override forward method completely
    # @add_start_docstrings_to_model_forward(QWEN2_5OMNITHINKER_INPUTS_DOCSTRING)
    # @replace_return_docstrings(
    #     output_type=Qwen2_5OmniThinkerCausalLMOutputWithPast, config_class="Qwen2_5OmniThinkerConfig"
    # )
    def _forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None
        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text , audios , image and video
        if input_ids is not None and input_ids.shape[1] != 1:  # Prefill stage
            if input_features is not None:
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )
                feature_lens = (
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )
                audio_outputs = self.audio_tower(
                    input_features,
                    feature_lens=feature_lens,
                    aftercnn_lens=audio_feat_lengths,
                )
                audio_features = audio_outputs.last_hidden_state
                if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                    raise ValueError("length of audio_features should match audio_output_lengths")
                audio_mask = (
                    (input_ids == self.config.audio_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None, # REMOVED
        video_grid_thw: Optional[torch.LongTensor] = None, # REMOVED
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False, # REMOVED
        audio_seqlens: Optional[torch.LongTensor] = None, # Keep arg for signature consistency, but won't be used
        second_per_grids: Optional[torch.Tensor] = None, # REMOVED
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the RoPE index for sequences containing text and audio tokens
        in an audio-only context.

        Following the original model's logic for non-visual inputs, this method
        assigns sequential position IDs (like text-only) regardless of token type.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices (1 for non-masked, 0 for masked).
            audio_seqlens (`torch.LongTensor`, *optional*):
                This argument is ignored in the audio-only context, following the
                original model's behavior for non-visual inputs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`):
                Sequential position IDs (0, 1, 2, ...) for non-padded tokens, expanded to 3 dimensions.
                Padded positions are set to 1.
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size, 1)`):
                Always returns 0 in this context, as no position expansion occurs relative
                to the sequence length.
        """
        if input_ids is None or attention_mask is None:
            # Or handle based on how your forward pass ensures these exist
            raise ValueError("`input_ids` and `attention_mask` must be provided.")

        # Simple sequential position calculation (like text-only)
        # This matches the original code's 'else' block logic for non-visual inputs.
        position_ids_1d = attention_mask.long().cumsum(-1) - 1
        # Use 1 for padding, matching original code's text-only behavior
        position_ids_1d.masked_fill_(attention_mask == 0, 1)

        # Expand to 3 dimensions for RoPE application
        position_ids = position_ids_1d.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)

        # Calculate deltas: (max_pos + 1) - valid_len = (valid_len - 1 + 1) - valid_len = 0
        # Original code calculates this slightly differently but should result in 0 for seq pos ids.
        valid_lengths = torch.sum(attention_mask, dim=-1, keepdim=True)
        # Ensure delta is float and has shape (batch_size, 1)
        mrope_position_deltas = torch.zeros((input_ids.shape[0], 1), dtype=torch.float32, device=input_ids.device)

        # Original calculation from else branch for reference (should also be ~0):
        # max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0] # Max calculated ID
        # mrope_position_deltas_orig = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas
    

# model = AudioOnlyThinker.from_pretrained("/data/01/chunhui/huggingface_cache/transformers/AudioOnlyThinker")

# print(model)



# AudioOnlyThinker(
#   (audio_tower): Qwen2_5OmniAudioEncoder(
#     (conv1): Conv1d(128, 1280, kernel_size=(3,), stride=(1,), padding=(1,))
#     (conv2): Conv1d(1280, 1280, kernel_size=(3,), stride=(2,), padding=(1,))
#     (positional_embedding): SinusoidsPositionEmbedding()
#     (audio_bos_eos_token): Embedding(2, 3584)
#     (layers): ModuleList(
#       (0-31): 32 x Qwen2_5OmniAudioEncoderLayer(
#         (self_attn): Qwen2_5OmniAudioSdpaAttention(
#           (k_proj): Linear(in_features=1280, out_features=1280, bias=False)
#           (v_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (q_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
#         )
#         (self_attn_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#         (activation_fn): GELUActivation()
#         (fc1): Linear(in_features=1280, out_features=5120, bias=True)
#         (fc2): Linear(in_features=5120, out_features=1280, bias=True)
#         (final_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#       )
#     )
#     (ln_post): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#     (avg_pooler): AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))
#     (proj): Linear(in_features=1280, out_features=3584, bias=True)
#   )
#   (model): Qwen2_5OmniThinkerTextModel(
#     (embed_tokens): Embedding(152064, 3584)
#     (layers): ModuleList(
#       (0-27): 28 x Qwen2_5OmniDecoderLayer(
#         (self_attn): Qwen2_5OmniSdpaAttention(
#           (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
#           (k_proj): Linear(in_features=3584, out_features=512, bias=True)
#           (v_proj): Linear(in_features=3584, out_features=512, bias=True)
#           (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
#           (rotary_emb): Qwen2_5OmniRotaryEmbedding()
#         )
#         (mlp): Qwen2MLP(
#           (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
#           (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
#           (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
#         (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
#       )
#     )
#     (norm): Qwen2RMSNorm((3584,), eps=1e-06)
#     (rotary_emb): Qwen2_5OmniRotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=3584, out_features=152064, bias=False)
# )