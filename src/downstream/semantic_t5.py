import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack


class MultiLayerTokenEmbedding(nn.Module):
    """
    Separate embedding table for each semantic layer.

    For layer-aware tokenization:
        token id x at layer 0 uses embedding table 0
        token id x at layer 1 uses embedding table 1
        ...
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        d_model: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, d_model) for _ in range(num_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        layer_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: LongTensor [B, S]
            layer_ids: LongTensor [B, S], values in [0, num_layers-1]

        Returns:
            inputs_embeds: FloatTensor [B, S, D]
        """
        if input_ids.shape != layer_ids.shape:
            raise ValueError(
                f"Shape mismatch: input_ids={tuple(input_ids.shape)} "
                f"layer_ids={tuple(layer_ids.shape)}"
            )

        # [L, V, D]
        stacked_weight = torch.stack(
            [emb.weight for emb in self.embeddings],
            dim=0,
        )
        # advanced indexing -> [B, S, D]
        return stacked_weight[layer_ids, input_ids]


class SemanticT5ForNextItem(T5PreTrainedModel):
    """
    T5-style encoder-decoder for semantic-ID next-item prediction.

    Key differences from vanilla T5:
    1. no NLP tokenizer
    2. layer-specific token embedding spaces
    3. encoder input is flattened semantic IDs
    4. decoder predicts a fixed-width semantic token sequence
    """

    config_class = T5Config

    def __init__(
        self,
        config: T5Config,
        sid_num_layers: int,
        sid_vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
    ):
        super().__init__(config)

        self.sid_num_layers = sid_num_layers
        self.sid_vocab_size = sid_vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.model_dim = config.d_model
        self.tie_word_embeddings = True

        self.token_embeddings = MultiLayerTokenEmbedding(
            num_layers=sid_num_layers,
            vocab_size=sid_vocab_size,
            d_model=config.d_model,
        )

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        decoder_config.use_cache = False

        self.encoder = T5Stack(encoder_config, embed_tokens=None)
        self.decoder = T5Stack(decoder_config, embed_tokens=None)

        self.post_init()

    def _build_encoder_layer_ids(
        self,
        input_ids: torch.Tensor,
        sid_width: int,
    ) -> torch.Tensor:
        """
        Encoder input is flattened from [T, L] into [T*L].
        Layer id is determined by position % L.
        """
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        layer_ids = pos % sid_width
        layer_ids = layer_ids.masked_fill(input_ids.eq(self.pad_token_id), 0)
        return layer_ids.long()

    def _build_decoder_input_ids(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher forcing decoder inputs:
            decoder_input = [BOS, y_0, y_1, ..., y_{L-2}]
            labels        = [y_0, y_1, ..., y_{L-1}]
        """
        batch_size, target_len = labels.shape
        decoder_input_ids = torch.full(
            (batch_size, target_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=labels.device,
        )
        decoder_input_ids[:, 0] = self.bos_token_id
        if target_len > 1:
            decoder_input_ids[:, 1:] = labels[:, :-1]
        return decoder_input_ids

    def _build_decoder_input_layer_ids(
        self,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decoder input token origins:
            pos 0 -> BOS, assign layer 0
            pos 1 -> previous token from layer 0
            pos 2 -> previous token from layer 1
            ...
        """
        batch_size, seq_len = decoder_input_ids.shape
        pos = torch.arange(seq_len, device=decoder_input_ids.device).unsqueeze(0).expand(batch_size, -1)
        layer_ids = torch.clamp(pos - 1, min=0)
        layer_ids = torch.clamp(layer_ids, max=self.sid_num_layers - 1)
        return layer_ids.long()

    def _build_prediction_layer_ids(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Output position i predicts semantic layer i.
        """
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos = torch.clamp(pos, max=self.sid_num_layers - 1)
        return pos.long()

    def _compute_logits(
        self,
        hidden_states: torch.Tensor,
        prediction_layer_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use the corresponding layer-specific embedding table as tied LM head.

        Args:
            hidden_states: [B, S, D]
            prediction_layer_ids: [B, S]

        Returns:
            logits: [B, S, V]
        """
        batch_size, seq_len, _ = hidden_states.shape
        logits = hidden_states.new_zeros(batch_size, seq_len, self.sid_vocab_size)

        if self.tie_word_embeddings:
            hidden_states = hidden_states * (self.model_dim ** -0.5)

        for layer_idx in range(self.sid_num_layers):
            mask = prediction_layer_ids.eq(layer_idx)  # [B, S]
            if not torch.any(mask):
                continue

            layer_hidden = hidden_states[mask]  # [N, D]
            weight = self.token_embeddings.embeddings[layer_idx].weight  # [V, D]
            layer_logits = F.linear(layer_hidden, weight)  # [N, V]
            logits[mask] = layer_logits

        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sid_width: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [B, S_enc]
            attention_mask: [B, S_enc]
            sid_width: int
            labels: [B, L]

        Returns:
            {
                "loss": scalar tensor or None,
                "logits": [B, L, V],
            }
        """
        encoder_layer_ids = self._build_encoder_layer_ids(
            input_ids=input_ids,
            sid_width=sid_width,
        )
        encoder_inputs_embeds = self.token_embeddings(
            input_ids=input_ids,
            layer_ids=encoder_layer_ids,
        )

        encoder_outputs = self.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=encoder_inputs_embeds,
            return_dict=True,
        )

        if labels is None:
            raise ValueError("labels must be provided during training/validation.")

        decoder_input_ids = self._build_decoder_input_ids(labels)
        decoder_input_layer_ids = self._build_decoder_input_layer_ids(decoder_input_ids)
        decoder_inputs_embeds = self.token_embeddings(
            input_ids=decoder_input_ids,
            layer_ids=decoder_input_layer_ids,
        )

        batch_size, target_len = labels.shape
        decoder_attention_mask = torch.ones(
            batch_size,
            target_len,
            dtype=torch.long,
            device=labels.device,
        )

        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        prediction_layer_ids = self._build_prediction_layer_ids(
            batch_size=batch_size,
            seq_len=target_len,
            device=labels.device,
        )
        logits = self._compute_logits(
            hidden_states=decoder_outputs.last_hidden_state,
            prediction_layer_ids=prediction_layer_ids,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.pad_token_id,
        )

        return {
            "loss": loss,
            "logits": logits,
        }

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sid_width: int,
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Greedy decoding for a fixed-width semantic ID target.

        Args:
            input_ids: [B, S_enc]
            attention_mask: [B, S_enc]
            sid_width: usually equals target width L
            max_new_tokens: if None, use sid_width

        Returns:
            generated_ids: [B, L]
        """
        batch_size = input_ids.size(0)
        target_len = sid_width if max_new_tokens is None else max_new_tokens

        encoder_layer_ids = self._build_encoder_layer_ids(
            input_ids=input_ids,
            sid_width=sid_width,
        )
        encoder_inputs_embeds = self.token_embeddings(
            input_ids=input_ids,
            layer_ids=encoder_layer_ids,
        )

        encoder_outputs = self.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=encoder_inputs_embeds,
            return_dict=True,
        )

        generated = []
        decoder_input_ids = torch.full(
            (batch_size, 1),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        for step in range(target_len):
            decoder_input_layer_ids = self._build_decoder_input_layer_ids(decoder_input_ids)
            decoder_inputs_embeds = self.token_embeddings(
                input_ids=decoder_input_ids,
                layer_ids=decoder_input_layer_ids,
            )

            decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long)

            decoder_outputs = self.decoder(
                input_ids=None,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )

            prediction_layer_ids = self._build_prediction_layer_ids(
                batch_size=batch_size,
                seq_len=decoder_input_ids.size(1),
                device=input_ids.device,
            )
            logits = self._compute_logits(
                hidden_states=decoder_outputs.last_hidden_state,
                prediction_layer_ids=prediction_layer_ids,
            )

            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # [B]
            generated.append(next_token)

            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token.unsqueeze(1)],
                dim=1,
            )

        return torch.stack(generated, dim=1)


def build_t5_config_from_dict(
    model_cfg: Dict,
    total_vocab_size: int,
    pad_token_id: int,
    bos_token_id: int,
) -> T5Config:
    d_model = int(model_cfg["d_model"])
    num_heads = int(model_cfg["num_heads"])

    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
        )

    return T5Config(
        vocab_size=total_vocab_size,
        d_model=d_model,
        d_ff=int(model_cfg["d_ff"]),
        d_kv=d_model // num_heads,
        num_layers=int(model_cfg["num_encoder_layers"]),
        num_decoder_layers=int(model_cfg["num_decoder_layers"]),
        num_heads=num_heads,
        dropout_rate=float(model_cfg.get("dropout_rate", 0.1)),
        layer_norm_epsilon=float(model_cfg.get("layer_norm_epsilon", 1e-6)),
        initializer_factor=float(model_cfg.get("initializer_factor", 1.0)),
        feed_forward_proj=str(model_cfg.get("feed_forward_proj", "relu")),
        pad_token_id=pad_token_id,
        decoder_start_token_id=bos_token_id,
        eos_token_id=None,
        is_encoder_decoder=True,
        use_cache=False,
    )