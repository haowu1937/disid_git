import math,random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import random

@dataclass
class BeamCandidate:
    tokens: List[int]
    score: float


class PrefixTrieNode:
    def __init__(self):
        self.children: Dict[int, "PrefixTrieNode"] = {}
        self.is_terminal: bool = False


class PrefixTrie:
    def __init__(self):
        self.root = PrefixTrieNode()

    def insert(self, token_seq: List[int]) -> None:
        node = self.root
        for token in token_seq:
            if token not in node.children:
                node.children[token] = PrefixTrieNode()
            node = node.children[token]
        node.is_terminal = True

    def allowed_next_tokens(self, prefix: List[int]) -> List[int]:
        node = self.root
        for token in prefix:
            if token not in node.children:
                return []
            node = node.children[token]
        return sorted(node.children.keys())

    def contains(self, token_seq: List[int]) -> bool:
        node = self.root
        for token in token_seq:
            if token not in node.children:
                return False
            node = node.children[token]
        return node.is_terminal


def build_sid_mappings(semantic_id_pt_path: str):
    """
    Returns:
        sid_to_items: Dict[Tuple[int, ...], List[int]]
        trie: PrefixTrie
        sid_width: int
        sid_vocab_size: int
    """
    obj = torch.load(semantic_id_pt_path, map_location="cpu")
    semantic_id = obj["semantic_id"].long()   # [N, L]
    item_id = obj["item_id"].long()           # [N]

    sid_to_items: Dict[Tuple[int, ...], List[int]] = {}
    trie = PrefixTrie()

    for i in range(semantic_id.size(0)):
        sid_tuple = tuple(int(x) for x in semantic_id[i].tolist())
        raw_item_id = int(item_id[i].item())

        if sid_tuple not in sid_to_items:
            sid_to_items[sid_tuple] = []
        sid_to_items[sid_tuple].append(raw_item_id)

    for sid_tuple in sid_to_items.keys():
        trie.insert(list(sid_tuple))

    for sid_tuple in sid_to_items.keys():
        sid_to_items[sid_tuple] = sorted(sid_to_items[sid_tuple])

    sid_width = int(semantic_id.size(1))
    sid_vocab_size = int(semantic_id.max().item()) + 1
    return sid_to_items, trie, sid_width, sid_vocab_size


def _encode_single_sample(
    model,
    input_ids: torch.Tensor,       # [1, S]
    attention_mask: torch.Tensor,  # [1, S]
    sid_width: int,
):
    encoder_layer_ids = model._build_encoder_layer_ids(
        input_ids=input_ids,
        sid_width=sid_width,
    )
    encoder_inputs_embeds = model.token_embeddings(
        input_ids=input_ids,
        layer_ids=encoder_layer_ids,
    )

    encoder_outputs = model.encoder(
        input_ids=None,
        attention_mask=attention_mask,
        inputs_embeds=encoder_inputs_embeds,
        return_dict=True,
    )
    return encoder_outputs.last_hidden_state


def _decode_next_token_log_probs(
    model,
    encoder_hidden_states: torch.Tensor,   # [1, S, D]
    encoder_attention_mask: torch.Tensor,  # [1, S]
    decoder_input_ids: torch.Tensor,       # [Bbeam, cur_len]
) -> torch.Tensor:
    """
    Returns:
        log_probs: [Bbeam, vocab_size] for the next token at current step
    """
    beam_size = decoder_input_ids.size(0)

    decoder_input_layer_ids = model._build_decoder_input_layer_ids(decoder_input_ids)
    decoder_inputs_embeds = model.token_embeddings(
        input_ids=decoder_input_ids,
        layer_ids=decoder_input_layer_ids,
    )

    decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long)

    expanded_encoder_hidden_states = encoder_hidden_states.expand(beam_size, -1, -1)
    expanded_encoder_attention_mask = encoder_attention_mask.expand(beam_size, -1)

    decoder_outputs = model.decoder(
        input_ids=None,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        encoder_hidden_states=expanded_encoder_hidden_states,
        encoder_attention_mask=expanded_encoder_attention_mask,
        return_dict=True,
    )

    prediction_layer_ids = model._build_prediction_layer_ids(
        batch_size=beam_size,
        seq_len=decoder_input_ids.size(1),
        device=decoder_input_ids.device,
    )

    logits = model._compute_logits(
        hidden_states=decoder_outputs.last_hidden_state,
        prediction_layer_ids=prediction_layer_ids,
    )  # [Bbeam, cur_len, V]

    next_token_logits = logits[:, -1, :]   # [Bbeam, V]
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    return log_probs



def _encode_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, sid_width: int):
    encoder_layer_ids = model._build_encoder_layer_ids(input_ids=input_ids, sid_width=sid_width)
    encoder_inputs_embeds = model.token_embeddings(input_ids=input_ids, layer_ids=encoder_layer_ids)
    encoder_outputs = model.encoder(
        input_ids=None,
        attention_mask=attention_mask,
        inputs_embeds=encoder_inputs_embeds,
        return_dict=True,
    )
    return encoder_outputs.last_hidden_state


# --- Added: Batched constrained beam search ---
@torch.no_grad()
def constrained_beam_search_batch(
        model,
        input_ids: torch.Tensor,  # [B, S]
        attention_mask: torch.Tensor,  # [B, S]
        sid_width: int,
        trie: PrefixTrie,
        beam_size: int,
        num_return_sequences: int,
) -> List[List[BeamCandidate]]:
    """
    Efficient batched prefix-constrained beam search.

    Returns:
        List[List[BeamCandidate]] with shape [batch_size, num_return_sequences].
    """
    batch_size = input_ids.size(0)
    device = input_ids.device

    # 1. Encode the full batch once.
    # encoder_hidden_states: [B, S, D]
    encoder_hidden_states = _encode_batch(model, input_ids, attention_mask, sid_width)

    # Initialize beams: each sample starts with one empty sequence and score 0.
    # current_beams structure: List[List[BeamCandidate]]
    current_beams: List[List[BeamCandidate]] = [[BeamCandidate([], 0.0)] for _ in range(batch_size)]

    for step in range(sid_width):
        all_step_candidates: List[List[BeamCandidate]] = [[] for _ in range(batch_size)]

        # To enable efficient tensorized decoding, flatten all active beams
        # across the batch into one list.
        # Example: batch_size=32, beam_size=20 -> up to 640 sequences.
        flat_decoder_input_ids = []
        flat_beam_indices = []  # Maps each flattened beam back to its original batch index.
        flat_parent_beam_objs = []

        for b_idx in range(batch_size):
            for beam_obj in current_beams[b_idx]:
                allowed = trie.allowed_next_tokens(beam_obj.tokens)
                if not allowed:
                    continue

                # Build decoder input as: [BOS] + generated prefix
                decoder_tokens = [model.bos_token_id] + beam_obj.tokens
                flat_decoder_input_ids.append(decoder_tokens)
                flat_beam_indices.append(b_idx)
                flat_parent_beam_objs.append((beam_obj, allowed))

        if not flat_decoder_input_ids:
            break

        # Convert flattened decoder inputs to a tensor.
        padded_input_ids = torch.tensor(flat_decoder_input_ids, device=device)  # [N_flat, cur_len]

        # Expand encoder states and masks to match the flattened beam dimension.
        # N_flat is the total number of valid beams in the current batch.
        curr_batch_indices = torch.tensor(flat_beam_indices, device=device)
        curr_encoder_hidden = encoder_hidden_states[curr_batch_indices]  # [N_flat, S, D]
        curr_encoder_mask = attention_mask[curr_batch_indices]  # [N_flat, S]

        # Run the decoder on all flattened beams at once.
        # Reuse the existing _decode_next_token_log_probs logic, but in batch form.
        decoder_input_layer_ids = model._build_decoder_input_layer_ids(padded_input_ids)
        decoder_inputs_embeds = model.token_embeddings(padded_input_ids, decoder_input_layer_ids)

        decoder_outputs = model.decoder(
            input_ids=None,
            attention_mask=torch.ones_like(padded_input_ids),
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=curr_encoder_hidden,
            encoder_attention_mask=curr_encoder_mask,
            return_dict=True,
        )

        prediction_layer_ids = model._build_prediction_layer_ids(
            batch_size=len(flat_decoder_input_ids),
            seq_len=padded_input_ids.size(1),
            device=device,
        )
        logits = model._compute_logits(decoder_outputs.last_hidden_state, prediction_layer_ids)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [N_flat, V]

        # Apply trie constraints and collect next-step candidates.
        for i, (parent_beam, allowed_tokens) in enumerate(flat_parent_beam_objs):
            b_idx = flat_beam_indices[i]

            # Only keep tokens allowed by the prefix trie.
            allowed_tensor = torch.tensor(allowed_tokens, device=device)
            current_log_probs = log_probs[i, allowed_tensor]

            for j, token_id in enumerate(allowed_tokens):
                new_score = parent_beam.score + current_log_probs[j].item()
                all_step_candidates[b_idx].append(
                    BeamCandidate(tokens=parent_beam.tokens + [int(token_id)], score=float(new_score))
                )

        # Keep top-k beams for each sample independently.
        for b_idx in range(batch_size):
            sorted_cands = sorted(all_step_candidates[b_idx], key=lambda x: x.score, reverse=True)
            current_beams[b_idx] = sorted_cands[:beam_size]

    # Final filtering: keep only complete sequences that exist in the trie,
    # then truncate to num_return_sequences.
    results = []
    for b_idx in range(batch_size):
        final = [b for b in current_beams[b_idx] if trie.contains(b.tokens)]
        results.append(sorted(final, key=lambda x: x.score, reverse=True)[:num_return_sequences])

    return results

@torch.no_grad()
def constrained_beam_search_single(
    model,
    input_ids: torch.Tensor,        # [1, S]
    attention_mask: torch.Tensor,   # [1, S]
    sid_width: int,
    trie: PrefixTrie,
    beam_size: int,
    num_return_sequences: int,
) -> List[BeamCandidate]:
    """
    Prefix-constrained beam search over valid semantic-ID sequences only.
    """
    device = input_ids.device
    encoder_hidden_states = _encode_single_sample(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        sid_width=sid_width,
    )

    beams = [BeamCandidate(tokens=[], score=0.0)]

    for step in range(sid_width):
        decoder_input_ids = []
        valid_prefixes = []

        for beam in beams:
            prefix_tokens = beam.tokens
            allowed = trie.allowed_next_tokens(prefix_tokens)
            if len(allowed) == 0:
                continue

            decoder_tokens = [model.bos_token_id] + prefix_tokens
            decoder_input_ids.append(decoder_tokens)
            valid_prefixes.append((beam, allowed))

        if len(decoder_input_ids) == 0:
            return []

        max_len = max(len(x) for x in decoder_input_ids)
        padded = torch.full(
            (len(decoder_input_ids), max_len),
            fill_value=model.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for i, seq in enumerate(decoder_input_ids):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        log_probs = _decode_next_token_log_probs(
            model=model,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            decoder_input_ids=padded,
        )  # [Bbeam, V]

        all_candidates: List[BeamCandidate] = []

        for i, (beam, allowed_tokens) in enumerate(valid_prefixes):
            allowed_tensor = torch.tensor(allowed_tokens, device=device, dtype=torch.long)
            allowed_scores = log_probs[i, allowed_tensor]  # [A]

            for j in range(allowed_tensor.size(0)):
                next_token = int(allowed_tensor[j].item())
                next_score = float(beam.score + allowed_scores[j].item())
                all_candidates.append(
                    BeamCandidate(tokens=beam.tokens + [next_token], score=next_score)
                )

        all_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)
        beams = all_candidates[:beam_size]

    final_beams = [b for b in beams if trie.contains(b.tokens)]
    final_beams = sorted(final_beams, key=lambda x: x.score, reverse=True)
    return final_beams[:num_return_sequences]

def beams_to_ranked_items_random_sampling(
    beams: List[BeamCandidate],
    sid_to_items: Dict[Tuple[int, ...], List[int]],
    max_items: int,
    collision_mode: str = "sample_one",
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    Expand ranked SID beams into ranked item ids.

    Args:
        beams:
            Ranked SID beams.
        sid_to_items:
            Mapping from SID tuple to colliding item ids.
        max_items:
            Maximum number of ranked items to return.
        collision_mode:
            - "sample_one":
                Randomly sample exactly one item from each predicted SID bucket.
            - "shuffle_all":
                Randomly shuffle all items inside each predicted SID bucket,
                then append them all.
            - "deterministic_all":
                Original behavior: append all items in their existing order
                (in your codebase this is sorted item_id order).
        rng:
            Optional random generator for reproducibility.
            If None, use the global random module.

    Returns:
        ranked_items: ranked item ids.
    """
    if rng is None:
        rng = random

    ranked_items: List[int] = []
    seen = set()

    for beam in beams:
        sid_tuple = tuple(beam.tokens)
        items = sid_to_items.get(sid_tuple, [])
        if not items:
            continue

        # Remove items that have already been added.
        available_items = [item_id for item_id in items if item_id not in seen]
        if not available_items:
            continue

        if collision_mode == "sample_one":
            candidate_items = [rng.choice(available_items)]

        elif collision_mode == "shuffle_all":
            candidate_items = list(available_items)
            rng.shuffle(candidate_items)

        elif collision_mode == "deterministic_all":
            candidate_items = available_items

        else:
            raise ValueError(
                f"Unsupported collision_mode={collision_mode}. "
                f"Choose from ['sample_one', 'shuffle_all', 'deterministic_all']."
            )

        for item_id in candidate_items:
            ranked_items.append(item_id)
            seen.add(item_id)
            if len(ranked_items) >= max_items:
                return ranked_items

    return ranked_items



def beams_to_ranked_items(
    beams: List[BeamCandidate],
    sid_to_items: Dict[Tuple[int, ...], List[int]],
    max_items: int,
    sample_one_per_sid: bool = True,
) -> List[int]:
    """
    Expand ranked SID beams into ranked item ids.

    Args:
        sample_one_per_sid:
            - True: for each SID, randomly sample one non-duplicated item
            - False: expand all items under that SID (original behavior)
    """
    ranked_items: List[int] = []
    seen = set()

    for beam in beams:
        sid_tuple = tuple(beam.tokens)
        items = sid_to_items.get(sid_tuple, [])

        if not items:
            continue

        if sample_one_per_sid:
            # --- random sampling ---
            candidate_items = list(items)
            random.shuffle(candidate_items)

            chosen_item_id = None
            for item_id in candidate_items:
                if item_id not in seen:
                    chosen_item_id = item_id
                    break

            if chosen_item_id is None:
                continue

            ranked_items.append(chosen_item_id)
            seen.add(chosen_item_id)

            if len(ranked_items) >= max_items:
                return ranked_items

        else:
            # --- take all items in sid ---
            for item_id in items:
                if item_id not in seen:
                    ranked_items.append(item_id)
                    seen.add(item_id)

                    if len(ranked_items) >= max_items:
                        return ranked_items

    return ranked_items

def recall_at_k(ranked_items: List[int], target_item_id: int, k: int) -> float:
    topk = ranked_items[:k]
    return 1.0 if target_item_id in topk else 0.0


def ndcg_at_k(ranked_items: List[int], target_item_id: int, k: int) -> float:
    topk = ranked_items[:k]
    for idx, item_id in enumerate(topk):
        if item_id == target_item_id:
            rank = idx + 1
            return 1.0 / math.log2(rank + 1)
    return 0.0