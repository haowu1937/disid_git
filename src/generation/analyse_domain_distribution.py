#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check whether the last two semantic-ID layers are domain-private.

Expected input (.pt) format inferred from semantic_ids_all.pt:
{
    "domain_id": LongTensor [N] or int,
    "item_id": LongTensor [N],
    "semantic_id": LongTensor [N, L],
    optional: "config", "model_type", "num_layers", "ckpt_path", "ckpt_meta", ...
}

Important:
- This file is NOT the raw codebook centroid tensor.
- It is an item -> semantic_id assignment dump.
- So the most direct way to check "domain private" is to study:
    domain_id  <->  semantic_id[:, -2:]
"""

import json
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch


# -------------------------
# IO
# -------------------------
def load_semantic_pt(pt_path: str):
    obj = torch.load(pt_path, map_location="cpu")

    for k in ["domain_id", "item_id", "semantic_id"]:
        if k not in obj:
            raise ValueError(f"Missing key '{k}' in {pt_path}. Keys: {list(obj.keys())}")

    domain_id = obj["domain_id"]
    item_id = obj["item_id"]
    semantic_id = obj["semantic_id"]

    # allow single-domain dump
    if isinstance(domain_id, int):
        n = int(item_id.shape[0])
        domain_id = torch.full((n,), fill_value=domain_id, dtype=torch.long)

    if not torch.is_tensor(domain_id) or domain_id.dtype != torch.long or domain_id.dim() != 1:
        raise ValueError("domain_id must be int or torch.LongTensor [N]")
    if not torch.is_tensor(item_id) or item_id.dtype != torch.long or item_id.dim() != 1:
        raise ValueError("item_id must be torch.LongTensor [N]")
    if not torch.is_tensor(semantic_id) or semantic_id.dtype != torch.long or semantic_id.dim() != 2:
        raise ValueError("semantic_id must be torch.LongTensor [N, L]")

    if domain_id.shape[0] != item_id.shape[0] or item_id.shape[0] != semantic_id.shape[0]:
        raise ValueError(
            f"Length mismatch: domain_id={tuple(domain_id.shape)}, "
            f"item_id={tuple(item_id.shape)}, semantic_id={tuple(semantic_id.shape)}"
        )

    meta = {
        k: obj.get(k)
        for k in ["ckpt_path", "model_type", "num_layers", "config", "ckpt_meta"]
    }
    return domain_id.cpu(), item_id.cpu(), semantic_id.cpu(), meta


def infer_codebook_width(
    semantic_id: torch.Tensor,
    meta: dict,
    hardcoded_width: Optional[int] = None,
) -> int:
    if hardcoded_width is not None:
        return int(hardcoded_width)

    config = meta.get("config", None)
    if isinstance(config, dict):
        model_cfg = config.get("model", {})
        for key in ["codebook_size", "num_embeddings", "vocab_size"]:
            if key in model_cfg:
                try:
                    return int(model_cfg[key])
                except Exception:
                    pass

    # safest fallback
    return int(semantic_id.max().item()) + 1


def resolve_domain_names(
    unique_domain_ids: List[int],
    config: Optional[dict] = None,
    fallback_domain_names: Optional[List[str]] = None,
) -> Dict[int, str]:
    id2name: Dict[int, str] = {}

    # 1) Best effort from config.data.domain_configs
    if isinstance(config, dict):
        data_cfg = config.get("data", {})
        domain_cfgs = data_cfg.get("domain_configs", None)
        if isinstance(domain_cfgs, list):
            for idx, dc in enumerate(domain_cfgs):
                if not isinstance(dc, dict):
                    continue
                name = dc.get("name", None)
                did = dc.get("domain_id", idx)
                if name is None:
                    continue
                try:
                    did = int(did)
                except Exception:
                    continue
                id2name[did] = str(name)

    # 2) Fallback by position
    if fallback_domain_names is not None:
        for idx, name in enumerate(fallback_domain_names):
            if idx not in id2name:
                id2name[idx] = str(name)

    # 3) Final fallback
    for did in unique_domain_ids:
        if did not in id2name:
            id2name[did] = f"domain_{did}"

    return id2name


# -------------------------
# Math helpers
# -------------------------
def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.to(torch.float64)
    p = p[p > 0]
    if p.numel() == 0:
        return 0.0
    return float(-(p * torch.log2(p.clamp_min(eps))).sum().item())


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.to(torch.float64)
    q = q.to(torch.float64)
    m = 0.5 * (p + q)

    def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mask = a > 0
        a = a[mask]
        b = b[mask]
        return (a * torch.log2(a.clamp_min(eps) / b.clamp_min(eps))).sum()

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def counts_to_probs(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(torch.float64)
    denom = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    return counts / denom


# -------------------------
# Statistics
# -------------------------
def build_domain_feature_counts(
    domain_id: torch.Tensor,
    feature_id: torch.Tensor,
    num_buckets: int,
) -> Tuple[List[int], torch.Tensor]:
    unique_domains = sorted(int(x) for x in torch.unique(domain_id).tolist())
    counts = torch.zeros((len(unique_domains), num_buckets), dtype=torch.float64)

    for row, did in enumerate(unique_domains):
        vals = feature_id[domain_id == did]
        counts[row] = torch.bincount(vals, minlength=num_buckets).to(torch.float64)

    return unique_domains, counts


def pairwise_distribution_stats(
    counts: torch.Tensor,
    domain_names: List[str],
) -> Dict[str, object]:
    probs = counts_to_probs(counts)
    supports = counts > 0

    pair_details = []
    jsd_list = []
    jaccard_list = []

    for i, j in combinations(range(counts.size(0)), 2):
        jsd_val = js_divergence(probs[i], probs[j])
        inter = int((supports[i] & supports[j]).sum().item())
        union = int((supports[i] | supports[j]).sum().item())
        jaccard = (inter / union) if union > 0 else 0.0

        jsd_list.append(jsd_val)
        jaccard_list.append(jaccard)
        pair_details.append(
            {
                "domain_a": domain_names[i],
                "domain_b": domain_names[j],
                "jsd": jsd_val,
                "support_intersection": inter,
                "support_union": union,
                "support_jaccard": jaccard,
            }
        )

    out = {
        "avg_pairwise_jsd": float(sum(jsd_list) / len(jsd_list)) if jsd_list else 0.0,
        "avg_support_jaccard": float(sum(jaccard_list) / len(jaccard_list)) if jaccard_list else 0.0,
        "pairwise_details": pair_details,
    }
    return out


def feature_domain_predictability_stats(counts: torch.Tensor) -> Dict[str, float]:
    """
    counts: [D, B]
        D domains
        B feature buckets (single token or token-pair bucket)

    Returns:
      - majority_purity: use feature bucket -> majority domain to classify each item
      - normalized_purity_gain: purity improvement over always predicting the global majority domain
      - domain_nmi: MI(domain; feature) / H(domain)
    """
    counts = counts.to(torch.float64)
    domain_totals = counts.sum(dim=1)
    bucket_totals = counts.sum(dim=0)
    n = float(domain_totals.sum().item())

    if n <= 0:
        return {
            "majority_purity": 0.0,
            "global_majority_baseline": 0.0,
            "normalized_purity_gain": 0.0,
            "domain_nmi": 0.0,
        }

    # 1) bucket -> majority-domain classifier purity
    bucket_major = counts.max(dim=0).values
    majority_purity = float(bucket_major.sum().item() / n)

    global_majority_baseline = float(domain_totals.max().item() / n)
    if global_majority_baseline < 1.0:
        normalized_purity_gain = (majority_purity - global_majority_baseline) / (1.0 - global_majority_baseline)
    else:
        normalized_purity_gain = 0.0

    normalized_purity_gain = float(max(0.0, min(1.0, normalized_purity_gain)))

    # 2) NMI(domain; feature) / H(domain)
    p_domain = domain_totals / n
    h_domain = entropy_from_probs(p_domain)

    nonzero = bucket_totals > 0
    if int(nonzero.sum().item()) == 0 or h_domain <= 0:
        domain_nmi = 0.0
    else:
        nz_counts = counts[:, nonzero]  # [D, B_nonzero]
        nz_bucket_totals = bucket_totals[nonzero]  # [B_nonzero]
        p_bucket = nz_bucket_totals / n

        p_domain_given_bucket = nz_counts / nz_bucket_totals.unsqueeze(0).clamp_min(1.0)
        h_domain_given_bucket = -(p_domain_given_bucket * torch.log2(p_domain_given_bucket.clamp_min(1e-12))).sum(dim=0)
        h_cond = float((p_bucket * h_domain_given_bucket).sum().item())

        mi = max(0.0, h_domain - h_cond)
        domain_nmi = float(mi / h_domain)

    return {
        "majority_purity": majority_purity,
        "global_majority_baseline": global_majority_baseline,
        "normalized_purity_gain": normalized_purity_gain,
        "domain_nmi": domain_nmi,
    }


def summarize_feature(
    domain_id: torch.Tensor,
    feature_id: torch.Tensor,
    num_buckets: int,
    id2name: Dict[int, str],
) -> Dict[str, object]:
    unique_domains, counts = build_domain_feature_counts(domain_id, feature_id, num_buckets)
    domain_names = [id2name[did] for did in unique_domains]

    pairwise_stats = pairwise_distribution_stats(counts, domain_names)
    pred_stats = feature_domain_predictability_stats(counts)

    per_domain = []
    probs = counts_to_probs(counts)
    supports = counts > 0
    for i, did in enumerate(unique_domains):
        per_domain.append(
            {
                "domain_id": did,
                "domain_name": id2name[did],
                "num_items": int(counts[i].sum().item()),
                "used_buckets": int(supports[i].sum().item()),
                "coverage": float(supports[i].sum().item() / num_buckets),
                "entropy_bits": entropy_from_probs(probs[i]),
            }
        )

    out = {
        "num_domains": len(unique_domains),
        "num_buckets": int(num_buckets),
        "per_domain": per_domain,
    }
    out.update(pairwise_stats)
    out.update(pred_stats)
    return out


def build_joint_pair_id(
    last_two_tokens: torch.Tensor,
    codebook_width: int,
) -> torch.Tensor:
    if last_two_tokens.dim() != 2 or last_two_tokens.size(1) != 2:
        raise ValueError("last_two_tokens must be [N, 2]")
    return last_two_tokens[:, 0] * codebook_width + last_two_tokens[:, 1]


def build_final_private_score(
    layer_a_stats: Dict[str, object],
    layer_b_stats: Dict[str, object],
    joint_stats: Dict[str, object],
) -> float:
    """
    Heuristic score in [0, 1], higher means "more domain-private".

    Combines:
    - pairwise JSD of last layer-2
    - pairwise JSD of last layer-1
    - pairwise JSD of joint(last2)
    - domain NMI of joint(last2)
    - normalized purity gain of joint(last2)
    - 1 - support_jaccard of joint(last2)
    """
    parts = [
        float(layer_a_stats["avg_pairwise_jsd"]),
        float(layer_b_stats["avg_pairwise_jsd"]),
        float(joint_stats["avg_pairwise_jsd"]),
        float(joint_stats["domain_nmi"]),
        float(joint_stats["normalized_purity_gain"]),
        float(1.0 - float(joint_stats["avg_support_jaccard"])),
    ]
    return float(sum(parts) / len(parts))


# -------------------------
# Pretty print
# -------------------------
def print_feature_block(title: str, stats: Dict[str, object]) -> None:
    print("=" * 100)
    print(title)
    print("-" * 100)
    print(f"num_domains               : {stats['num_domains']}")
    print(f"num_buckets               : {stats['num_buckets']}")
    print(f"avg_pairwise_jsd          : {stats['avg_pairwise_jsd']:.6f}   (higher -> more domain-separated)")
    print(f"avg_support_jaccard       : {stats['avg_support_jaccard']:.6f}   (lower  -> more support-disjoint)")
    print(f"majority_purity           : {stats['majority_purity']:.6f}   (higher -> feature predicts domain better)")
    print(f"global_majority_baseline  : {stats['global_majority_baseline']:.6f}")
    print(f"normalized_purity_gain    : {stats['normalized_purity_gain']:.6f}   (higher -> better)")
    print(f"domain_nmi               : {stats['domain_nmi']:.6f}   (higher -> feature carries more domain info)")
    print()

    print("Per-domain stats:")
    for row in stats["per_domain"]:
        print(
            f"  {row['domain_name']:>12s} | "
            f"items={row['num_items']:>8d} | "
            f"used={row['used_buckets']:>6d} | "
            f"coverage={row['coverage']:.6f} | "
            f"entropy_bits={row['entropy_bits']:.6f}"
        )
    print()

    print("Pairwise domain details:")
    for row in stats["pairwise_details"]:
        print(
            f"  {row['domain_a']:>12s} vs {row['domain_b']:<12s} | "
            f"JSD={row['jsd']:.6f} | "
            f"support_jaccard={row['support_jaccard']:.6f} | "
            f"inter={row['support_intersection']:<6d} union={row['support_union']:<6d}"
        )
    print()


def save_json(obj: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# Main
# -------------------------
def main():
    pt_path = "/home/hao/FRec/DisenID/codebook/ninerec/semantic_ids_all.pt"
    json_out_path = "/home/hao/FRec/DisenID/codebook/ninerec/domain_private_score.json"

    # If you know the codebook width, set it here.
    # If not, set None and the script will try:
    #   config.model.codebook_size -> fallback to semantic_id.max()+1
    codebook_width = 256

    # Only for prettier printing if config does not contain domain names.
    fallback_domain_names = ["QB", "TN", "Bili"]

    domain_id, item_id, semantic_id, meta = load_semantic_pt(pt_path)

    n = int(semantic_id.size(0))
    l = int(semantic_id.size(1))
    if l < 2:
        raise ValueError(f"Need at least 2 layers, but got semantic_id shape [N, {l}]")

    codebook_width = infer_codebook_width(
        semantic_id=semantic_id,
        meta=meta,
        hardcoded_width=codebook_width,
    )

    unique_domain_ids = sorted(int(x) for x in torch.unique(domain_id).tolist())
    if len(unique_domain_ids) < 2:
        raise ValueError(
            f"Only found {len(unique_domain_ids)} unique domain(s): {unique_domain_ids}. "
            f"Domain-private analysis needs >= 2 domains."
        )

    id2name = resolve_domain_names(
        unique_domain_ids=unique_domain_ids,
        config=meta.get("config", None),
        fallback_domain_names=fallback_domain_names,
    )

    print("Loaded:", pt_path)
    print(f"Total items           : {n}")
    print(f"Num layers            : {l}")
    print(f"Codebook width        : {codebook_width}")
    print(f"Unique domains        : {[(did, id2name[did]) for did in unique_domain_ids]}")
    print(f"Model type            : {meta.get('model_type', None)}")
    print(f"Checkpoint            : {meta.get('ckpt_path', None)}")
    print()

    # use last two layers as "domain_private" layers
    layer_a = semantic_id[:, l - 2]   # second last
    layer_b = semantic_id[:, l - 1]   # last
    joint_last2 = build_joint_pair_id(semantic_id[:, l - 2:l], codebook_width)

    layer_a_stats = summarize_feature(
        domain_id=domain_id,
        feature_id=layer_a,
        num_buckets=codebook_width,
        id2name=id2name,
    )
    layer_b_stats = summarize_feature(
        domain_id=domain_id,
        feature_id=layer_b,
        num_buckets=codebook_width,
        id2name=id2name,
    )
    joint_stats = summarize_feature(
        domain_id=domain_id,
        feature_id=joint_last2,
        num_buckets=codebook_width * codebook_width,
        id2name=id2name,
    )

    private_score_v1 = build_final_private_score(
        layer_a_stats=layer_a_stats,
        layer_b_stats=layer_b_stats,
        joint_stats=joint_stats,
    )

    print_feature_block(f"Layer {l - 2} (second last private layer)", layer_a_stats)
    print_feature_block(f"Layer {l - 1} (last private layer)", layer_b_stats)
    print_feature_block(f"Joint pair of last two layers: (layer {l - 2}, layer {l - 1})", joint_stats)

    print("=" * 100)
    print("Final summary")
    print("-" * 100)
    print(f"private_score_v1      : {private_score_v1:.6f}   (higher -> more domain-private)")
    print()
    print("Interpretation:")
    print("1) avg_pairwise_jsd higher  -> different domains use clearly different token distributions")
    print("2) domain_nmi higher        -> the last two layers contain more domain identity information")
    print("3) normalized_purity_gain higher -> last-two tokens can predict domain much better")
    print("4) avg_support_jaccard lower -> different domains occupy more different token/pair regions")
    print()
    print("Recommended comparison:")
    print("- run the same script on normal rqvae")
    print("- run the same script on domain-specific rqvae")
    print("- compare mainly: joint(last2) JSD / NMI / purity_gain / private_score_v1")
    print()

    out = {
        "input_pt_path": pt_path,
        "num_items": n,
        "num_layers": l,
        "codebook_width": codebook_width,
        "unique_domains": [{"domain_id": did, "domain_name": id2name[did]} for did in unique_domain_ids],
        "model_type": meta.get("model_type", None),
        "ckpt_path": meta.get("ckpt_path", None),
        "layer_second_last": layer_a_stats,
        "layer_last": layer_b_stats,
        "joint_last_two_layers": joint_stats,
        "private_score_v1": private_score_v1,
    }

    save_json(out, json_out_path)
    print(f"Saved JSON summary to: {json_out_path}")


if __name__ == "__main__":
    main()