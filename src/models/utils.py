from typing import Dict, Any
import torch


from src.data_loading.multi_domain_dataset import DomainBalancedBatchSampler
from torch.utils.data import DataLoader

from src.models.rqvae import RQVAE
from src.models.dis_rqvae import DisRQVAE
from src.models.dis_rqvae_shared_vocab import DisRQVAE_SV
from src.models.dis_rqvae_spdr import DisRQVAE_SPDR
from src.models.dis_rqvae_spdr_dscb import DisRQVAE_SPDR_DSCB

from src.models.rq_kmeans import RQKMeans

def build_train_dataloader(
    dataset,
    config: Dict[str, Any],
    shuffle: bool,
):
    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    num_workers = int(training_cfg.get("num_workers", 4))
    pin_memory = bool(training_cfg.get("pin_memory", torch.cuda.is_available()))

    domain_balance_cfg = training_cfg.get("domain_balance", {})
    use_domain_balance = bool(domain_balance_cfg.get("enabled", False))

    batch_sampler = None

    print("=" * 80)
    print("[DataLoader] Building training dataloader")
    print(f"[DataLoader] dataset              = {type(dataset).__name__}")
    print(f"[DataLoader] num_samples          = {len(dataset)}")
    print(f"[DataLoader] batch_size           = {batch_size}")
    print(f"[DataLoader] num_workers          = {num_workers}")
    print(f"[DataLoader] pin_memory           = {pin_memory}")
    print(f"[DataLoader] requested shuffle    = {shuffle}")
    print(f"[DataLoader] domain_balance       = {use_domain_balance}")

    if use_domain_balance:
        if not hasattr(dataset, "domain_ids"):
            raise ValueError(
                "DomainBalancedBatchSampler requires dataset.domain_ids, "
                f"but got dataset={type(dataset).__name__} without this attribute."
            )

        drop_last = bool(domain_balance_cfg.get("drop_last", False))
        mode = str(domain_balance_cfg.get("mode", "temperature"))
        rho = float(domain_balance_cfg.get("rho", 0.7))
        tau = float(domain_balance_cfg.get("tau", 0.5))
        replacement = bool(domain_balance_cfg.get("replacement", True))
        num_batches = domain_balance_cfg.get("num_batches", None)
        seed = int(training_cfg.get("seed", 42))

        print("[DataLoader] sampler              = DomainBalancedBatchSampler")
        print(f"[DataLoader] sampler.drop_last    = {drop_last}")
        print(f"[DataLoader] sampler.mode         = {mode}")
        print(f"[DataLoader] sampler.rho          = {rho}")
        print(f"[DataLoader] sampler.tau          = {tau}")
        print(f"[DataLoader] sampler.replacement  = {replacement}")
        print(f"[DataLoader] sampler.num_batches  = {num_batches}")
        print(f"[DataLoader] sampler.seed         = {seed}")
        print("[DataLoader] note                 = shuffle is ignored because batch_sampler is used")

        batch_sampler = DomainBalancedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            mode=mode,
            rho=rho,
            tau=tau,
            replacement=replacement,
            num_batches=num_batches,
            seed=seed,
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    else:
        print("[DataLoader] sampler              = default DataLoader sampler")
        print(f"[DataLoader] shuffle              = {shuffle}")
        print("[DataLoader] drop_last            = False")
        print("[DataLoader] note                 = domain-balanced sampling is disabled")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    print(f"[DataLoader] num_batches          = {len(dataloader)}")
    print("=" * 80)
    print()

    return dataloader, batch_sampler

def build_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    data_cfg = config["data"]

    model_type = model_cfg["type"]

    input_dim = int(model_cfg["input_dim"])
    codebook_size = int(model_cfg["codebook_size"])
    num_layers = int(model_cfg["num_layers"])

    if model_type == "rqkmeans":
        model = RQKMeans(
            input_dim=input_dim,
            codebook_size=codebook_size,
            num_layers=num_layers,
            keep_empty_codes=bool(model_cfg.get("keep_empty_codes", True)),
        )

    elif model_type == "baseline":
        hidden_dim = int(model_cfg["hidden_dim"])

        commitment_weight = float(model_cfg.get("commitment_weight", 0.25))
        codebook_weight = float(model_cfg.get("codebook_weight", 1.0))
        normalize_codebook_loss = bool(model_cfg.get("normalize_codebook_loss", True))
        encoder_hidden_dims = model_cfg.get("encoder_hidden_dims", None)
        decoder_hidden_dims = model_cfg.get("decoder_hidden_dims", None)
        dropout = float(model_cfg.get("dropout", 0.0))
        bias = bool(model_cfg.get("bias", True))

        model = RQVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            num_layers=num_layers,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            normalize_codebook_loss=normalize_codebook_loss,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            bias=bias,
            dropout=dropout,
        )

    elif model_type == "disentangle":
        hidden_dim = int(model_cfg["hidden_dim"])

        commitment_weight = float(model_cfg.get("commitment_weight", 0.25))
        codebook_weight = float(model_cfg.get("codebook_weight", 1.0))
        normalize_codebook_loss = bool(model_cfg.get("normalize_codebook_loss", True))
        alpha = model_cfg.get("alpha", 0.02)
        encoder_hidden_dims = model_cfg.get("encoder_hidden_dims", None)
        decoder_hidden_dims = model_cfg.get("decoder_hidden_dims", None)
        dropout = float(model_cfg.get("dropout", 0.0))
        bias = bool(model_cfg.get("bias", True))

        num_shared_layers = int(model_cfg["num_shared_layers"])
        num_domains = int(data_cfg["num_domains"])

        variant = str(model_cfg.get("variant", "original")).lower()

        common_kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            num_layers=num_layers,
            num_shared_layers=num_shared_layers,
            num_domains=num_domains,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            normalize_codebook_loss=normalize_codebook_loss,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            alpha=alpha,
            bias=bias,
            dropout=dropout,
        )

        if variant == "original":
            model = DisRQVAE(**common_kwargs)

        elif variant in {"shared_vocab", "sv"}:
            model = DisRQVAE_SV(**common_kwargs)

        elif variant == "spdr":
            model = DisRQVAE_SPDR(
                **common_kwargs,
                spdr_align_weight=float(model_cfg.get("spdr_align_weight", 0.0)),
                spdr_separation_weight=float(model_cfg.get("spdr_separation_weight", 0.0)),
                spdr_sep_margin=float(model_cfg.get("spdr_sep_margin", 0.1)),
                spdr_temperature=float(model_cfg.get("spdr_temperature", 0.1)),
                spdr_eps=float(model_cfg.get("spdr_eps", 1e-8)),
            )
        elif variant == "spdr_ds": #spdr based on domain specific codebook
            model = DisRQVAE_SPDR_DSCB(
                **common_kwargs,
                spdr_align_weight=float(model_cfg.get("spdr_align_weight", 0.0)),
                spdr_separation_weight=float(model_cfg.get("spdr_separation_weight", 0.0)),
                spdr_sep_margin=float(model_cfg.get("spdr_sep_margin", 0.1)),
                spdr_temperature=float(model_cfg.get("spdr_temperature", 0.1)),
                spdr_eps=float(model_cfg.get("spdr_eps", 1e-8)),
            )

        else:
            raise ValueError(
                f"Unknown disentangle variant: {variant}. "
                "Expected one of: original, shared_vocab, sv, spdr."
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    return model