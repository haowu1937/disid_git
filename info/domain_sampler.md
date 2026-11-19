- **Motivation**
  - In multi-domain training, large domains can dominate mini-batches because they contain far more items, causing the codebook to bias toward source-domain statistics.
  - `src.data_loading.multi_domain_dataset.DomainBalancedBatchSampler` increases the exposure of small target domains, improving cross-domain transfer while preserving domain-specific representation quality.

- **Implementation**
  - Build per-domain index pools from `dataset.domain_ids`, i.e., collect all sample indices belonging to each domain.
  - Let \(n_d\) be the item count of domain \(d\), and define domain sampling probability as  
    \[
    p_d=(1-\rho)\frac{n_d}{\sum_k n_k}+\rho\frac{n_d^{-\tau}}{\sum_k n_k^{-\tau}}
    \]
  - For each mini-batch, first sample domain labels according to \(p_d\), then sample one item index from the corresponding domain pool.
  - Return the sampled global indices to the standard `DataLoader`, without changing the existing dataset or model interface.
  - The balancing strength is modularly controlled by input arguments such as `mode`, `rho`, `tau`, and `replacement`, making ablation easy.