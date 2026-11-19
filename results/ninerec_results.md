| Model | Dataset | Recall@5 | Recall@10 |
|---|---|---:|---:|
| dsdt | all | 0.03817 | 0.05116 |
| dsdt | Bili_50K | 0.002020 | 0.00356 |
| dsdt | QB | 0.1331 | 0.16956 |
| dsdt | TN | 0.04438 | 0.06511 |
| dt | all | 0.029511 | 0.041725 |
| dt | Bili_50K | 0.002039 | 0.00356 |
| dt | QB | 0.102866 | 0.141237 |
| dt | TN | 0.033150 | 0.048884 |
| rqvae | all | 0.022915 | 0.033173 |
| rqvae | Bili | 0.002379 | 0.00406 |
| rqvae | QB | 0.07493 | 0.1060 |
| rqvae | TN | 0.02810 | 0.0413 |
| rqkmeans| all | 0.02571 | 0.037016|
| rqkmeans| Bili | 0.0024| 0.00398|
| rqkmeans| QB | 0.08774| 0.1232|
| rqkmeans| TN | 0.02899| 0.04314|
| opq| all | 0.02241 | 0.0330|
| opq| Bili | 0.0022| 0.0036|
| opq| QB | 0.0923| 0.1109|
| opq| TN | 0.02543| 0.04105|

Introduce a new module to enhance domain-shared information:
**Distributional Regularization:**


Add an alignment loss to encourage the distribution of different domains to be similar in domain-shared layers:

Let $P_d(z^{shared})$ be the prefix distribution of domain $d$. A alignment loss can added:

$$\mathcal{L}_{\text{align}} = \operatorname{Align}\left(P_{d_a}(z^{shared}), P_{d_b}(z^{shared})\right),$$

where $\operatorname{Align}(\cdot, \cdot)$ can be implemented using JS divergence, or a contrastive objective.

Populated experiment results of multi-domain training (train on bili+qb+tn, direct evaluate on individual domains):

- Baselines:
  - rqvae
  - rqkmeans
  - opq

- dt: Distangled RQVAE
- ds: Domain balance sampling
- sv: shared vocabulary in domain specific layer
- spdr: add distributional regularization to encourage domain shared information.


| Dataset  | Model    | Recall@5 | Recall@10 |
| -------- | -------- | -------: | --------: |
| all      | dsdt     |  0.03817 |   0.05116 |
| all      | dt       | 0.029511 |  0.041725 |
| all      | dt_sv    | 0.027885 |  0.040576 |
| all      | dt_spdr  | 0.023700 |  0.033469 |
| all      | rqvae    | 0.022915 |  0.033173 |
| all      | rqkmeans |  0.02571 |  0.037016 |
| all      | opq      |  0.02241 |    0.0330 |
| Bili_50K | dsdt     | 0.002020 |   0.00356 |
| Bili_50K | dt       | 0.002039 |   0.00356 |
| Bili_50K | dt_sv    | 0.002380 |  0.004060 |
| Bili_50K | dt_spdr  | 0.002160 |  0.003800 |
| Bili_50K | rqvae    | 0.002379 |   0.00406 |
| Bili_50K | rqkmeans |   0.0024 |   0.00398 |
| Bili_50K | opq      |   0.0022 |    0.0036 |
| QB       | dsdt     |   0.1331 |   0.16956 |
| QB       | dt       | 0.102866 |  0.141237 |
| QB       | dt_sv    | 0.093556 |  0.133563 |
| QB       | dt_spdr  | 0.079731 |  0.110879 |
| QB       | rqvae    |  0.07493 |    0.1060 |
| QB       | rqkmeans |  0.08774 |    0.1232 |
| QB       | opq      |   0.0923 |    0.1109 |
| TN       | dsdt     |  0.04438 |   0.06511 |
| TN       | dt       | 0.033150 |  0.048884 |
| TN       | dt_sv    | 0.033398 |  0.049379 |
| TN       | dt_spdr  | 0.027856 |  0.038989 |
| TN       | rqvae    |  0.02810 |    0.0413 |
| TN       | rqkmeans |  0.02899 |   0.04314 |
| TN       | opq      |  0.02543 |   0.04105 |
