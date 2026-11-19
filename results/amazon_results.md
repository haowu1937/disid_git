| # | Setting |  Train | Finetune | Test | Model | Recall@10 | NDCG@10 |
|---:|---|---|---|---|---|---:|---:|
| **No pretraining** |
| **Scientific -> Scientific** |
| *Hao* | No pretraining | S | - | S | RQVAE GRID T5 | 0.07107 | 0.04653 |
| 1 | No pretraining (`ours`) | S | - | S| RQVAE + TIGER | 0.0768 | 0.0473 |
| 2 |No pretraining (`ours`) | S| - | S| Disen-RQVAE + TIGER | 0.0862 | 0.0515 | 
| 3 |No pretraining (`ours`) | S| - | S| Disen-DBS-RQVAE + TIGER | 0.0805 | 0.0482|
| 3 |No pretraining (`ours`) | S| - | S| SasRec | 0.09476 | 0.05489|
| **No pretraining** |
| **FHCKM -> FHCKM**|
| 4 | No pretraining (`graphstructuredcodes`) | FHCKM | - | FHCKM | RQVAE + TIGER | 0.0146| 0.0085 |
| 5 |No pretraining (`graphstructuredcodes`) | FHCKM | - | FHCKM | Disen-RQVAE + TIGER | 0.0203 | 0.0126 |
| 6 |No pretraining (`graphstructuredcodes`) | FHCKM | - | FHCKM | Disen-DBS-RQVAE + TIGER | 0.0183 | 0.0113 |
| 6 |No pretraining (`graphstructuredcodes`) | FHCKM | - | FHCKM | SASRec | 0.07047 | 0.04932 |
| **With pretraining**|
| **FHCKM -> Scientific** |
| 7 | With pretraining | FHCKM | TN | TN | RQVAE + TIGER | 0.0529 | 0.0288 |
| 8 | With pretraining | FHCKM | TN | TN | Disen-RQVAE + TIGER | 0.0539  | 0.0290 | 
| 9 | With pretraining | FHCKM | TN | TN | Disen-DBS-RQVAE + TIGER | 0.0557  | 0.0305 |
| 9 | With pretraining | FHCKM | TN | TN | SASRec | 0.030916 | 0.01712 |


#### 1.2 NineRec data, Bili10K -> TN

| # | Setting | Train | Finetune | Test | Model | Recall@10 | NDCG@10 |
|---:|---|---|---|---|---|---:|---:|
| **No pretraining** | | | | | | | |
| **TN -> TN** | | | | | | | |
| *Hao* | No pretraining | TN | - | TN | Standard RQVAE | 0.111276 | 0.064822 |
| *Hao* | No pretraining | TN | - | TN | Disen RQVAE | 0.124487 | 0.072578 |
| 0 | No pretraining | TN | - | TN | SASRec | **0.257583** | **0.147570** |
| 1 | No pretraining (`ours`) | TN | - | TN | RQVAE + TIGER | 0.1431 | 0.0817 |
| 2 | No pretraining (`ours`) | TN | - | TN | Disen-RQVAE + TIGER | 0.1386 | 0.0797 |
| 3 | No pretraining (`ours`) | TN | - | TN | Disen-DBS-RQVAE + TIGER | 0.1404 | 0.0828 |
| **No pretraining** | | | | | | | |
| **Bili -> Bili** | | | | | | | |
| 4 | No pretraining (`ours`) | Bili | - | Bili | RQVAE + TIGER | 0.0029 | **0.0014** |
| 5 | No pretraining (`ours`) | Bili | - | Bili | Disen-RQVAE + TIGER | 0.0027 | 0.0011 |
| 6 | No pretraining (`ours`) | Bili | - | Bili | Disen-DBS-RQVAE + TIGER | **0.0030** | 0.0012 |
| 7 | No pretraining (`ours`) | Bili | - | Bili | SASRec  | 0.00290 | 0.001486 |
| **With pretraining** | | | | | | | |
| **Bili -> TN** | | | | | | | |
| *Hao* | Has pretraining | Bili | TN | TN | RQVAE | 0.114923 | 0.069088 |
| *Hao* | Has pretraining | Bili | TN | TN | Disen RQVAE | 0.113289 | 0.070111 |
| 8 | Has pretraining | Bili_10K | TN | TN | SASRec, LR=0.003 | 0.244273 | 0.140729 |
| 9 | Has pretraining | Bili_10K | TN | TN | SASRec, LR=0.0015 | 0.251299 | **0.148383** |
| 10 | Has pretraining | Bili_10K | TN | TN | SASRec, LR=0.001 | **0.253674** | 0.146313 |
| 11 | Has pretraining (`ours`) | Bili | TN | TN | RQVAE | 0.0903 | 0.0509 |
| 12| Has pretraining (`ours`) | Bili | TN | TN | Disen-RQVAE | 0.0925 | 0.0527 |
| 13 | Has pretraining (`ours`) | Bili | TN | TN | Disen-DBS-RQVAE | 0.0923 | 0.0525 |
