#bin bash

python -m src.training.train_rqkmeans --config configs/train_sid_ninerec_qwen_rqkmeans.yaml
python -m src.generation.generate_semantic_ids --config configs/generate_sid_ninerec_qwen_rqkmeans.yaml
cp /home/hao/FRec/DisenID/sid_outputs/ninerec/qwen_bili500k_TN_QB_rqkmeans/sid_rqkmeans.pt /home/shared/ninerec/sids/
