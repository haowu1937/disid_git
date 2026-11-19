#!/bin/bash
set -euo pipefail

python -m src.training.train_opq_sid --config configs/train_sid_ninerec_qwen_opq.yaml
