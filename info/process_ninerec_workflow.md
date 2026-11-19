1. generate qwen/clip embedding for items:
cmd: [build_ninerec_qwen_embedding.py](../preprocess/build_ninerec_qwen_embedding.py) ~/FRec/DisenID/data/TN/TN_item.csv
2. build tfrecords from behaviour csv
cmd: [build_tfrecord.py](../preprocess/build_tfrecord.py) ~/FRec/DisenID/data/TN/QB_behaviour.tsv
3. build incremental ids
4. convert tfrecords to align incremental ids