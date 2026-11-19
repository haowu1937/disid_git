# TN

python /home/hao/FRec/DisenID/preprocess/build_tfrecord.py \
  --emb_dir=/mnt/data_2/data_hao/DisenID/data/TN/Qwen3VL_emb \
  --behaviour_path=/mnt/data_2/data_hao/DisenID/data/TN/TN_behaviour.tsv \
  --out_dir=/mnt/data_2/data_hao/DisenID/data/TN/tfrecords

python /home/hao/FRec/DisenID/preprocess/remap_seq_data.py \
  --data_dir=/mnt/data_2/data_hao/DisenID/data/TN/tfrecords \
  --id_mapping_path=/mnt/data_2/data_hao/DisenID/data/TN/Qwen3VL_emb/shard_00000_ids_incremental.npy

cp -r /mnt/data_2/data_hao/DisenID/data/TN/tfrecords/remapped_incremental_tfrecord/* \
  /home/shared/ninerec/tn/

# QB

python /home/hao/FRec/DisenID/preprocess/build_tfrecord.py \
  --emb_dir=/mnt/data_2/data_hao/DisenID/data/QB/Qwen3VL_emb \
  --behaviour_path=/mnt/data_2/data_hao/DisenID/data/QB/QB_behaviour.tsv \
  --out_dir=/mnt/data_2/data_hao/DisenID/data/QB/tfrecords

python /home/hao/FRec/DisenID/preprocess/remap_seq_data.py \
  --data_dir=/mnt/data_2/data_hao/DisenID/data/QB/tfrecords \
  --id_mapping_path=/mnt/data_2/data_hao/DisenID/data/QB/Qwen3VL_emb/shard_00000_ids_incremental.npy

cp -r /mnt/data_2/data_hao/DisenID/data/QB/tfrecords/remapped_incremental_tfrecord/* \
  /home/shared/ninerec/qb/


# Bili_500K

python /home/hao/FRec/DisenID/preprocess/build_tfrecord.py \
  --emb_dir=/mnt/data_2/data_hao/DisenID/data/Bili_500K/Qwen3VL_emb \
  --behaviour_path=/mnt/data_2/data_hao/DisenID/data/Bili_500K/Bili_500K_behaviour.tsv \
  --out_dir=/mnt/data_2/data_hao/DisenID/data/Bili_500K/tfrecords


python /home/hao/FRec/DisenID/preprocess/remap_seq_data.py \
  --data_dir=/mnt/data_2/data_hao/DisenID/data/Bili_500K/tfrecords \
  --id_mapping_path=/mnt/data_2/data_hao/DisenID/data/Bili_500K/Qwen3VL_emb/shard_00000_ids_incremental.npy

cp -r /mnt/data_2/data_hao/DisenID/data/Bili_500K/tfrecords/remapped_incremental_tfrecord/* \
  /home/shared/ninerec/Bili_500K/


  python main.py     --model=TIGER     --dataset=DisenSIDs     --data_path=/home/shared/ninerec     --dataset_name=tn     --sid_variant=rqvae     --num_hierarchies=3     --rq_codebook_size=256     --rq_n_codebooks=3     --share_n_codebooks=0     --n_user_tokens=1