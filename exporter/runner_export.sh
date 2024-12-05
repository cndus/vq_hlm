MODEL=neulab/gpt2-finetuned-wikitext103
# available splits: train/validation/test
python -u export_hs.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset $1 \
  --local_rank -1