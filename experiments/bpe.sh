cd ../..;
python -u fairseq_cli/bpe_prior.py \
/home/jskai/workspace/fairseq/data-bin/iwslt14.tokenized.de-en \
--seed 1 \
--source-lang de \
--target-lang en \
--arch transformer \
--dataset-impl mmap \
--max-epoch 4 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--distributed-world-size 0 \
| tee -a /home/jskai/workspace/fairseq/experiments/iwslt14de-en/bpe.log
