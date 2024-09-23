function run() {
SEED=${1:-608}
N_MASK=${2:-1}
MASK_RATE=${3:-0.15}
LM_SCALER=${4:-0.5}
BPE_DIM=${5:-256}
LR=${6:-5e-4}
N_LM=${7:-5}
N_PARSER=${8:-3}
LOG_DIR=/path/to/save/logs/
DATA_BIN=/path/to/data/ # e.g. iwslt14.tokenized.de-en

mkdir -p $LOG_DIR
cp run-grammarformer.sh $LOG_DIR/run.sh
cd ../..

CUDA_VISIBLE_DEVICES=4,5 python -u fairseq_cli/train.py $DATA_BIN \
    --bpe-path /path/to/bpe-prior/ \
    --arch grammarformer_iwslt_de_en --share-decoder-input-output-embed \
    --n-parser-layers $N_PARSER \
    --bpe-dim $BPE_DIM \
    --n-mask-layers $N_MASK \
    --n-lm-layers $N_LM \
    --mask-rate $MASK_RATE  \
    --task translation --source-lang de --target-lang en \
    --seed $SEED \
    --lm-scaler $LM_SCALER \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr $LR --lr-scheduler inverse_sqrt --stop-min-lr '1e-09' \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 1 \
    --validate-after-updates 10000 \
    --max-epoch 60 \
    --keep-last-epochs 5 \
    --no-progress-bar \
    --ddp-backend=no_c10d \
    --save-dir $LOG_DIR \
    --tensorboard-logdir $LOG_DIR/tensorboard_log \
| tee -a $LOG_DIR/train.log


CUDA_VISIBLE_DEVICES=5 python -u scripts/average_checkpoints.py \
    --inputs $LOG_DIR \
    --num-epoch-checkpoints 5 \
    --output $LOG_DIR/checkpoint_avg5.pt \
| tee -a $LOG_DIR/average.log 


CUDA_VISIBLE_DEVICES=5 python -u fairseq_cli/generate.py $DATA_BIN \
    --bpe-path /home/jskai/workspace/fairseq/bpe-prior/iwslt14de2en \
    --task translation --source-lang de --target-lang en \
    --path $LOG_DIR/checkpoint_avg5.pt \
    --batch-size 128 \
    --beam 5 --remove-bpe \
| tee -a $LOG_DIR/test.log

cd experiments/iwslt14de-en
}

run 1
run 2
run 3
run 4
run 5
