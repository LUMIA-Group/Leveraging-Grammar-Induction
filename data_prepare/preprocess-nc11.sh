TEXT=data-prepare/nc11de-en
cd ..;
python fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/nc11de-en \
    --workers 20