# Preprocess/binarize the data
TEXT=data-prepare/aspecch2ja
cd ..;
python fairseq_cli/preprocess.py --source-lang ch --target-lang ja \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/aspec_ch_ja \
    --workers 20