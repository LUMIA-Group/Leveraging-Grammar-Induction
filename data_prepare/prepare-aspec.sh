
MOSES_SCRIPT=mosesdecoder/scripts
SCRIPT_DIR=script.converter.distribution

# cd corpus.org/

# for name in dev test train; do
#  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[1], "\n";' < ${name}.txt > ${name}.ja.txt
#  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[2], "\n";' < ${name}.txt > ${name}.zh.txt
# done

# cd ..

mkdir corpus.tok
cd corpus.tok

for file in train dev test; do
  cat ../corpus.org/${file}.ja.txt | \
    perl -CSD -Mutf8 -pe 's/　/ /g;' | \
    juman -b | \
    perl -ne 'chomp; if($_ eq "EOS"){print join(" ",@b),"\n"; @b=();} else {@a=split/ /; push @b, $a[0];}' | \
    perl -pe 's/^ +//; s/ +$//; s/ +/ /g;' | \
    perl -CSD -Mutf8 -pe 'tr/\|[]/｜［］/; ' \
    > ${file}.ja
done

for name in dev test train; do
 /home/jskai/workspace/fairseq/data-bin/stanford-segmenter-2014-01-04/segment.sh ctb ../corpus.org/${name}.zh.txt UTF-8 0 | \
   perl -CSD -Mutf8 -pe 'tr/\|[]/｜［］/; ' \
   > ${name}.zh
done

cd ..

mkdir corpus.bpe
cd corpus.bpe

subword-nmt learn-joint-bpe-and-vocab --input ../corpus.tok/train.zh ../corpus.tok/train.ja -s 100000 -o bpe_codes --write-vocabulary vocab.zh vocab.ja

for name in train dev test; do
  subword-nmt apply-bpe -c bpe_codes --vocabulary vocab.zh --vocabulary-threshold 10 < ../corpus.tok/${name}.zh > ${name}.zh
  subword-nmt apply-bpe -c bpe_codes --vocabulary vocab.ja --vocabulary-threshold 10 < ../corpus.tok/${name}.ja > ${name}.ja
done

cd ..

