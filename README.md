Leveraging Grammar Induction for Language Understanding and Generation
===

Code for the paper [Leveraging Grammar Induction for Language Understanding and Generation](https://arxiv.org/abs/2410.04878).

This repo is build upon [fairseq toolkit](https://github.com/facebookresearch/fairseq).

## Environment setup

```
git clone https://github.com/LUMIA-Group/Leveraging-Grammar-Induction
cd Leveraging-Grammar-Induction
pip install --editable ./
```

## Example experiments
```
cd data_prepare # prepare and prepocess the data
bash prepare-iwslt14.sh
bash preprocess-iwslt14.sh
```

- **Training and evalutaion**
```
cd ../experiments
```

- Baseline: Transformer
```
bash run-transformer.sh
```

- Grammar-enhanced Transformer
```
bash bpe.sh
bash run-grammarformer.sh
```

## Citation
If you find this work helpful, please cite our paper:
```
@misc{kai2024leveraginggrammarinductionlanguage,
      title={Leveraging Grammar Induction for Language Understanding and Generation}, 
      author={Jushi Kai and Shengyuan Hou and Yusheng Huang and Zhouhan Lin},
      year={2024},
      eprint={2410.04878},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.04878}, 
}
```
