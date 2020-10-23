# S2S-AMR-Parser

Part of Code for our paper "Improving AMR parsing with Sequence-to-Sequence Pre-training" in EMNLP-2020.

Now we make our part of code and temporary best pre-trained model available to predict AMR graph for arbitrary sentences.

## 0. Environment Setup
0. Pytorch 1.1.0
   
   Models are trained under Pytorch 1.1.0 and they may not be loaded by lower version directly (such as 0.4.0), even the code works well.

1. AllenNLP
   
   We use [AllenNLP](https://allennlp.org/) to tokenize the source sentences for AMR Parsing.

2. subword-nmt
   
   We employs BPE to segment all the tokens into subwords by byte pair encoding. See [https://github.com/rsennrich/subword-nmt/](https://github.com/rsennrich/subword-nmt/) for more details.

3. Post-Processing tool
   
   See [https://github.com/RikVN/AMR](https://github.com/RikVN/AMR) for more details.

4. pre-trained model
   
   We provide our temporary best model PTM-MT(WMT14B)-SemPar(WMT14M), which greately advances the state-of-the-art performance **with 81.4 Smatch on AMR2.0**.
   > Download Here </br>
   > 链接：https://pan.baidu.com/s/1bdIKXBtlSldC-IPMxkG04A </br>
   > 提取码：SUDA 


## 1. Predict AMR Graph

Assuming that the file named "sent" contains sentences waiting for parsing .

### Step0 : Tokenization
See [AllenNLP Documents](http://docs.allennlp.org/v0.9.0/api/allennlp.data.tokenizers.html#word-tokenizer) for more details.

Here is a python demo for Tokenization.
```python
from allennlp.data.tokenizers import WordTokenizer
token = WordTokenizer()
sent = "Has history given us too many lessons?, 530, 412, 64"
tokenized_sent = " ".join(str(tok) for tok in token.tokenize(sent.strip()))
print(tokenized_sent)
# OUTPUT : 
# 'Has history given us too many lessons ? , 530 , 412 , 64'
```

### Step1 : BPE
we employ BPE to segment word sequence into subword sequence.

Here is a demo for BPE.
```bash
# $subword_nmt is path of BPE tools, E.G.
# subword_nmt=XXX/subword-nmt/subword_nmt/
python3 $subword_nmt/apply_bpe.py -c bpe.codes < sent.tok > sent.tok.bpe
```
### Step2 : Decoding
Now we can use subword sequence and pre-trained model to generate AMR Graph.

Here is a command demo for Decoding.
```bash
# $GPU_ID is the N-th GPU used
# $amr_parser_model is pre-trained model path, here is PTM-MT(WMT14B)-SemPar(WMT14M), E.G.
# GPU_ID=0
# amr_parser_model=ptm_mt_en2deB_sem_enM.pt
CUDA_VISIBLE_DEVICES=$GPU_ID python3 codes/translate.py -model $amr_parser_model -beam_size 5 -src sent.tok.bpe -output sent.amr.bpe -task_type task2 -decode_extra_length 1000 -minimal_relative_prob 0.01 -gpu 0
```
### Step3 : Remove BPE
From Step2 we can get the BPE AMR sequence and we should remove BPE symbol, @@, with under command.
``` bash
sed -r 's/(@@ )|(@@ ?$)//g' sent.amr.bpe > sent.amr
```
### Step4: Post-Processing
Now we get sequence AMR from Step3. We should do post-processing if need to recover its full graph.

See [Pre- and post-processing scripts for neural sequence-to-sequence AMR parsing](https://github.com/RikVN/AMR) for more details.

Here is a command demo for post-processing.
```bash
python2 postprocess_AMRs.py -f sent.amr -s sent
```

# Acknowledgements
We adopted some modules or code from [AllenNLP](https://allennlp.org/), [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), [subword-nmt](https://github.com/rsennrich/subword-nmt/) and [RikVN/AMR](https://github.com/RikVN/AMR). Thanks to these open-source projects!


# Cite
If you like our paper or parser, please cite
```
@misc{xu2020improving,
      title={Improving AMR Parsing with Sequence-to-Sequence Pre-training}, 
      author={Dongqin Xu and Junhui Li and Muhua Zhu and Min Zhang and Guodong Zhou},
      year={2020},
      eprint={2010.01771},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```