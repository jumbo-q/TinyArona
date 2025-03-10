# TinyArona
A Tiny CasualLM With Developed Only with Pytorch


# Model Structure
- tokenizer-encode
- embedding layer
- decoder layer (defualt is 4)
  - MultiHeadAttention
  - FFN
- Softmax
- tokenizer-decode

# Setup
### dataset

#### SOURCE

[Genshin Impact (from huggingface)](https://huggingface.co/datasets/simon3000/genshin-voice)

#### DOWNLOAD

[Genshin Data ](https://drive.google.com/file/d/1gCua0vShgr1_xG2WOImsu6-PrSzLRFvr/view?usp=sharing)

> [!IMPORTANT]
> Need to use [data_generator.py](./tools/data_generator.py) to generate format data (*.npy)


### Training

run [trian.py](./trainer.py)
