# TinyArona
A Tiny CasualLM With Developed Only with Pytorch


# Model Structure
- tokenizer
- MultiHeadAttention
- FFN
- Softmax

# Setup
## trainner

### dataset

[Genshin Impact (from huggingface)](https://huggingface.co/datasets/simon3000/genshin-voice)

[!IMPORTANT]
> Need to use [data_generator.py](./tools/data_generator.py) to generate format data


### Training

run [trian.py](./trainer.py)


# TODO
- [ ] GQA
- [ ] RoPE