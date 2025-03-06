import torch

from src.arona import ARONA
from src.config import ModelConfig
if __name__ == '__main__':
    device = torch.device('cuda')
    model = ARONA(ModelConfig).to(device)


    checkpoint = torch.load('checkpoints/model_epoch_1.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    #初始输入是空，每次加上后面的对话信息
    sentence = ''
    while True:
        temp_sentence = input("我:")
        sentence += (temp_sentence + '\t')
        if len(sentence) > 512:
            #由于该模型输入最大长度为512，避免长度超出限制长度过长需要进行裁剪
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]
        
        print("ARONA:", model.generate_sentence(sentence))


