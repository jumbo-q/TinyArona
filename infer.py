import torch
from src.arona import ARONA
from src.config import ModelConfig

if __name__ == '__main__':
    device = torch.device('cuda')
    model = ARONA(ModelConfig).to(device)
    checkpoint = torch.load('checkpoints/model_epoch_1.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    context = []
    while True:
        user_input = input("Q:")
        context.append(user_input)
        context_str = "\t".join(context[-ModelConfig.block_size:])  
        
        response = model.generate_sentence(context_str)
        print("ARONA:", response)
        
        context.append(response)
        if len(context) > ModelConfig.block_size:
            context = context[-ModelConfig.block_size:]