import torch
from collections import deque
from src.arona import ARONA
from src.config import ModelConfig

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = ARONA(ModelConfig).to(device)
    checkpoint = torch.load('checkpoints/epoch_checkpoint_e2_s8684_1.8905.pth',
                          map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("模型加载完成，开始对话")

    
    dialogue_history = deque(maxlen=ModelConfig.block_size * 3)
    
    try:
        while True:
            
            user_input = input("\nQ: ").strip()
            
           
            if not user_input:
                print("A: 请输入有效内容")
                continue

            
            dialogue_history.append(f"Q: {user_input}")
            context_str = "\n".join(dialogue_history)
            
            
            with torch.no_grad():
                raw_response = model.generate_sentence(context_str)
                
  
            cleaned_response = raw_response.split("Q:")[0].strip()
            
            
            print(f"\nA: {cleaned_response}")
            dialogue_history.append(f"A: {cleaned_response}")
            
    except KeyboardInterrupt:
        print("\n对话已结束")