import torch
import argparse
import math
from collections import deque
from src.arona import ARONA
from src.config import ModelConfig

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    

    print(f"Loading model from {args.model_path}...")
    model = ARONA(ModelConfig()).to(device)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with untrained model weights.")
    
    model.eval()
    

    print("\n" + "="*50)
    print("ARONA Interactive Chat")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    print("="*50 + "\n")
    

    history_size = args.context_length
    conversation = deque(maxlen=history_size)
    
    try:
        while True:

            user_input = input("\nUser: ").strip()
            

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
                
            if not user_input:
                print("Please enter a message.")
                continue
            

            conversation.append(f"User: {user_input}")
            context_str = "\n".join(conversation)
            

            print("\nARONA: ", end="", flush=True)
            
            try:
                with torch.no_grad():
                    response = model.generate_sentence(context_str)
                    

                    if "User:" in response:
                        response = response.split("User:")[0].strip()
                    
                    print(response)
                    

                    conversation.append(f"ARONA: {response}")
            
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                print("Sorry, I couldn't generate a proper response.")

    except KeyboardInterrupt:
        print("\n\nConversation ended by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARONA Interactive Chat")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if GPU is available")
    parser.add_argument("--context_length", type=int, default=10,
                        help="Number of conversation turns to keep in context")
    
    args = parser.parse_args()
    main(args)