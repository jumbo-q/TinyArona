import os
import time
import torch
import numpy as np
import math
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.config import ModelConfig
from src.arona import ARONA, AronaDataset

def main():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    print("Loading dataset...")
    dataset = AronaDataset(config)
    print(f"Dataset size: {len(dataset)} examples")
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=(device.type == 'cuda')
    )
    
    print("Initializing model...")
    model = ARONA(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params / 1e6:.2f}M parameters")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = config.num_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        progress = float(current_step - config.warmup_steps) / float(max(1, total_steps - config.warmup_steps))
        return max(config.min_learning_rate / config.learning_rate, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("Starting training...")
    best_val_loss = float('inf')
    step = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, loss = model(x, targets=y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            
            step += 1
            epoch_loss += loss.item()
            lr = scheduler.get_last_lr()[0]
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{lr:.2e}"
            })
            
            if step % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(config.save_dir, f"step_{step}.pt")
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                _, loss = model(x, targets=y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(config.save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
            }, best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        latest_model_path = os.path.join(config.save_dir, "latest_model.pt")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val_loss,
        }, latest_model_path)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        

if __name__ == "__main__":
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()
