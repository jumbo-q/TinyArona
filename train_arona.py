import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from src.arona import AronaDataset, ARONA
from src.config import ModelConfig
import time
def main():

    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    dataset = AronaDataset(config)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4
    )


    model = ARONA(config).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.t_max,
        eta_min=config.min_learning_rate
    )

   
    global_step = 0
    samples_processed = 0
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(config.num_epochs):

        localtime = time.asctime( time.localtime(time.time()) )

        print('🕐 Current time: ', localtime)
        model.train()
        epoch_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config.num_epochs}",
            bar_format="{l_bar}{bar:20}{r_bar}"
        )
        
        for batch_idx, (x, y) in enumerate(epoch_progress):
            x, y = x.to(device), y.to(device)
            
            _, loss = model(x, target=y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            samples_processed += x.size(0)

            epoch_progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

            if samples_processed >= 10000:
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, global_step, samples_processed,
                    loss.item(), "step_checkpoint"
                )
                samples_processed %= 10000  

        val_loss = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch+1} validation loss: {val_loss:.4f}")

        save_checkpoint(
            model, optimizer, scheduler,
            epoch, global_step, samples_processed,
            val_loss, "epoch_checkpoint"
        )

def evaluate(model, data_loader, device):

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Validating", leave=False):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, target=y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def save_checkpoint(model, optimizer, scheduler, epoch, step, samples, loss, ctype):

    checkpoint = {
        "epoch": epoch,
        "global_step": step,
        "samples_processed": samples,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "loss": loss,
    }
    
    filename = f"{ctype}_e{epoch+1}_s{step}_{loss:.4f}.pth"
    torch.save(checkpoint, os.path.join("checkpoints", filename))
    print(f"Saved {ctype} at step {step}")

if __name__ == "__main__":
    main()