from trainer import Trainer
from model import get_model
from config import get_config

import torch
import logging
import torch.nn.functional as F
from PIL import Image

config = get_config("config/config.yml")
model = get_model(config)
trainer = Trainer(model, config)

logging.getLogger().setLevel(logging.INFO)

print(f"num of parameters: {sum(p.numel() / 1048576 for p in model.parameters() if p.requires_grad)}M")

global_step = 0

total_length = len(trainer.dataloader)

for epoch in range(config["num_epochs"]):
    for step, batch in enumerate(trainer.dataloader):
        clean_images = batch["image"].to(trainer.device)
        bs = clean_images.shape[0]
        noise = torch.randn(clean_images.shape, device=trainer.device)

        timesteps = torch.randint(0, trainer.noise_scheduler.config.num_train_timesteps, (bs, ), device=trainer.device, dtype=torch.int64)
        noisy_images = trainer.noise_scheduler.add_noise(clean_images, noise, timesteps) 
        
        noise_pred = trainer.model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()

        trainer.step()

        logs = {"loss": loss.detach().item(), "lr": trainer.scheduler.get_last_lr()[0], "step": global_step, "epoch": epoch + step / total_length}
        logging.info(logs)
        if trainer.logger is not None:
            trainer.logger.log(logs)
        global_step += 1

        if step % 300 == 0:
            trainer.eval(mode="L")
    
    if (epoch + 1) % config["save_every_epoch"] == 0:
        trainer.save()
    


