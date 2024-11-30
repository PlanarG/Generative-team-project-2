from trainer import Trainer
from model import get_model
from config import get_config

import torch.nn.functional as F

config = get_config("config/config.yml")
model = get_model(config)
trainer = Trainer(model, config, eval_only=True)

trainer.eval()
