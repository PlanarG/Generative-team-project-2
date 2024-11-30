import os
import time
import datasets
import torch
import inspect
from torchvision import transforms
from diffusers import DDPMScheduler, DDPMPipeline
from PIL import Image

class Trainer:
    def __init__(self, model, config: dir, eval_only: bool = False):
        self.rundir = os.getcwd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)
        self.logger = None
        if "load_checkpoint" in config:
            model.load_state_dict(torch.load(os.path.join(self.rundir, config["checkpoint_dir"], config["load_checkpoint"]), weights_only=True))
        self.model = model.to(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_training_steps"])

        if not eval_only:
            self.load_dataset()
            self.load_logger()
            self.load_optim()
            self.load_scheduler()

    def load_dataset(self):
        dataset = datasets.load_dataset(os.path.join(self.rundir, self.config["dataset_dir"]), split="train")
        # dataset = datasets.load_dataset("huggan/smithsonian_butterflies_subset", split="train")
        preprocess = transforms.Compose([
            transforms.Resize((self.config["image_size"], self.config["image_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        def transform(examples):
            images = [preprocess(image.convert("L")) for image in examples["image"]]
            return {"image": images}   

        dataset.set_transform(transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
    
    def load_logger(self):
        if "logger" in self.config:
            if self.config["logger"] == "wandb":
                import wandb
                wandb.init(project="generative-diffusion", name=time.strftime("%Y-%m-%d_%H-%M-%S"))
                self.logger = wandb
            else:
                raise ValueError("Invalid logger")
        
    def load_optim(self):
        optimizer_name = self.config.get("optimizer", "AdamW")
        optimizer = getattr(torch.optim, optimizer_name)
        optimizer_params = self.config.get("optimizer_params", {})
        weight_decay = optimizer_params.get("weight_decay", 0)

        if weight_decay > 0:
            self.model_params_no_decay = {}
            if hasattr(self.model, "no_weight_decay"):
                self.model_params_no_decay = self.model.no_weight_decay()

            params_decay, params_no_decay, name_no_decay = [], [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(name.endswith(no_decay) for no_decay in self.model_params_no_decay):
                    params_no_decay.append(param)
                    name_no_decay.append(name)
                else:
                    params_decay.append(param)
            
            self.optimizer = optimizer(
                [
                    {"params": params_decay, "weight_decay": weight_decay},
                    {"params": params_no_decay, "weight_decay": 0.0}
                ],
                lr=self.config["lr_initial"],
                **optimizer_params
            )
        else:
            self.optimizer = optimizer(
                self.model.parameters(), 
                lr=self.config["lr_initial"],
                **optimizer_params
            )

    def load_scheduler(self):
        scheduler = getattr(torch.optim.lr_scheduler, self.config.get("scheduler", "StepLR"))
        
        sig = inspect.signature(scheduler)
        filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
        
        args = { arg: self.config[arg] for arg in self.config if (arg in filter_keys and arg != "optimizer") }
        self.scheduler = scheduler(self.optimizer, **args)
    
    def step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def save(self):
        file_name = os.path.join(self.rundir, self.config["checkpoint_dir"], f"model_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pt")
        torch.save(self.model.state_dict(), file_name)

    def make_image_grid(self, images, rows: int, cols: int, resize: int = None, mode: str = "RGB"):
        assert len(images) == rows * cols

        if resize is not None:
            images = [img.resize((resize, resize)) for img in images]

        w, h = images[0].size
        grid = Image.new(mode, size=(cols * w, rows * h))

        for i, img in enumerate(images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid
    
    def eval(self, mode="RGB"):
        pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)

        batch_size = self.config["eval_batch_size"]
        generator = torch.Generator(device=self.device).manual_seed(self.config["eval_seed"])

        images = pipeline(
            batch_size=batch_size, 
            generator=generator, 
            num_inference_steps=self.config["num_inference_steps"]
        ).images 

        image_grid = self.make_image_grid(images, rows=batch_size // 4, cols=4, mode=mode)
        output_dir = os.path.join(self.rundir, self.config["output_dir"])
        image_grid.save(f"{output_dir}/output_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png")