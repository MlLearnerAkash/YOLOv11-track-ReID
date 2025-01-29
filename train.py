from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
# Initialize a Weights & Biases run
wandb.init(project="opervu-needle-dark-removed-27012025", job_type="training")

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)
add_wandb_callback(model, enable_model_checkpointing=False)

# Train the model with 2 GPUsd
results = model.train(project= "/mnt/data/weights/opervu_needle_dark_removed_27012025",
                      name="opervu_needle_dark_removed_27012025_",
                      data="/mnt/data/dataset/YOLODataset/dataset.yaml", 
                      epochs=2000, imgsz=2480, device=0, batch=6,
                      plots =True, resume = False, save_period=-1,
                      save= True,
                      pretrained = False,
                      cache = False,
                    
                      )

# Finalize the W&B Run
wandb.finish()


