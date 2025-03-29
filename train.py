from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
# Initialize a Weights & Biases run
wandb.init(project="opervu_28SI_training_1", job_type="training")


# Load a model
model = YOLO("/mnt/data/weights/all_data_v11/yolov11_all_data_02122024_5/weights/best.pt") 
add_wandb_callback(model, enable_model_checkpointing=False)

# Train the model with 2 GPUsd
results = model.train(project= "/mnt/data/weights/opervu_28SIs_07032025",
                      name="opervu_28SIs_07032025_",
                      data="/mnt/data/dataset/YOLODataset/dataset.yaml", 
                      epochs=200, imgsz=2480, device=0, batch=6,
                      plots =True, resume = False, save_period=50,
                      save= True,
                      pretrained = False,
                      cache = False,
                    
                      )

# Finalize the W&B Run
wandb.finish()


