from asr_trainer import ASRTrainer, CVDataset
import torch

trainer = ASRTrainer()
trainer.load_model()
cvd = CVDataset(trainer.processor)
cvd.preload_datasets()

optimizer_ft = torch.optim.AdamW(trainer.model.parameters(), lr=0.0003, weight_decay=0.01)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = trainer.train(cvd, optimizer_ft, exp_lr_scheduler, batch_size=16, num_epochs=3)
