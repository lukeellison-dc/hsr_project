from asr_trainer import ASRTrainer, DataLoader
import torch

for gender in ['female', 'male', 'mixed']:
    success = False
    while not success:
        trainer = ASRTrainer(name=gender)
        trainer.load_model()
        dataloader = DataLoader(gender)
        dataloader.load()

        optimizer_ft = torch.optim.AdamW(trainer.model.parameters(), lr=0.0001, weight_decay=0.01)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        success = trainer.train(dataloader, optimizer_ft, exp_lr_scheduler, batch_size=52, num_epochs=20)
