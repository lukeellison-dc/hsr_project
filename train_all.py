from asr_trainer import ASRTrainer, CVDataset
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()

for method in ['full', 'female', 'male']:
    trainer = ASRTrainer(name=method)
    trainer.load_model()
    cvd = CVDataset(trainer.processor)
    if method == 'female':
        cvd = CVDataset(trainer.processor, filters={
            "gender": "female",
        })
    if method == 'male':
        cvd = CVDataset(trainer.processor, filters={
            "gender": "male",
        })
    cvd.preload_datasets()

    optimizer_ft = torch.optim.AdamW(trainer.model.parameters(), lr=0.0003, weight_decay=0.01)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = trainer.train(cvd, optimizer_ft, exp_lr_scheduler, batch_size=64, num_epochs=25)
