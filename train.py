### args
import sys
method = 'full'
if len(sys.argv) > 2:
    print('Error: expected 1 arg at most!')
    sys.exit(1)
elif len(sys.argv) < 2:
    print('No args supplied: Training on full dataset by default')
else:
    if sys.argv[1] == 'female':
        method = 'female'
    elif sys.argv[1] == 'male':
        method = 'male'
    elif sys.argv[1] == 'full':
        method = 'full'
    else:
        print(f'Error: Do not recognise method "{sys.argv[1]}"')
        sys.exit(1)
#######

from asr_trainer import ASRTrainer, CVDataset
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer = ASRTrainer(method)
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

trainer.train(cvd, optimizer_ft, exp_lr_scheduler, batch_size=8, num_epochs=25)
