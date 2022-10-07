### args
import sys
gender = 'mixed'
if len(sys.argv) > 2:
    print('Error: expected 1 arg at most!')
    sys.exit(1)
elif len(sys.argv) < 2:
    print('No args supplied: Training on mixed dataset by default')
else:
    if sys.argv[1] == 'female':
        gender = 'female'
    elif sys.argv[1] == 'male':
        gender = 'male'
    elif sys.argv[1] == 'mixed':
        gender = 'mixed'
    else:
        print(f'Error: Do not recognise method "{sys.argv[1]}"')
        sys.exit(1)
#######

from asr_trainer import ASRTrainer, DataLoader
import torch

trainer = ASRTrainer(name=gender)
trainer.load_model()
dataloader = DataLoader(gender)
dataloader.load()

optimizer_ft = torch.optim.AdamW(trainer.model.parameters(), lr=0.0003, weight_decay=0.01)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

trainer.train(dataloader, optimizer_ft, exp_lr_scheduler, batch_size=64, num_epochs=20)
