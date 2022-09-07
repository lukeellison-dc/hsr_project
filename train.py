from asr_trainer import ASRTrainer, CVDataset, CTCLoss
import torch
import torchaudio

import time
import copy

#### util funcs
def td_string(td):
    return f'{td // 60:.0f}m {td % 60:.0f}s'

def progress(iterator, total, every=20):
    i = 0
    loop_start = time.time()
    for x in iterator:
        yield x
        i+=1
        if i % every == 0:
            perc = i*1.0/total
            time_elapsed = time.time() - loop_start
            t_per_it = time_elapsed*1.0 / i
            its_remaining = total - i
            t_remaining = its_remaining * t_per_it
            print(f'{i}/{total} = {perc:.2f}% -- ETA = {td_string(t_remaining)}')


asrt = ASRTrainer()
asrt.load_model()
cvd = CVDataset(asrt.processor)
cvd.preload_datasets()
ctc_loss = CTCLoss()

optimizer_ft = torch.optim.AdamW(asrt.model.parameters(), lr=0.0001, weight_decay=0.01)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

def train_model(trainer, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(trainer.model.state_dict())
    best_wer = 100.0
    torch.autograd.set_detect_anomaly(True)


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                trainer.model.train()  # Set model to training mode
            else:
                trainer.model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_wer = 0.0

            # Iterate over data.
            for d in progress(cvd.single_dataloader(phase), total=cvd.dataset_sizes[phase]):
                # zero the parameter gradients
                optimizer.zero_grad()

                logits = trainer.get_logits(d["input_values"], grad=(phase == 'train'))
                pred = trainer.predict_argmax_from_logits(logits)
                print(f'sentence = {d["sentence"]}')
                print(f'pred = {pred}')

                wer = torchaudio.functional.edit_distance(d["sentence"], pred)
                print(f'wer = {wer}')

                target = ctc_loss.sentence_to_target(d["sentence"])
                loss = ctc_loss.singleLoss(input_logits=logits, target=target)
                print(f'loss = {loss}')

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * d["input_values"].size(0)
                running_wer += wer

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / cvd.dataset_sizes[phase]
            epoch_wer = running_wer / cvd.dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} WER: {epoch_wer:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_wer < best_wer:
                best_wer = epoch_wer
                best_model_wts = copy.deepcopy(trainer.model.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_wer:4f}')

    # load best model weights
    trainer.model.load_state_dict(best_model_wts)
    return trainer.model

model_ft = train_model(asrt, optimizer_ft, exp_lr_scheduler, num_epochs=25)