from math import ceil
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import time
import jiwer
import statistics
import lzma
import pickle


#### util funcs
def td_string(td):
    return f'{td // 60:.0f}m {td % 60:.0f}s'

def progress(iterator, total, prefix='', every=20):
    def log(i):
        perc = i*100.0/total
        time_elapsed = time.time() - loop_start
        t_per_it = time_elapsed*1.0 / i
        its_remaining = total - i
        t_remaining = its_remaining * t_per_it
        print(f'{prefix}{i}/{total} = {perc:.2f}% -- ETA = {td_string(t_remaining)}')

    i = 0
    loop_start = time.time()
    for x in iterator:
        yield x
        i+=1
        if i % every == 0:
            log(i)
    log(i)

class ASRTrainer():
    def __init__(self, name, device=None):
        torch.random.manual_seed(0)

        self.device = device
        self.name = name
        if self.device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name="facebook/wav2vec2-base-960h"):
        print(f"Loading model '{model_name}' to device '{self.device}'")

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)

    def get_logits(self, input_values, grad=False):
        with torch.set_grad_enabled(grad):
            return self.model(input_values.to(self.device)).logits

    def pred_ids_from_logits(self, logits):
        return torch.argmax(logits, dim=-1)

    def predict_argmax(self, input_values):
        logits = self.get_logits(input_values)
        return self.predict_argmax_from_logits(logits)

    def predict_argmax_from_logits(self, logits):
        pred_ids = self.pred_ids_from_logits(logits)
        return self.processor.batch_decode(pred_ids)

    def train(self, dataloader, optimizer, scheduler, num_epochs=25, batch_size=64):
        ctc_loss = CTCLoss()
        since = time.time()

        best_wer = 100.0
        best_loss = 10000.0
        losses = []
        wers = []

        for epoch in progress(range(num_epochs), total=num_epochs, prefix='epochs: ', every=1):
            print('-' * 10)
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_wers = []

                # Iterate over data.
                loader = dataloader.batch_generator(phase, batch_size=batch_size)
                print(dataloader.length)
                total = ceil(dataloader.length[phase] * 1.0/batch_size)
                every = 20
                batch_i = 0
                for d in progress(loader, total=total, prefix=f'{phase}_batch: ', every=every):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    logits = self.get_logits(d["input_values"], grad=(phase == 'train'))
                    pred = self.predict_argmax_from_logits(logits)

                    if batch_i%every == 0:
                        for i in range(2):
                            print(f'- sentence[{i}] = {d["sentences"][i]}')
                            print(f'- pred[{i}] = {pred[i]}')
                        w = jiwer.wer(d["sentences"], pred)
                        print(f'- wer = {w}')

                    loss = ctc_loss(logits, d["targets"])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * d["input_values"].size(0)
                    running_wers.append(jiwer.wer(d["sentences"], pred))

                    batch_i+=1
                    # if batch_i > 60:
                    #     break
                    # print(torch.cuda.max_memory_reserved())

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataloader.length[phase]
                epoch_wer = statistics.fmean(running_wers)

                print('-' * 10)
                print(f'{phase} Loss: {epoch_loss:.4f} WER: {epoch_wer:.4f}')

                if phase == 'test':
                    losses.append(epoch_loss)
                    wers.append(epoch_wer)
                    if epoch_wer < best_wer:
                        print(f'New best found, saving...')
                        best_wer = epoch_wer
                        self.save(self.model.state_dict())
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Loss: {best_loss:4f}')
        print(f'Best WER: {best_wer:4f}')
        print(f'All losses: {losses}')
        print(f'All WERs: {wers}')

    def save(self, wts):
        path = f'./models/{self.name}_model.pt'
        if wts:
            torch.save(wts, path)


class CTCLoss(nn.Module):
    """Convenient wrapper for CTCLoss that handles log_softmax and taking input/target lengths."""

    def __init__(self, blank: int = 0) -> None:
        """Init method.

        Args:
            blank (int, optional): Blank token. Defaults to 0.
        """
        super().__init__()
        self.blank = blank

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            preds (torch.Tensor): Model predictions. Tensor of shape (batch, sequence_length, num_classes), or (N, T, C).
            targets (torch.Tensor): Target tensor of shape (batch, max_seq_length). max_seq_length may vary
                per batch.

        Returns:
            torch.Tensor: Loss scalar.
        """
        # print(f'preds.shape = {preds.shape}')
        # print(f'targets.shape = {targets.shape}')
        preds = preds.log_softmax(-1)
        batch, seq_len, classes = preds.shape
        preds = rearrange(preds, "n t c -> t n c") # since ctc_loss needs (T, N, C) inputs
        # equiv. to preds = preds.permute(1, 0, 2), if you don't use einops

        pred_lengths = torch.full(size=(batch,), fill_value=seq_len, dtype=torch.long)
        target_lengths = torch.count_nonzero(targets, axis=1)
        m = torch.max(target_lengths)
        targets = targets[:,:m]

        return F.ctc_loss(preds, targets, pred_lengths, target_lengths, blank=self.blank, zero_infinity=True)

class DataLoader():
    def __init__(self, gender):
        self.gender = gender
        self.data = {}
        self.length = {}

    def load(self, root="/raid/lellison_data/hsr_project/processed/"):
        for phase in ['train', 'test']:
            print(f'Loading dataset for {self.gender},{phase}...')
            start = time.time()
            with lzma.open(f'{root}{self.gender}/{phase}.xz','rb') as infile:
                self.data[phase] = pickle.load(infile)
            self.length[phase] = len(self.data[phase]['sentences'])
            print(f'Loaded {phase} data in {td_string(time.time() - start)}.')

        print(f'Loaded datasets.')
    
    def batch_generator(self, phase, batch_size=64):
        keys = ['input_values', 'sentences', 'targets']
        for i in range(0, self.length[phase], batch_size):
            upper = min(i+batch_size, self.length[phase])
            batch = {
                x: self.data[phase][x][i:upper] for x in keys
            }
            batch['input_values'] = torch.nn.utils.rnn.pad_sequence(batch['input_values'], batch_first=True)
            yield batch