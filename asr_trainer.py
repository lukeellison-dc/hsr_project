from importlib.metadata import metadata
import unicodedata
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import json
import re
import time
import copy
import jiwer
import statistics


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

    def train(self, cvdataset, optimizer, scheduler, num_epochs=25, batch_size=64):
        target_creator = TargetCreator()
        ctc_loss = CTCLoss()
        wer = WER()
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
                loader = cvdataset.batch_dataloader(phase, batch_size=batch_size)
                total = cvdataset.dataset_sizes[phase] * 1.0/batch_size
                every = 20
                batch_i = 0
                for d in progress(loader, total=total, prefix=f'{phase}_batch: ', every=every):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    logits = self.get_logits(d["input_values"], grad=(phase == 'train'))
                    pred = self.predict_argmax_from_logits(logits)
                    normalised_sents = [target_creator.normalise(x) for x in d["sentences"]]

                    if batch_i%every == 0:
                        for i in range(2):
                            print(f'- sentence[{i}] = {normalised_sents[i]}')
                            print(f'- pred[{i}] = {pred[i]}')
                        w = wer.wer(normalised_sents, pred)
                        print(f'- wer = {w}')


                    targets = torch.stack([target_creator.sentence_to_target(x)[0] for x in normalised_sents]) #.to(self.device)
                    loss = ctc_loss(logits, targets)
                    # print(f'loss = {loss.item()}')

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * d["input_values"].size(0)
                    running_wers.append(wer.wer(normalised_sents, pred))

                    batch_i+=1
                    # if batch_i > 60:
                    #     break
                    # print(torch.cuda.max_memory_reserved())

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / cvdataset.dataset_sizes[phase]
                epoch_wer = statistics.fmean(running_wers)

                print('-' * 10)
                print(f'{phase} Loss: {epoch_loss:.4f} WER: {epoch_wer:.4f}')

                if phase == 'test':
                    losses.append(epoch_loss)
                    wers.append(epoch_wer)
                    # deep copy the model
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
        # print(preds.shape)
        # print(targets.shape)
        preds = preds.log_softmax(-1)
        batch, seq_len, classes = preds.shape
        preds = rearrange(preds, "n t c -> t n c") # since ctc_loss needs (T, N, C) inputs
        # equiv. to preds = preds.permute(1, 0, 2), if you don't use einops

        pred_lengths = torch.full(size=(batch,), fill_value=seq_len, dtype=torch.long)
        target_lengths = torch.count_nonzero(targets, axis=1)

        return F.ctc_loss(preds, targets, pred_lengths, target_lengths, blank=self.blank, zero_infinity=True)


class TargetCreator():
    def __init__(self, vocab="vocab.json"):
        self.re_chars_to_remove = re.compile(r"[^A-Z\s]+")
        self.re_whitespace = re.compile(r"\s+")
        # self.expandCommonEnglishContractions = jiwer.ExpandCommonEnglishContractions()

        with open("vocab.json", "r") as fp:
            self.vocab = json.load(fp)

        # self.test_vocab()
        assert self.vocab["<pad>"] == 0
        self.torch_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        self.unk = self.vocab["<unk>"]

    def test_vocab(self):
        number_of_classes = len(self.vocab)
        classes = list(range(number_of_classes))
        for x in self.vocab.values():
            try:
                classes.remove(x)
            except Exception as e:
                print(f"Could not remove '{x}'")
                print(f"Remaining classes: '{classes}'")
                raise e

        assert len(classes) == 0

    def normalise(self, sentence):
        sentence = unicodedata.normalize('NFKD', sentence)
        sentence = sentence.upper().replace('-', ' ')
        sentence = self.re_chars_to_remove.sub('', sentence)
        return sentence
            
    def sentence_to_target(self, sentence, pad_len=400):
        sentence = self.normalise(sentence)
        sentence = self.re_whitespace.sub('|', sentence) #replace all whitespace with |
        t = torch.zeros([1, pad_len], dtype=torch.int)
        for i,x in enumerate(sentence):
            if i < pad_len:
                t[0][i] = self.vocab[x]
        return t
        # return {
        #     "target": t,
        #     "target_len": len(sentence),
        #     "pad_len": pad_len,
        # }

    def singleLoss(self, input_logits, target):
        assert input_logits.shape[0] == 1 
        assert target["target"].shape[0] == 1

        input_lengths = torch.tensor([input_logits.shape[1]]) # in array of length 1 because single loss (no batch)
        target_lengths = torch.tensor([target["target_len"]])
        inp = input_logits.squeeze().unsqueeze(1)
        # Try `inp.log_softmax(2)`
        loss = self.torch_loss(inp.log_softmax(2), target["target"], input_lengths, target_lengths)
        return loss

    def backward(self):
        self.torch_loss.backward()

class CVDataset():
    def __init__(self, processor, filters={}):
        self._raw_datasets = {
            "train": None,
            "test": None,
        }
        self.dataset_sizes = {
            "train": None,
            "test": None,
        }
        self.filters = filters
        self.processor = processor
        self.target_sample_rate = processor.feature_extractor.sampling_rate

        self.resamplers = {}

    def _load_dataset(self, phase, root="_cv_corpus/en"):
        self._raw_datasets[phase] = torchaudio.datasets.COMMONVOICE(
            root=root,
            tsv=f"{phase}.tsv",
        )
        self.dataset_sizes[phase] = len(self._raw_datasets[phase])
        if len(self.filters) > 0:
            self.dataset_sizes[phase] = sum(1 for dummy in self._filtered_ds(phase, log_prog=True))
        print(f'{phase} dataset size = {self.dataset_sizes[phase]}')
        # self.dataset_sizes[phase] = 20

    def preload_datasets(self):
        print('Preloading datasets...')
        for phase in self._raw_datasets:
            self._load_dataset(phase)
        print('Preloading Complete.')

    def _filtered_ds(self, phase, log_prog=False):
        if(self._raw_datasets[phase] == None):
            self._load_dataset(phase)
        
        i = 0
        iterator = progress(self._raw_datasets[phase], total=len(self._raw_datasets[phase]), every=10000) if log_prog else self._raw_datasets[phase]
        for datum in iterator:
            _, _, metadata = datum
            for key,val in self.filters.items():
                if metadata[key] == val:
                    yield datum
    
    def single_dataloader(self, phase):
        if(self._raw_datasets[phase] == None):
            self._load_dataset(phase)

        for datum in self._raw_datasets[phase]:
            yield self.process_raw_data_item(datum)

    def batch_dataloader(self, phase, batch_size=32):
        if(self._raw_datasets[phase] == None):
            self._load_dataset(phase)

        batch_wavs = []
        batch_sentences = []

        i = 0
        iterator = self._filtered_ds(phase) if len(self.filters) else self._raw_datasets[phase]
        for datum in iterator:
            wav, sr, metadata = self.resample(datum)
            if wav.size(0) < 211585: #GPU can't handle sentences longer than this with the memory
                batch_wavs.append(wav)
                batch_sentences.append(metadata["sentence"])
            i += 1
            if i >= batch_size:
                input_values = self.process_batch_wavs(batch_wavs)
                yield {
                    "input_values": self.process_batch_wavs(batch_wavs),
                    "sentences": batch_sentences
                }
                batch_wavs = []
                batch_sentences = []
                i = 0
            # if i == 20: break
        yield {
            "input_values": self.process_batch_wavs(batch_wavs),
            "sentences": batch_sentences
        }
    
    def resample(self, datum):
        in_wav, in_sample_rate, metadata = datum
        resampler = self.resamplers.get(in_sample_rate, None)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(in_sample_rate, self.target_sample_rate, dtype=in_wav.dtype)

        return resampler(in_wav)[0], self.target_sample_rate, metadata

    def process_batch_wavs(self, batch_wavs):
        vals = []
        for x in batch_wavs:
            features = self.processor(
                x, 
                sampling_rate=self.target_sample_rate, 
                return_tensors="pt",
            )
            vals.append(features.input_values.t())
        input_values = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True).squeeze()
        del vals
        return input_values

        
    def process_raw_data_item(self, datum):
        raw_wav, in_sample_rate, metadata = datum
        resampled_wav = torchaudio.functional.resample(raw_wav, in_sample_rate, self.target_sample_rate)
        features = self.processor(
            resampled_wav[0], 
            sampling_rate=self.target_sample_rate, 
            return_tensors="pt",
            padding=True
        )

        return {
            "input_values": features.input_values,
            "sentence": metadata["sentence"]
        }

class WER():
    def __init__(self) -> None:
        self.transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.SubstituteRegexes({
                r"[^\w\d\s]+": "",
            }),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ]) 
    
    def wer(self, truths, preds):
        return jiwer.wer(truths, preds, truth_transform=self.transformation, hypothesis_transform=self.transformation)