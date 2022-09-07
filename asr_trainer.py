import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import json
import re

class ASRTrainer():
    def __init__(self, device=None):
        torch.random.manual_seed(0)

        self.device = device
        if self.device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name="facebook/wav2vec2-base-960h"):
        print(f"Loading model '{model_name}' to device '{self.device}'")

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)

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
        return self.processor.batch_decode(pred_ids)[0]


class CTCLoss():
    def __init__(self, vocab="vocab.json"):
        self.re_chars_to_remove = re.compile(r"[^A-Z ']")

        with open("vocab.json", "r") as fp:
            self.vocab = json.load(fp)

        self.test_vocab()
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
            
    def sentence_to_target(self, sentence, pad_len=300):
        sentence = sentence.upper()
        sentence = self.re_chars_to_remove.sub('', sentence).replace(' ', '|')
        t = torch.zeros([1, pad_len], dtype=torch.int)
        for i,x in enumerate(sentence):
            t[0][i] = self.vocab[x]
        return {
            "target": t,
            "target_len": len(sentence),
            "pad_len": pad_len,
        }

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
    def __init__(self, processor):
        self._raw_datasets = {
            "train": None,
            "test": None,
        }
        self.dataset_sizes = {
            "train": None,
            "test": None,
        }
        self.processor = processor
        self.target_sample_rate = processor.feature_extractor.sampling_rate

    def _load_dataset(self, phase, root="_cv_corpus/en"):
        self._raw_datasets[phase] = torchaudio.datasets.COMMONVOICE(
            root=root,
            tsv=f"{phase}.tsv",
        )
        self.dataset_sizes[phase] = len(self._raw_datasets[phase])

    def preload_datasets(self):
        for phase in self._raw_datasets:
            self._load_dataset(phase)
    
    def single_dataloader(self, phase):
        if(self._raw_datasets[phase] == None):
            self._load_dataset(phase)

        for datum in self._raw_datasets[phase]:
            yield self.process_raw_data_item(datum)
        
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
            "wav": resampled_wav,
            "sentence": metadata["sentence"]
        }
