import pandas as pd
from transformers import Wav2Vec2Processor
import torchaudio
import json
import torch
from tqdm import tqdm
import pickle
import bz2

class Resampler():
    def __init__(self) -> None:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.target_sample_rate = processor.feature_extractor.sampling_rate

        self.resamplers = {}

    def resample(self, in_wav, in_sample_rate):
        resampler = self.resamplers.get(in_sample_rate, None)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(in_sample_rate, self.target_sample_rate, dtype=in_wav.dtype)
            self.resamplers[in_sample_rate] = resampler

        return resampler(in_wav)[0]

    def load(self, path):
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate

class TargetCreator():
    def __init__(self, vocab="vocab.json"):
        with open(vocab, "r") as fp:
            self.vocab = json.load(fp)

        # self.test_vocab()
        assert self.vocab["<pad>"] == 0
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

    def sentence_to_target(self, sentence, pad_len=400):
        sentence = sentence.replace(' ', '|') #replace all whitespace with |
        t = torch.zeros([1, pad_len], dtype=torch.int)
        for i,x in enumerate(sentence):
            if i < pad_len:
                t[0][i] = self.vocab[x]
            else:
                raise Exception("longer than pad len")
        return t

rs = Resampler()
tc = TargetCreator()
for gender in ['mixed', 'male', 'female']:
    for phase in ['train', 'test']:
        print(f'Processing {gender},{phase}...')
        df = pd.read_csv(f'./data/{gender}/{phase}.tsv', sep='\t', names=['path', 'sentence'])
        pad_len = 400
        data = {
            'wavs': [],
            'sentences': [],
            'targets': torch.zeros([len(df), pad_len], dtype=torch.int),
        }
        
        for i,row in tqdm(df.iterrows(), total=len(df)):
            wav, sr = rs.load(f'_cv_corpus/en/clips/{row["path"]}')
            wav = rs.resample(wav, sr)
            data['wavs'].append(wav)

            data['sentences'].append(row['sentence'])
            target = tc.sentence_to_target(row['sentence'], pad_len)[0]
            data['targets'][i] = tc.sentence_to_target(row['sentence'], pad_len)[0]

        with bz2.BZ2File(f'_cv_corpus/en/processed/{gender}/{phase}.pbz2','wb') as outfile:
            pickle.dump(data, outfile)
