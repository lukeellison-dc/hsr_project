import pandas as pd
from transformers import Wav2Vec2Processor
import torchaudio
import json
import torch
from tqdm import tqdm
import pickle
import lzma

class Processor():
    def __init__(self) -> None:
        self.w2v2processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.target_sample_rate = self.w2v2processor.feature_extractor.sampling_rate

        self.resamplers = {}

    def resample(self, in_wav, in_sample_rate):
        resampler = self.resamplers.get(in_sample_rate, None)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(in_sample_rate, self.target_sample_rate, dtype=in_wav.dtype)
            self.resamplers[in_sample_rate] = resampler

        return resampler(in_wav)[0]

    def extract_features(self, wav):
        features = self.w2v2processor(
            wav, 
            sampling_rate=self.target_sample_rate, 
            return_tensors="pt",
        )
        return features.input_values[0]

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

    def sentence_to_target(self, sentence, pad_len=250):
        sentence = sentence.replace(' ', '|') #replace all whitespace with |
        t = torch.zeros([1, pad_len], dtype=torch.int)
        longer = False
        for i,x in enumerate(sentence):
            if i < pad_len:
                t[0][i] = self.vocab[x]
            else:
                longer = True
        if longer:
            print(f'Setence longer than pad len with len={len(sentence)}:\n"{sentence}"')
        return t

proc = Processor()
tc = TargetCreator()
for gender in ['female', 'male', 'mixed']:
    for phase in ['train', 'test']:
        print(f'Processing {gender},{phase}...')
        df = pd.read_csv(f'./data/{gender}/{phase}.tsv', sep='\t', names=['path', 'sentence'])
        pad_len = 250
        data = {
            'input_values': [],
            'sentences': [],
            'targets': torch.zeros([len(df), pad_len], dtype=torch.int),
        }
        
        for i,row in tqdm(df.iterrows(), total=len(df), mininterval=5):
            wav, sr = proc.load(f'_cv_corpus/en/clips/{row["path"]}')
            wav = proc.resample(wav, sr)
            if wav.size(0) < 211585: #GPU can't handle sentences longer than this with the memory
                input_values = proc.extract_features(wav)
                data['input_values'].append(input_values)

                data['sentences'].append(row['sentence'])
                target = tc.sentence_to_target(row['sentence'], pad_len)[0]
                data['targets'][i] = tc.sentence_to_target(row['sentence'], pad_len)[0]

        with lzma.open(f'/raid/lellison_data/hsr_project/processed/{gender}/{phase}.xz','wb') as outfile:
            pickle.dump(data, outfile)
