import pandas as pd
import unicodedata
import re
import csv
from tqdm import tqdm

ROOT = '_cv_corpus/en/'
re_chars_to_remove = re.compile(r"[^A-Z\s]+")
re_whitespace = re.compile(r"\s+")

def normalise(sentence):
    sentence = unicodedata.normalize('NFKD', sentence)
    sentence = sentence.upper().replace('-', ' ')
    sentence = re_chars_to_remove.sub('', sentence)
    sentence = re_whitespace.sub(' ', sentence) #collapse spaces
    return sentence

df = pd.read_csv(f'{ROOT}validated.tsv', sep='\t', header=0)
print(f'total rows = {len(df)}')
counts = df.count()
print(f"rows with gender = {counts['gender']}")

df = df.dropna(subset=['gender'])
print(f"rows with gender = {len(df)}")
df = df[['path', 'sentence', 'up_votes', 'down_votes', 'gender']]
df = df.loc[df['down_votes'] == 0]
print(f"rows with no down votes = {len(df)}")
df = df.loc[df['up_votes'] >= 2]
print(f"rows with >1 up votes = {len(df)}")
df = df.reset_index()
print(df.head())

def write_output(df, name):
    print(f'Writing "{name}" data...')
    file = open(f'data/{name}/train.tsv', 'w', newline='')
    writer = csv.writer(file, delimiter='\t', lineterminator='\n')
    train_thresh = len(df)*8.0/10
    file_changed = False
    counts = {
        'train': {
            'male': 0,
            'female': 0,
        },
        'test': {
            'male': 0,
            'female': 0,
        }
    }
    for i,row in tqdm(df.iterrows(), total=len(df)):
        if not file_changed and i > train_thresh:
            file.close()
            file = open(f'data/{name}/test.tsv', 'w', newline='')
            writer = csv.writer(file, delimiter='\t', lineterminator='\n')
            file_changed = True
        writer.writerow([row["path"], normalise(row["sentence"])])
        counts['test' if file_changed else 'train']['male'] += 1 if row["gender"] == 'male' else 0
        counts['test' if file_changed else 'train']['female'] += 1 if row["gender"] == 'female' else 0
    print(counts)

write_output(df, "mixed")
write_output(df.loc[df['gender'] == 'female'], 'female')
write_output(df.loc[df['gender'] == 'male'], 'male')
# {'train': {'male': 503667, 'female': 175780}, 'test': {'male': 129659, 'female': 37775}}
# train = 2.865, test = 3.4324023825