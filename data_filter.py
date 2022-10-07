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

df = pd.read_csv(f'{ROOT}validated.tsv', sep='\t', header=0, quoting=3)
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
df['gender'] = df['gender'].str.strip()
df = df.reset_index()
print(df.head())
print(f"male rows = {len(df.loc[df['gender'] == 'male'])}")
print(f"female rows = {len(df.loc[df['gender'] == 'female'])}")
test_df = df.loc[df['gender'] != 'male']
test_df = test_df.loc[test_df['gender'] != 'female']
print(f'non-male, non-female rows = {len(test_df)}')
print(df.head())


def write_output(temp_df, name):
    print(f'Writing "{name}" data...')
    file = open(f'data/{name}/train.tsv', 'w', newline='')
    writer = csv.writer(file, delimiter='\t', lineterminator='\n')
    train_thresh = len(temp_df)*8.0/10
    print(f"train thresh = {train_thresh}")
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
    for i,row in tqdm(temp_df.iterrows(), total=len(temp_df)):
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
female_df = df.loc[df['gender'] == 'female']
female_df = female_df.reset_index()
write_output(female_df, 'female')
male_df = df.loc[df['gender'] == 'male']
male_df = male_df.reset_index()
write_output(male_df, 'male')
# train = 2.864, test = 3.431
# {'train': {'male': 503781, 'female': 175893}, 'test': {'male': 129688, 'female': 37803}}