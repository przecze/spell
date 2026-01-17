from tqdm import tqdm
import plotly.express as px
import Levenshtein
import datasets

ds = datasets.load_dataset('zhk/wiki-edits', split='train')
# filter only intent = "Fluency"
ds = ds.filter(lambda x: x['intent'] == 'Fluency')
df = ds.to_pandas()




def get_edit_distance(base_sentence, edited_sentence):
    return Levenshtein.distance(base_sentence, edited_sentence)

def get_diff_string(source, target):
    return ''.join(
        '.' if source[i] == target[i] else target[i] for i in range(min(len(source), len(target)))
    )

def get_first_different(source, target):
    return next((i for i in range(min(len(source), len(target))) if source[i] != target[i]), None)

def get_edited_word(source, target):
    source_words = source.split(' ')
    target_words = target.split(' ')
    first_different = get_first_different(source_words, target_words)
    last_different = get_first_different(source_words[::-1], target_words[::-1])
    return ' '.join(target_words[first_different:-last_different]), ' '.join(source_words[first_different:-last_different])

tqdm.pandas()
df['edit_distance'] = df.apply(lambda row: get_edit_distance(row['source'], row['target']), axis=1)
df = df[df.edit_distance == 1]
df['diff_string'] = df.progress_apply(lambda row: get_diff_string(row['source'], row['target']), axis=1)
df['edit_type'] = (df.target.str.len() - df.source.str.len()).map({1: 'insertion', -1: 'deletion', 0: 'replacement'})
print(df.edit_type.value_counts(normalize=True))
df['edited_word'] = df.progress_apply(lambda row: get_edited_word(row['source'], row['target']), axis=1)
print(df.edited_word.value_counts(normalize=True))
