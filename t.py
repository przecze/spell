import os
import pandas as pd
import comet_ml
comet_ml.login(project_name='token-classification')


if not 'torch' in locals():
    import torch
    import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
import time
import psutil


tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def log_memory(message):
    memory = psutil.Process().memory_info().rss / 1024 ** 2
    print(f"{message}: {memory} MB")


def get_first_different_idx(source, target):
    l = min(len(source), len(target))
    diff = source[:l] != target[:l]
    return diff.nonzero()[0].item()


def prepare_batch(df_batch, tokenizer):
    source_tokens = tokenizer(df_batch['source'].tolist(), return_tensors='pt', padding=True)
    target_tokens = tokenizer(df_batch['target'].tolist(), return_tensors='pt', padding=True)
    shape = source_tokens.input_ids.shape
    diff_mask = torch.zeros(shape)
    for i in range(len(df_batch)):
        source = source_tokens.input_ids[i]
        target = target_tokens.input_ids[i]
        source = source[source != tokenizer.pad_token_id]
        target = target[target != tokenizer.pad_token_id]
        if len(target) > len(source):
            # insert case - we mark the token before the first different token
            first_diff_idx = get_first_different_idx(source, target)
            diff_mask[i, first_diff_idx] = 1
        else:
            # we mark source tokens that are replaced or removed
            first_diff_idx = get_first_different_idx(source, target)
            suffix_len = get_first_different_idx(torch.flip(source, dims=[0]), torch.flip(target, dims=[0]))
            last_diff_idx = len(source) - suffix_len
            diff_mask[i, first_diff_idx:last_diff_idx] = 1
    return source_tokens, diff_mask


def load_df(sample_size=None):
    df = pd.read_parquet('df.parquet')
    df.drop_duplicates(subset=['source'], inplace=True)
    if sample_size:
        df = df.sample(sample_size)
    return df


def load_model():
    model = AutoModelForTokenClassification.from_pretrained('roberta-base', num_labels=1)
    return model


def print_tokens(row):
    ids_source = row['source_tokens']
    ids_target = row['target_tokens']
    while ids_source[0] == ids_target[0]:
        ids_source = ids_source[1:]
        ids_target = ids_target[1:]
    while ids_source[-1] == ids_target[-1]:
        ids_source = ids_source[:-1]
        ids_target = ids_target[:-1]
    tokens_source = '|'.join(tokenizer.convert_ids_to_tokens(ids_source))
    tokens_target = '|'.join(tokenizer.convert_ids_to_tokens(ids_target))

    print(f"""
    Source: {row['source']}
    Target: {row['target']}
    Source Tokens: {tokens_source}
    Target Tokens: {tokens_target}
    """)

tqdm.pandas()

if 'df' not in locals():
    df = load_df()
    df['source_len'] = df.source.str.len()
    print("df loaded")
    print(df.info())
    print(df.head())


if 'model' not in locals():
    model = load_model()


def prepare_testset(test_df, tokenizer, batch_size):
    test_batch_size = batch_size * 2 # no backward pass so we can use bigger batch
    test_df_chunks = [test_df.iloc[i:i+test_batch_size] for i in range(0, len(test_df), test_batch_size)]
    test_batches = [prepare_batch(df_chunk, tokenizer) for df_chunk in test_df_chunks]
    def get_source_tokens_from_batch(batch):
        return [batch.input_ids[i][batch.attention_mask[i] == 1] for i in range(len(batch.input_ids))]
    source_tokens = [t.tolist() for batch in test_batches for t in get_source_tokens_from_batch(batch[0])]

    def clean_labels(batch, labels):
        return [labels[i][batch.attention_mask[i] == 1] for i in range(len(labels))]
    labels = [l.tolist() for (batch, labels) in test_batches for l in clean_labels(batch, labels)]
    test_df['source_tokens'] = source_tokens
    test_df['labels'] = labels
    return test_df, test_batches


def print_example_with_token_loss(row):
    print("Source: ", row['source'])
    print("Target: ", row['target'])
    print("Loss: ", row['loss'])
    tokens_source = tokenizer.convert_ids_to_tokens(row['source_tokens'])
    max_token_len = max((len(t) for t in tokens_source))
    print(" " * max_token_len, "| Pred | T | Loss")
    for i in range(len(tokens_source)):
        # pad with spaces to max_token_len
        print(f"{tokens_source[i]:{max_token_len}} | {row['predicted_probs'][i]:.2f} | {int(row['labels'][i])} | {row['token_loss'][i]:.2f}")


def handle_test_outputs(test_df, test_batches_outputs, tokenizer):
    logits = [l.flatten().tolist() for batch in test_batches_outputs for l in batch.logits]
    logits_trimmed = [l[:len(t)] for l, t in zip(logits, test_df.source_tokens)]
    probabilities = [F.sigmoid(torch.tensor(l)).tolist() for l in logits_trimmed]
    test_df['predicted_probs'] = probabilities
    def get_token_loss(row):
        return F.binary_cross_entropy(torch.tensor(row['predicted_probs']), torch.tensor(row['labels']), reduction='none')
    test_df['token_loss'] = test_df.apply(get_token_loss, axis=1)
    test_df['loss'] = test_df.token_loss.apply(lambda x: x.mean().item())
    print("Best loss:")
    print_example_with_token_loss(test_df.sort_values('loss').iloc[0])
    print("Worst loss:")
    print_example_with_token_loss(test_df.sort_values('loss', ascending=False).iloc[0])
    print()

  

def evaluate(model, test_batches, test_df, exp):
    time_start = time.time()
    outputs = []
    for batch, _ in test_batches:
        with torch.no_grad():
            outputs.append(model(**batch))
    handle_test_outputs(test_df, outputs, tokenizer)
    # conver all tensors to numpy arrays
    for col in test_df.columns:
        if isinstance(test_df[col].iloc[0], torch.Tensor):
            test_df[col] = test_df[col].apply(lambda x: x.numpy())
    test_df.to_parquet('test_df.parquet')
    eval_loss = test_df.loss.mean()
    df = test_df.explode(['labels', 'predicted_probs'])
    avg_prob = df.predicted_probs.mean()
    avg_prob_one = df[df.labels.eq(1)].predicted_probs.mean()
    avg_prob_zero = df[df.labels.eq(0)].predicted_probs.mean()
    time_eval = time.time()
    exp.log_metrics({
        'eval_loss': eval_loss,
        'avg_prob': avg_prob,
        'avg_prob_one': avg_prob_one,
        'avg_prob_zero': avg_prob_zero,
        'time_eval_ms': (time_eval - time_start) * 1000,
    })
    print(f"Eval Loss: {eval_loss}")
    print(f"Avg Prob: {avg_prob:.2f}")
    print(f"Avg Prob One: {avg_prob_one:.2f}")
    print(f"Avg Prob Zero: {avg_prob_zero:.2f}")


batch_size = 16
test_size = 20*batch_size
test_df = df.sample(test_size, random_state=42)
test_df = test_df.sort_values('source_len')
df = df[~df.index.isin(test_df.index)]
df = df.sample(frac=1)  # shuffle
test_df, test_batches = prepare_testset(test_df, tokenizer, batch_size)

# split df into chunks
df_chunks = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]

weight_decay = 0.01
initial_lr = 1e-5
lr_scheduler_t_max = 1000
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_scheduler_t_max)
exp = comet_ml.start()
comet_name = exp.get_name()
eval_every_n_steps = 30
save_every_n_steps = 30
exp.log_parameters({
    'batch_size': batch_size,
    'eval_every_n_steps': eval_every_n_steps,
    'weight_decay': weight_decay,
    'initial_lr': initial_lr,
    'lr_scheduler_t_max': lr_scheduler_t_max,
    'save_every_n_steps': save_every_n_steps,
    'eval_size': test_df.shape[0],
})
eval_steps_counter = eval_every_n_steps - 1 # start with eval
eval_steps_counter = 0
save_steps_counter = save_every_n_steps - 1
for df_chunk in df_chunks:
    time_start = time.time()
    batch, labels = prepare_batch(df_chunk, tokenizer)
    time_batch = time.time()
    
    outputs = model(**batch)
    time_forward = time.time()
    optimizer.zero_grad()
    loss = F.binary_cross_entropy_with_logits(outputs.logits.view(-1), labels.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    time_step = time.time()
    save_steps_counter += 1
    if save_steps_counter == save_every_n_steps:
        save_steps_counter = 0
        start_time = time.time()
        model.save_pretrained(f'checkpoints/{comet_name}')
        tokenizer.save_pretrained(f'checkpoints/{comet_name}')
        save_time = time.time()
        file_size = os.path.getsize(f'checkpoints/{comet_name}')
        exp.log_metrics({
            'time_save_ms': (save_time - start_time) * 1000,
            'file_size': file_size,
        })
    eval_steps_counter += 1
    if eval_steps_counter == eval_every_n_steps:
        evaluate(model, test_batches, test_df, exp)
        eval_steps_counter = 0
    time_total_ms = (time.time() - time_start) * 1000
    time_batch_ms = (time_batch - time_start) * 1000
    time_forward_ms = (time_forward - time_batch) * 1000
    time_step_ms = (time_step - time_forward) * 1000
    print(f"Batch: {batch_size} x {batch.input_ids.shape[1]} | "
          f"Total: {time_total_ms:.2f} ms | "
          f"Prep: {time_batch_ms:.2f} ms | "
          f"Forward: {time_forward_ms:.2f} ms | "
          f"Step: {time_step_ms:.2f} ms | "
    )
    exp.log_metrics({
        'time_total_ms': time_total_ms,
        'time_batch_ms': time_batch_ms,
        'time_forward_ms': time_forward_ms,
        'time_step_ms': time_step_ms,
        'lr': scheduler.get_last_lr()[0],
        'loss': loss.item(),
    })
    print(f"Loss: {loss.item()}")