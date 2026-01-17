import streamlit as st
import pandas as pd
import Levenshtein
import datasets
import difflib

def show_diff(source, target):
    """Display source and target with highlighted differences in monospace."""
    sm = difflib.SequenceMatcher(None, source, target)
    
    source_html = []
    target_html = []
    
    for op, s1, e1, s2, e2 in sm.get_opcodes():
        if op == 'equal':
            source_html.append(source[s1:e1])
            target_html.append(target[s2:e2])
        elif op == 'delete':
            source_html.append(f'<span style="background-color: #ffcccc; color: #cc0000;">{source[s1:e1]}</span>')
        elif op == 'insert':
            target_html.append(f'<span style="background-color: #ccffcc; color: #008800;">{target[s2:e2]}</span>')
        elif op == 'replace':
            source_html.append(f'<span style="background-color: #ffcccc; color: #cc0000;">{source[s1:e1]}</span>')
            target_html.append(f'<span style="background-color: #ccffcc; color: #008800;">{target[s2:e2]}</span>')
    
    source_str = ''.join(source_html)
    target_str = ''.join(target_html)
    
    sample_style = "white-space: pre-wrap; font-family: monospace; font-size: 0.85em; background-color: #f8f8f8; padding: 8px; border-radius: 4px; border-left: 3px solid #ddd;"
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div style="{sample_style}">{source_str}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="{sample_style}">{target_str}</div>', unsafe_allow_html=True)

st.title("Dataset Preparation")
st.write("Step-by-step preparation of the wiki-edits dataset for spell correction training.")

# Step 0: Source dataset
st.header("Step 0: Source Dataset")
st.markdown("[zhk/wiki-edits](https://huggingface.co/datasets/zhk/wiki-edits)")
st.caption('The pre-training dataset of paper "G-SPEED: General SParse Efficient Editing MoDel". Visit https://github.com/Banner-Z/G-SPEED.git for more details.')
st.write("""
Dataset contains 506,255 pairs of sentences representing edits from Wikipedia, classified by edit intent:
- Readability (35.6%)
- Fluency (27.8%)
- Neutralization (18.5%)
- Simplification (18.0%)
""")

# Step 1: Load dataset
st.header("Step 1: Load Dataset")
st.code("ds = datasets.load_dataset('zhk/wiki-edits', split='train')", language="python")

@st.cache_data
def load_raw_dataset():
    ds = datasets.load_dataset('zhk/wiki-edits', split='train')
    return ds.to_pandas()

with st.spinner("Loading dataset..."):
    df_raw = load_raw_dataset()

st.write(f"**{len(df_raw):,}** total edits loaded")
st.dataframe(df_raw.head(10))

# Step 2: Filter by intent
st.header("Step 2: Filter by Intent")

st.write("Dataset categorizes edits by 4 intents:")

intent_counts = df_raw['intent'].value_counts()
# Stacked horizontal bar
intent_df = pd.DataFrame({'count': intent_counts}).T
st.bar_chart(intent_df, horizontal=True, stack=True)

# Order intents with Fluency last
intents = [i for i in df_raw['intent'].unique().tolist() if i != 'Fluency'] + ['Fluency']
intent_pcts = {i: f"{intent_counts[i] / len(df_raw) * 100:.1f}%" for i in intents}
tab_labels = [f"{i} ({intent_pcts[i]})" if i != 'Fluency' else f"**Fluency** ({intent_pcts[i]})" for i in intents]

with st.expander("🔍 Explore examples by intent"):
    tabs = st.tabs(tab_labels)
    for tab, intent in zip(tabs, intents):
        with tab:
            samples = df_raw[df_raw['intent'] == intent].head(5)
            for _, row in samples.iterrows():
                show_diff(row['source'], row['target'])
                st.divider()

st.write("For spelling correction, we extract only the **Fluency** category. This can be done using 🤗 dataset's `filter` method:")
st.code("ds = ds.filter(lambda x: x['intent'] == 'Fluency')", language="python")

df_fluency = df_raw[df_raw['intent'] == 'Fluency'].drop(columns=['intent'])
st.write(f"After filtering: **{len(df_fluency):,}** examples")
st.dataframe(df_fluency.head(10), hide_index=True)

# Step 3: Edit distance filtering
st.header("Step 3: Edit Distance Filtering")

@st.cache_data
def add_edit_distance(df):
    df = df.copy()
    df['edit_distance'] = df.apply(lambda row: Levenshtein.distance(row['source'], row['target']), axis=1)
    return df

with st.spinner("Computing edit distances..."):
    df_with_dist = add_edit_distance(df_fluency)

st.write("In the Fluency category we see edits where full words/phrases are replaced with better fitting ones:")
high_ed_samples = df_with_dist[df_with_dist['edit_distance'] > 8].head(3)
for _, row in high_ed_samples.iterrows():
    show_diff(row['source'], row['target'])
st.divider()

st.write("...and simple typo corrections:")
low_ed_samples = df_with_dist[df_with_dist['edit_distance'] == 1].head(3)
for _, row in low_ed_samples.iterrows():
    show_diff(row['source'], row['target'])

st.write("""
To remove the first category and focus on samples relevant for spellchecker, we calculate the 
[Levenshtein Edit Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) from source to target for every example
using the [python-Levenshtein](https://rapidfuzz.github.io/Levenshtein/levenshtein.html) library:
""")
st.code("df['edit_distance'] = df.apply(lambda row: Levenshtein.distance(row['source'], row['target']), axis=1)", language="python")

with st.expander("🔍 Explore examples by edit distance"):
    max_ed = int(df_with_dist['edit_distance'].max())
    ed_value = st.slider("Edit distance", min_value=1, max_value=min(max_ed, 20), value=1)
    ed_samples = df_with_dist[df_with_dist['edit_distance'] == ed_value].head(5)
    st.write(f"**{len(df_with_dist[df_with_dist['edit_distance'] == ed_value]):,}** examples with ED={ed_value}")
    for _, row in ed_samples.iterrows():
        show_diff(row['source'], row['target'])
        st.divider()

st.write("Edit distance distribution:")
dist_counts = df_with_dist['edit_distance'].value_counts().sort_index().head(20)
st.bar_chart(dist_counts)

st.write("For our training we filter for **edit distance = 1**:")
st.code("df = df[df.edit_distance == 1]", language="python")

df_final = df_with_dist[df_with_dist['edit_distance'] == 1]
st.write(f"After filtering: **{len(df_final):,}** examples ({len(df_final)/len(df_fluency)*100:.1f}% of Fluency edits)")

# Step 4: Analyze edit types
st.header("Step 4: Edit Type Analysis")

df_final = df_final.copy()
df_final['edit_type'] = (df_final['target'].str.len() - df_final['source'].str.len()).map({1: 'insertion', -1: 'deletion', 0: 'replacement'})

st.write("With ED=1, each edit falls into exactly one of three categories based on length change:")

st.write("**Insertion** (target is 1 character longer) — a missing character is added:")
for _, row in df_final[df_final['edit_type'] == 'insertion'].head(2).iterrows():
    show_diff(row['source'], row['target'])

st.write("**Deletion** (target is 1 character shorter) — an extra character is removed:")
for _, row in df_final[df_final['edit_type'] == 'deletion'].head(2).iterrows():
    show_diff(row['source'], row['target'])

st.write("**Replacement** (same length) — one character is substituted for another:")
for _, row in df_final[df_final['edit_type'] == 'replacement'].head(2).iterrows():
    show_diff(row['source'], row['target'])

st.caption("💡 Note: Transpositions (e.g. 'teh' → 'the') have ED=2 in standard Levenshtein distance, so they are excluded by our ED=1 filter.")

st.write("We can classify edits programmatically by comparing string lengths:")
st.code("df['edit_type'] = (df.target.str.len() - df.source.str.len()).map({1: 'insertion', -1: 'deletion', 0: 'replacement'})", language="python")

edit_type_counts = df_final['edit_type'].value_counts()
edit_type_df = pd.DataFrame({'count': edit_type_counts}).T
st.bar_chart(edit_type_df, horizontal=True, stack=True)

with st.expander("🔍 Explore examples by edit type"):
    edit_type = st.selectbox("Edit type:", df_final['edit_type'].unique().tolist())
    examples = df_final[df_final['edit_type'] == edit_type].head(10)
    for _, row in examples.iterrows():
        show_diff(row['source'], row['target'])
        st.divider()

# Exploration section
st.header("Step 5: Dataset Exploration")

st.write("""
To better understand our dataset, we calculate additional sample properties:
- **Source length (characters)** — total character count
- **Source length (words)** — word count (split by spaces)
- **Unique characters** — count of distinct characters in source
- **Longest word** — length of the longest word in source or target
""")

df_final['source_len_chars'] = df_final['source'].str.len()
df_final['source_len_words'] = df_final['source'].str.split().str.len()
df_final['unique_chars'] = df_final['source'].apply(lambda x: len(set(x)))
df_final['longest_word'] = df_final.apply(lambda row: max(len(w) for w in (row['source'] + ' ' + row['target']).split()), axis=1)

properties = {
    'Source length (characters)': 'source_len_chars',
    'Source length (words)': 'source_len_words',
    'Unique characters': 'unique_chars',
    'Longest word': 'longest_word',
}

property_code = {
    'Source length (characters)': "df['source_len_chars'] = df['source'].str.len()",
    'Source length (words)': "df['source_len_words'] = df['source'].str.split().str.len()",
    'Unique characters': "df['unique_chars'] = df['source'].apply(lambda x: len(set(x)))",
    'Longest word': "df['longest_word'] = df.apply(lambda row: max(len(w) for w in (row['source'] + ' ' + row['target']).split()), axis=1)",
}

with st.expander("🔍 Explore distributions and extreme values", expanded=True):
    prop_name = st.selectbox("Property:", list(properties.keys()))
    prop_col = properties[prop_name]
    
    st.code(property_code[prop_name], language="python")
    
    st.write(f"**Distribution of {prop_name}:**")
    hist_counts = df_final[prop_col].value_counts().sort_index()
    st.bar_chart(hist_counts)
    
    extreme = st.selectbox("Show examples with:", ["Highest values", "Lowest values"])
    if extreme == "Highest values":
        examples = df_final.nlargest(5, prop_col)
    else:
        examples = df_final.nsmallest(5, prop_col)
    
    for _, row in examples.iterrows():
        st.caption(f"{prop_name}: {row[prop_col]}")
        show_diff(row['source'], row['target'])
        st.divider()

# Step 6: Word-level analysis
st.header("Step 6: Word-Level Analysis")

st.write("First, let's classify edits by word count difference (target words − source words):")
st.code("df['word_diff'] = df['target'].str.split().str.len() - df['source'].str.split().str.len()", language="python")

df_final['word_diff'] = df_final['target'].str.split().str.len() - df_final['source'].str.split().str.len()
word_diff_counts = df_final['word_diff'].value_counts().sort_index()

st.write("With ED=1, we only see word differences of **-1**, **0**, or **+1**:")

import plotly.graph_objects as go
word_diff_counts = word_diff_counts.sort_index()  # -1, 0, +1 order
word_diff_pcts = (word_diff_counts / word_diff_counts.sum() * 100).round(1)
total = word_diff_counts.sum()
fig = go.Figure()
annotations = []
cumsum = 0
for idx in word_diff_counts.index:
    count = word_diff_counts[idx]
    pct = word_diff_pcts[idx]
    label = f"+{idx}" if idx > 0 else str(idx)
    fig.add_trace(go.Bar(
        x=[count], y=[""], orientation='h', name=f"{label}",
        showlegend=False,
    ))
    # Add annotation above the center of this segment
    annotations.append(dict(
        x=cumsum + count / 2,
        y=0,
        text=f"{label}: {count:,} ({pct}%)",
        showarrow=False,
        yshift=25,
        font=dict(size=11),
    ))
    cumsum += count
for ann in annotations:
    ann['xref'] = 'x'
    ann['yref'] = 'paper'
    ann['y'] = 1.3
fig.update_layout(
    barmode='stack', height=80,
    margin=dict(l=80, r=80, t=40, b=0),
    xaxis=dict(showticklabels=False, showgrid=False, range=[0, total]),
    yaxis=dict(showticklabels=False),
    annotations=annotations,
)
st.plotly_chart(fig, use_container_width=True)
st.caption("💡 Category \"-1\" is hardly visible on this chart because it's only 14 examples in the whole dataset.")

col1, col2 = st.columns(2)
with col1:
    st.write("**Word diff = +1** (space added or symbol replaced with space):")
    space_added = df_final[df_final['word_diff'] == 1].head(3)
    for _, row in space_added.iterrows():
        show_diff(row['source'], row['target'])

with col2:
    st.write("**Word diff = -1** (space removed or replaced with another symbol):")
    space_removed = df_final[df_final['word_diff'] == -1].head(3)
    for _, row in space_removed.iterrows():
        show_diff(row['source'], row['target'])

# Filter to same word count for typo analysis
df_same_words = df_final[df_final['word_diff'] == 0].copy()
st.write(f"For word-level mistake analysis, we filter to **word diff = 0**: **{len(df_same_words):,}** examples")
st.code("df_typos = df[df.word_diff == 0]", language="python")

st.divider()
st.subheader("Word-Level Analysis")

st.info("""
We are working with a dataset of corrections, meaning **source** is the mistake and **target** is the correct version.
However, when thinking about mistakes it's useful to think of them in reverse: *what correct word turns into what mistake?*
For that we reverse from source→target to target→source.
""")

view_mode = st.selectbox("View mode:", [
    "Mistake view [correct → mistake] (target → source)",
    "Correction view [mistake → correct] (source → target)"
])
is_mistake_view = "Mistake view" in view_mode

st.write("""
To find the edited word, we compare source and target word-by-word:
1. Split both sentences into words
2. Find the first differing word (since word count is the same, there's exactly one)
3. Extract the source and target versions of that word
""")

if is_mistake_view:
    st.caption("💡 Note: This method treats whitespace as word boundary, so a word with punctuation is treated as a separate token. This allows us to identify common punctuation mistakes like `However,` → `However`")
else:
    st.caption("💡 Note: This method treats whitespace as word boundary, so a word with punctuation is treated as a separate token. This allows us to identify common punctuation corrections like `However` → `However,`")

st.code("""
def get_edited_word(source, target):
    source_words, target_words = source.split(), target.split()
    i = next(i for i in range(len(source_words)) if source_words[i] != target_words[i])
    return source_words[i], target_words[i]
""", language="python")

def get_edited_word(source, target):
    source_words, target_words = source.split(), target.split()
    i = next(i for i in range(len(source_words)) if source_words[i] != target_words[i])
    return source_words[i], target_words[i]

df_same_words['edited_word'] = df_same_words.apply(lambda row: get_edited_word(row['source'], row['target']), axis=1)
df_same_words['source_word'] = df_same_words['edited_word'].apply(lambda x: x[0])
df_same_words['target_word'] = df_same_words['edited_word'].apply(lambda x: x[1])

df_typos = df_same_words.copy()

if is_mistake_view:
    df_typos['display_pair'] = df_typos['target_word'] + ' → ' + df_typos['source_word']
    pair_label = "Correct → Mistake"
    title = "**Most common mistakes:**"
else:
    df_typos['display_pair'] = df_typos['source_word'] + ' → ' + df_typos['target_word']
    pair_label = "Mistake → Correct"
    title = "**Most common corrections:**"

typo_counts = df_typos['display_pair'].value_counts().head(20)

st.write(title)
st.dataframe(typo_counts.reset_index().rename(columns={'display_pair': pair_label, 'count': 'Count'}), hide_index=True)

with st.expander("🔍 Explore examples"):
    selected_typo = st.selectbox("Select:", typo_counts.index.tolist())
    typo_examples = df_typos[df_typos['display_pair'] == selected_typo].head(5)
    for _, row in typo_examples.iterrows():
        if is_mistake_view:
            show_diff(row['target'], row['source'])
        else:
            show_diff(row['source'], row['target'])
        st.divider()

# Final summary
st.header("Summary")
st.write(f"""
| Step | Rows |
|------|------|
| Raw dataset | {len(df_raw):,} |
| After Fluency filter | {len(df_fluency):,} |
| After edit_distance=1 filter | {len(df_final):,} |
""")
