import json
import os
import difflib

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")


@st.cache_resource
def load_precomputed():
    with open(f"{DATA_DIR}/stats.json") as f:
        stats = json.load(f)
    return {
        "stats": stats,
        "raw_head": pd.read_parquet(f"{DATA_DIR}/raw_head.parquet"),
        "intent_samples": pd.read_parquet(f"{DATA_DIR}/intent_samples.parquet"),
        "fluency_head": pd.read_parquet(f"{DATA_DIR}/fluency_head.parquet"),
        "high_ed_samples": pd.read_parquet(f"{DATA_DIR}/high_ed_samples.parquet"),
        "low_ed_samples": pd.read_parquet(f"{DATA_DIR}/low_ed_samples.parquet"),
        "ed_samples": pd.read_parquet(f"{DATA_DIR}/ed_samples.parquet"),
        "final": pd.read_parquet(f"{DATA_DIR}/final.parquet"),
    }


@st.cache_resource
def compute_derived(_df_final):
    """Compute exploration and word-level columns from the pre-computed final dataset."""
    df = _df_final.copy()
    df["edit_type"] = (df["target"].str.len() - df["source"].str.len()).map(
        {1: "insertion", -1: "deletion", 0: "replacement"}
    )
    df["source_len_chars"] = df["source"].str.len()
    df["source_len_words"] = df["source"].str.split().str.len()
    df["unique_chars"] = df["source"].apply(lambda x: len(set(x)))
    df["longest_word"] = df.apply(
        lambda row: max(len(w) for w in (row["source"] + " " + row["target"]).split()), axis=1
    )
    df["word_diff"] = df["target"].str.split().str.len() - df["source"].str.split().str.len()

    df_same_words = df[df["word_diff"] == 0].copy()
    source_words_col = df_same_words["source"].str.split()
    target_words_col = df_same_words["target"].str.split()
    edited_idx = source_words_col.combine(target_words_col, lambda s, t: next(
        i for i in range(len(s)) if s[i] != t[i]
    ))
    df_same_words["source_word"] = source_words_col.combine(edited_idx, lambda words, i: words[i])
    df_same_words["target_word"] = target_words_col.combine(edited_idx, lambda words, i: words[i])

    return df, df_same_words


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

    sample_style = (
        "white-space: pre-wrap; font-family: monospace; font-size: 0.85em;"
        " background-color: #f8f8f8; padding: 8px; border-radius: 4px;"
        " border-left: 3px solid #ddd;"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div style="{sample_style}">{source_str}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="{sample_style}">{target_str}</div>', unsafe_allow_html=True)


data = load_precomputed()
stats = data["stats"]
df_final, df_same_words = compute_derived(data["final"])

# --- Page content ---

st.title("Dataset Preparation")
st.write("Step-by-step preparation of the wiki-edits dataset for spell correction training.")

# Step 0: Source dataset
st.header("Step 0: Source Dataset")
st.markdown("[zhk/wiki-edits](https://huggingface.co/datasets/zhk/wiki-edits)")
st.caption(
    'The pre-training dataset of paper "G-SPEED: General SParse Efficient Editing MoDel".'
    ' Visit https://github.com/Banner-Z/G-SPEED.git for more details.'
)
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

st.write(f"**{stats['total_rows']:,}** total edits loaded")
st.dataframe(data["raw_head"])

# Step 2: Filter by intent
st.header("Step 2: Filter by Intent")

st.write("Dataset categorizes edits by 4 intents:")

intent_counts = stats["intent_counts"]
intent_counts_series = pd.Series(intent_counts)
intent_df = pd.DataFrame({"count": intent_counts_series}).T
st.bar_chart(intent_df, horizontal=True, stack=True)

intents = stats["intents_ordered"]
intent_pcts = {i: f"{intent_counts[i] / stats['total_rows'] * 100:.1f}%" for i in intents}
tab_labels = [
    f"**Fluency** ({intent_pcts[i]})" if i == 'Fluency' else f"{i} ({intent_pcts[i]})"
    for i in intents
]

with st.expander("🔍 Explore examples by intent"):
    tabs = st.tabs(tab_labels)
    for tab, intent in zip(tabs, intents):
        with tab:
            samples = data["intent_samples"]
            samples = samples[samples["intent"] == intent].head(5)
            for _, row in samples.iterrows():
                show_diff(row['source'], row['target'])
                st.divider()

st.write(
    "For spelling correction, we extract only the **Fluency** category."
    " This can be done using 🤗 dataset's `filter` method:"
)
st.code("ds = ds.filter(lambda x: x['intent'] == 'Fluency')", language="python")

st.write(f"After filtering: **{stats['fluency_count']:,}** examples")
st.dataframe(data["fluency_head"], hide_index=True)

# Step 3: Edit distance filtering
st.header("Step 3: Edit Distance Filtering")

st.write(
    "In the Fluency category we see edits where full words/phrases"
    " are replaced with better fitting ones:"
)
for _, row in data["high_ed_samples"].iterrows():
    show_diff(row['source'], row['target'])
st.divider()

st.write("...and simple typo corrections:")
for _, row in data["low_ed_samples"].iterrows():
    show_diff(row['source'], row['target'])

st.write("""
To remove the first category and focus on samples relevant for spellchecker, we calculate the
[Levenshtein Edit Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) from source
to target for every example using the
[python-Levenshtein](https://rapidfuzz.github.io/Levenshtein/levenshtein.html) library:
""")
st.code(
    "df['edit_distance'] = df.apply("
    "lambda row: Levenshtein.distance(row['source'], row['target']), axis=1)",
    language="python",
)

with st.expander("🔍 Explore examples by edit distance"):
    max_ed = min(stats["max_ed"], 20)
    ed_value = st.slider("Edit distance", min_value=1, max_value=max_ed, value=1)
    ed_count = stats["ed_sample_counts"].get(str(ed_value), 0)
    ed_samples = data["ed_samples"]
    ed_samples_filtered = ed_samples[ed_samples["edit_distance"] == ed_value].head(5)
    st.write(f"**{ed_count:,}** examples with ED={ed_value}")
    for _, row in ed_samples_filtered.iterrows():
        show_diff(row['source'], row['target'])
        st.divider()

st.write("Edit distance distribution:")
ed_dist = stats["ed_distribution"]
dist_counts = pd.Series({int(k): v for k, v in ed_dist.items()}).sort_index()
st.bar_chart(dist_counts)

st.write("For our training we filter for **edit distance = 1**:")
st.code("df = df[df.edit_distance == 1]", language="python")

fluency_pct = stats["final_count"] / stats["fluency_count"] * 100
st.write(f"After filtering: **{stats['final_count']:,}** examples ({fluency_pct:.1f}% of Fluency edits)")

# Step 4: Analyze edit types
st.header("Step 4: Edit Type Analysis")

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

st.caption(
    "💡 Note: Transpositions (e.g. 'teh' → 'the') have ED=2 in standard Levenshtein distance,"
    " so they are excluded by our ED=1 filter."
)

st.write("We can classify edits programmatically by comparing string lengths:")
st.code(
    "df['edit_type'] = (df.target.str.len() - df.source.str.len())"
    ".map({1: 'insertion', -1: 'deletion', 0: 'replacement'})",
    language="python",
)

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
    'Longest word': (
        "df['longest_word'] = df.apply("
        "lambda row: max(len(w) for w in (row['source'] + ' ' + row['target']).split()), axis=1)"
    ),
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
st.code(
    "df['word_diff'] = df['target'].str.split().str.len()"
    " - df['source'].str.split().str.len()",
    language="python",
)

word_diff_counts = df_final['word_diff'].value_counts().sort_index()

st.write("With ED=1, we only see word differences of **-1**, **0**, or **+1**:")

word_diff_counts = word_diff_counts.sort_index()
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
st.plotly_chart(fig, width='stretch')
st.caption(
    "💡 Category \"-1\" is hardly visible on this chart"
    " because it's only 14 examples in the whole dataset."
)

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
st.write(
    "For word-level mistake analysis, we filter to"
    f" **word diff = 0**: **{len(df_same_words):,}** examples"
)
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

_whitespace_note = (
    "💡 Note: This method treats whitespace as word boundary,"
    " so a word with punctuation is treated as a separate token."
    " This allows us to identify common punctuation"
)
if is_mistake_view:
    st.caption(f"{_whitespace_note} mistakes like `However,` → `However`")
else:
    st.caption(f"{_whitespace_note} corrections like `However` → `However,`")

st.code("""
def get_edited_word(source, target):
    source_words, target_words = source.split(), target.split()
    i = next(i for i in range(len(source_words)) if source_words[i] != target_words[i])
    return source_words[i], target_words[i]
""", language="python")

if is_mistake_view:
    display_pair = df_same_words['target_word'] + ' → ' + df_same_words['source_word']
    pair_label = "Correct → Mistake"
    title = "**Most common mistakes:**"
else:
    display_pair = df_same_words['source_word'] + ' → ' + df_same_words['target_word']
    pair_label = "Mistake → Correct"
    title = "**Most common corrections:**"

typo_counts = display_pair.value_counts().head(20)

st.write(title)
st.dataframe(
    typo_counts.reset_index().rename(columns={0: pair_label, 'count': 'Count'}),
    hide_index=True,
)

with st.expander("🔍 Explore examples"):
    selected_typo = st.selectbox("Select:", typo_counts.index.tolist())
    typo_examples = df_same_words[display_pair == selected_typo].head(5)
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
| Raw dataset | {stats['total_rows']:,} |
| After Fluency filter | {stats['fluency_count']:,} |
| After edit_distance=1 filter | {stats['final_count']:,} |
""")

# Download buttons
st.subheader("Download Dataset")
df_export = df_final[['source', 'target', 'edit_type']].copy()

col1, col2, col3 = st.columns(3)
with col1:
    csv_data = df_export.to_csv(index=False)
    st.download_button(
        label="📄 CSV",
        data=csv_data,
        file_name="wiki_edits_ed1.csv",
        mime="text/csv",
    )
with col2:
    parquet_data = df_export.to_parquet(index=False)
    st.download_button(
        label="📦 Parquet",
        data=parquet_data,
        file_name="wiki_edits_ed1.parquet",
        mime="application/octet-stream",
    )
with col3:
    json_data = df_export.to_json(orient='records', indent=2)
    st.download_button(
        label="📋 JSON",
        data=json_data,
        file_name="wiki_edits_ed1.json",
        mime="application/json",
    )
