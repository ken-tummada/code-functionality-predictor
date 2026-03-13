import streamlit as st
import json
import yaml
import csv
import io
import os
import glob

st.set_page_config(page_title="Text Annotation Tool", layout="wide")


if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def save_annotations(corpus, criteria):
    output = io.StringIO()
    headers = ["user_name", "id"] + [v["alias"] for v in criteria.values()]
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    user_name = st.session_state.get("user_name", "")
    for item in corpus:
        item_id = item["id"]
        scores = st.session_state.annotations.get(item_id, {})
        if scores:
            row = {"user_name": user_name, "id": item_id}
            for criterion_key, config in criteria.items():
                alias = config["alias"]
                row[alias] = scores.get(alias, "")
            writer.writerow(row)

    return output.getvalue()


@st.cache_data
def load_corpus_and_criteria():
    ds_loc = "annotate_ui/data/"
    criteria_file = "annotate_ui/criteria.yml"

    json_files = glob.glob(os.path.join(ds_loc, "**/*.json"), recursive=True)
    corpus = []
    for filepath in sorted(json_files):
        rel_path = os.path.relpath(filepath, ds_loc)
        parts = rel_path.split(os.sep)
        folder_name = parts[0]
        with open(filepath, "r") as f:
            data = json.load(f)
        preds = data.get("preds", [])
        for idx, pred in enumerate(preds):
            corpus.append({"id": f"{folder_name}-{idx}", "text": pred})

    with open(criteria_file, "r") as f:
        criteria = yaml.safe_load(f)

    return corpus, criteria


corpus, criteria = load_corpus_and_criteria()

if not st.session_state.logged_in:
    st.set_page_config(layout="centered")
    st.title("Hello!")
    with st.form("login_form"):
        name = st.text_input("Name")
        starting_idx = st.number_input("Jump to", min_value=1, max_value=len(corpus))
        submit = st.form_submit_button("Start")
        if submit and name:
            st.session_state.user_name = name
            st.session_state.logged_in = True
            st.session_state.current_idx = starting_idx - 1
            st.rerun()

    with st.expander("How to annotate", expanded=True):
        st.markdown("""
        1. **Read the text** - Carefully read the description and code in the Corpus section
        2. **Rate each criterion** - Use the dropdown menus to select a score (1-5) for each criterion
        3. **Navigate** - Use Previous/Next buttons to move between items
        4. **Track progress** - Your progress is shown at the top
        5. **Download** - Click "Download Annotations" to save your work as CSV
        """)

    st.stop()


current_idx = st.session_state.current_idx
total = len(corpus)

item = corpus[current_idx]
item_id = item["id"]

st.title(f"Hello {st.session_state.user_name} 👋")

st.subheader("Progress")
progress = (current_idx + 1) / total
st.progress(progress)
st.write(f"{current_idx + 1} / {total} annotated")


col_corpus, col_criteria = st.columns(2)

with col_corpus:
    st.subheader("Corpus")
    with st.container(border=True, height=560):
        st.markdown(item["text"])

with col_criteria:
    st.subheader("Criteria")
    scores = st.session_state.annotations.get(item_id, {})

    new_scores = {}
    for criterion, config in criteria.items():
        alias = config["alias"]
        options = config["scores"]
        label = criterion
        value_desc = scores.get(alias)
        if value_desc is not None:
            default_idx = next(
                (i for i, opt in enumerate(options) if opt[1] == value_desc), 0
            )
        else:
            default_idx = 0
        choices = [f"{opt[0]}: {opt[1]}" for opt in options]
        selected = st.selectbox(label, choices, index=default_idx, key=alias)
        value = int(selected.split(":")[0])
        new_scores[alias] = value

    st.session_state.annotations[item_id] = new_scores

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button(
            "Previous",
            disabled=current_idx == 0,
            width="stretch",
            icon=":material/arrow_back_ios:",
        ):
            st.session_state.current_idx -= 1
            st.rerun()

    with col_next:
        if st.button(
            "Next",
            disabled=current_idx == total - 1,
            width="stretch",
            icon=":material/arrow_forward_ios:",
            type="primary",
        ):
            st.session_state.current_idx += 1
            st.rerun()

    annotation_data = save_annotations(corpus, criteria)
    file_name = "annotations.csv"
    if st.download_button(
        "Download Annotations",
        annotation_data,
        file_name=file_name,
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    ):
        st.success(f"Annotation saved as {file_name}!", icon="🎉")
