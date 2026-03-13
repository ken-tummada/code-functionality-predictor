import streamlit as st
import json
import yaml
import csv
import io
import glob
import random

st.set_page_config(page_title="Text Annotation Tool")

ANNOTATION_FILE_OUTPUT = "annotations.csv"

if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def save_annotations(corpus, criteria):
    output = io.StringIO()
    headers = ["user_name", "id"] + list(criteria.keys())
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    user_name = st.session_state.get("user_name", "")
    for item in corpus:
        item_id = item["id"]
        scores = st.session_state.annotations.get(item_id, {})
        if scores:
            row = {"user_name": user_name, "id": item_id}
            for criterion in criteria.keys():
                row[criterion] = scores.get(criterion, "")
            writer.writerow(row)

    return output.getvalue()


@st.cache_data
def load_corpus_and_criteria():
    data = [
        ("outputs/desc-gen-gpt-5-mini", "gpt-5-mini"),
        ("outputs/desc-gen-llama-3-8b", "llama-3.1-8b"),
        ("outputs/desc-gen-sonnet-4.5", "sonnet-4.5"),
    ]
    criteria_file = "annotate_ui/criteria.yml"

    corpus = []

    for dir, name in data:
        predictions = []
        for file_name in glob.glob(f"{dir}/*.json"):
            with open(file_name, "r") as f:
                predictions.extend(json.load(f)["preds"])

        for i, pred in enumerate(predictions):
            corpus.append({"id": f"{name}-{i}", "text": pred})

    with open(criteria_file, "r") as f:
        criteria = yaml.safe_load(f)

    # TODO: shuffle data, also find a way to resume session, idk how. Mb set a seed and do shuffle?
    # random.shuffle(corpus)
    return corpus, criteria


corpus, criteria = load_corpus_and_criteria()

if not st.session_state.logged_in:
    st.title("Login")
    with st.form("login_form"):
        name = st.text_input("Name")
        starting_idx = st.number_input("Jump to", min_value=1, max_value=len(corpus))
        submit = st.form_submit_button("Start")
        if submit and name:
            st.session_state.user_name = name
            st.session_state.logged_in = True
            st.session_state.current_idx = starting_idx - 1
            st.rerun()
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


st.subheader("Corpus")
with st.container(border=True):
    st.markdown(item["text"])


st.subheader("Criteria")
scores = st.session_state.annotations.get(item_id, {})

new_scores = {}
for criterion, options in criteria.items():
    label = f"{criterion}"
    value_desc = scores.get(criterion)
    if value_desc is not None:
        default_idx = next(
            (i for i, opt in enumerate(options) if opt[1] == value_desc), 0
        )
    else:
        default_idx = 0
    choices = [f"{opt[0]}: {opt[1]}" for opt in options]
    selected = st.selectbox(label, choices, index=default_idx, key=criterion)
    value = int(selected.split(":")[0])
    new_scores[criterion] = value

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
if st.download_button(
    "Download Annotations",
    annotation_data,
    file_name=ANNOTATION_FILE_OUTPUT,
    mime="text/csv",
    icon=":material/download:",
    width="stretch",
):
    st.success(f"Annotation saved as {ANNOTATION_FILE_OUTPUT}!", icon="🎉")
