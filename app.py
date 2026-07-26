import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Quote Generator",
    page_icon="📝",
    layout="centered"
)

# =====================================
# HEADER
# =====================================

st.markdown(
    """
    <h1 style='text-align:center;color:#4CAF50;'>
    📝 AI Quote Generator
    </h1>
    <h4 style='text-align:center;color:gray; margin-top:-10px;'>
    Built By - Rahul Maurya
    </h4>
    """,
    unsafe_allow_html=True
)

# =====================================
# LOAD FILES
# =====================================

@st.cache_resource
def load_files():

    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_files()

# =====================================
# WORD INDEX MAPPING
# =====================================

index_to_word = {}

for word, index in tokenizer.word_index.items():
    index_to_word[index] = word

# =====================================
# PREDICT NEXT WORD
# =====================================

def predict_next_word(text):

    text = text.lower()

    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    prediction = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(prediction)

    return index_to_word.get(predicted_index, "")

# =====================================
# GENERATE TEXT
# =====================================

def generate_text(seed_text, n_words):

    generated = seed_text

    for _ in range(n_words):

        next_word = predict_next_word(generated)

        if next_word == "":
            break

        generated += " " + next_word

    return generated

# =====================================
# UI DESIGN
# =====================================

st.markdown("---")

st.write("Enter a starting phrase and let the AI complete it.")

seed_text = st.text_input(
    "Enter Starting Text",
    placeholder="Example: life is"
)

num_words = st.slider(
    "Number of Words to Generate",
    min_value=5,
    max_value=50,
    value=20
)

st.markdown("")

if st.button("🚀 Generate Quote", use_container_width=True):

    if seed_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Generating..."):

            generated_quote = generate_text(
                seed_text,
                num_words
            )

        st.success("Quote Generated Successfully!")

        st.markdown("### Generated Text")

        st.markdown(
            f"""
            <div style="
            background-color:#f5f5f5;
            padding:20px;
            border-radius:10px;
            font-size:20px;
            font-weight:bold;
            color:black;
            ">
            {generated_quote}
            </div>
            """,
            unsafe_allow_html=True
        )