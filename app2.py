import streamlit as st

st.set_page_config(page_title="Image Caption Generator", layout="centered")

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("blip_caption_model")
    model = BlipForConditionalGeneration.from_pretrained("blip_caption_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_blip_model()

st.title("🖼️ Image Caption Generator")
st.markdown("Upload an image to generate **5 diverse captions** and evaluate with optional ground truth caption.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
ground_truth_caption = st.text_input("✏️ (Optional) Enter known caption for evaluation")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Generate Captions"):
        st.markdown("### 📋 Captions with Evaluation Metrics")
        inputs = processor(images=image, return_tensors="pt").to(device)

        if ground_truth_caption:
            gt_tokens = word_tokenize(ground_truth_caption.lower().strip())

        for i in range(5):
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                num_return_sequences=1
            )
            caption = processor.decode(output[0], skip_special_tokens=True)
            st.markdown(f"**Caption {i+1}:** _{caption}_")

            # METEOR Score
            if ground_truth_caption:
                pred_tokens = word_tokenize(caption.lower().strip())
                meteor = meteor_score([gt_tokens], pred_tokens)
                st.markdown(f"`METEOR Score:` {meteor:.4f}")
            else:
                st.markdown("`METEOR Score:` _(Enter ground truth caption to compute)_")

            # Precision, Recall, F1, Accuracy
            if ground_truth_caption:
                all_words = list(set(gt_tokens + pred_tokens))
                y_true = [1 if word in gt_tokens else 0 for word in all_words]
                y_pred = [1 if word in pred_tokens else 0 for word in all_words]

                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                accuracy = accuracy_score(y_true, y_pred)

                st.markdown(
                    f"`Precision:` {precision:.4f} | `Recall:` {recall:.4f} | `F1-Score:` {f1:.4f} | `Accuracy:` {accuracy:.4f}"
                )
