import os
import torch
import streamlit as st
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

st.set_page_config(page_title="Chest X-Ray Report Assistant", page_icon="🫁", layout="centered")

BASE_MODEL_ID = "google/paligemma-3b-pt-224"
ADAPTER_PATH = "adapter"
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2


@st.cache_resource
def load_model():
    hf_token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, token=hf_token)

    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        token=hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return processor, model


st.title("🫁 Chest X-Ray Report Assistant")
st.caption(
    "A LoRA fine-tuned PaliGemma-3B vision-language model for chest X-ray "
    "findings generation, trained on the IU-Xray dataset."
)

st.warning(
    "⚠️ **Research / educational project only — not a diagnostic tool.** "
    "This model has known limitations, including a bias toward generic "
    "'normal' findings and occasional fabricated details on abnormal cases. "
    "See the README for a full, honest breakdown of its evaluation."
)

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Generate Findings", type="primary"):
        with st.spinner("Loading model..."):
            processor, model = load_model()

        with st.spinner("Analyzing image..."):
            prompt = "<image> Describe the findings in this chest X-ray."
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                )

            input_len = inputs["input_ids"].shape[-1]
            generated_text = processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )

        st.subheader("Generated Findings")
        st.write(generated_text)

st.divider()
st.caption(
    "Built as a learning project exploring vision-language model fine-tuning "
    "for medical imaging. Base model: google/paligemma-3b-pt-224. "
    "Fine-tuned via LoRA on the IU-Xray dataset (Indiana University Chest X-rays)."
)
