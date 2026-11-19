
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from transformers import CLIPModel, CLIPProcessor


# ==================================================
# 1. CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Dog Breed Identification",
    page_icon="🐕",
    layout="centered",
)

st.title("🐕 Dog Breed Identification")
st.write(
    "Upload a clear dog photo to identify one of our five supported breeds."
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_dog_breed_model.pth"
CLASSES_PATH = BASE_DIR / "class_names.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPECTED_CLASSES = [
    "Golden Retriever",
    "Labrador Retriever",
    "Bichon Frise",
    "Toy Poodle",
    "Border Collie",
]

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

MIN_SUPPORTED_SCORE = 0.20
MIN_SUPPORTED_MARGIN = 0.015
MIN_BREED_CONFIDENCE = 0.60
MIN_BREED_MARGIN = 0.15


# ==================================================
# 2. CHECK REQUIRED FILES
# ==================================================
if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        "Required files are missing. Keep app.py, "
        "best_dog_breed_model.pth, and class_names.json "
        "in the same folder."
    )
    st.stop()

try:
    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if class_names != EXPECTED_CLASSES:
        st.error(
            "class_names.json does not match the trained model's "
            "five classes and class order."
        )
        st.stop()

except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Could not read class_names.json: {exc}")
    st.stop()


# ==================================================
# 3. LOAD EXISTING MOBILENETV2 MODEL
# ==================================================
@st.cache_resource
def load_breed_model():
    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(class_names),
    )

    state = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model


# ==================================================
# 4. LOAD PRETRAINED CLIP MODEL
# ==================================================
@st.cache_resource
def load_clip_model():
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

    model.to(device)
    model.eval()

    return model, processor


# ==================================================
# 5. SUPPORTED AND UNSUPPORTED BREED DESCRIPTIONS
# ==================================================
SUPPORTED_PROMPTS = {
    "Golden Retriever": [
        "a photo of a Golden Retriever dog",
        "a clear picture of a Golden Retriever",
        "the face of a Golden Retriever",
    ],
    "Labrador Retriever": [
        "a photo of a Labrador Retriever dog",
        "a clear picture of a Labrador Retriever",
        "the face of a Labrador Retriever",
    ],
    "Bichon Frise": [
        "a photo of a Bichon Frise dog",
        "a clear picture of a Bichon Frise",
        "the face of a Bichon Frise",
    ],
    "Toy Poodle": [
        "a photo of a Toy Poodle dog",
        "a clear picture of a Toy Poodle",
        "the face of a Toy Poodle",
    ],
    "Border Collie": [
        "a photo of a Border Collie dog",
        "a clear picture of a Border Collie",
        "the face of a Border Collie",
    ],
}

OTHER_BREED_PROMPTS = [
    "a photo of a Chihuahua dog",
    "a photo of a Rajapalayam dog, an Indian sighthound",
    "a photo of a German Shepherd dog",
    "a photo of a Pomeranian dog",
    "a photo of a Pug dog",
    "a photo of a Beagle dog",
    "a photo of a Dalmatian dog",
    "a photo of a Siberian Husky dog",
    "a photo of a Rottweiler dog",
    "a photo of an Indian Pariah dog",
    "a photo of a Doberman Pinscher dog",
    "a photo of a Great Dane dog",
    "a photo of a Boxer dog",
    "a photo of a Dachshund dog",
    "a photo of a Shih Tzu dog",
    "a photo of a dog of an unsupported breed",
    "a photo of an unfamiliar dog breed",
]

NON_DOG_PROMPTS = [
    "a photo of a person",
    "a photo of a cat",
    "a photo of a horse",
    "a photo of a different animal that is not a dog",
    "a photo of an object without a dog",
]

ALL_GROUPS = (
    list(SUPPORTED_PROMPTS.keys())
    + ["Other / Unsupported Breed", "Not a Dog"]
)

ALL_PROMPTS = []
PROMPT_GROUPS = []

for breed, prompts in SUPPORTED_PROMPTS.items():
    for prompt in prompts:
        ALL_PROMPTS.append(prompt)
        PROMPT_GROUPS.append(breed)

for prompt in OTHER_BREED_PROMPTS:
    ALL_PROMPTS.append(prompt)
    PROMPT_GROUPS.append("Other / Unsupported Breed")

for prompt in NON_DOG_PROMPTS:
    ALL_PROMPTS.append(prompt)
    PROMPT_GROUPS.append("Not a Dog")


# ==================================================
# 6. CLIP IMAGE SCREENING
# ==================================================
def check_uploaded_image(image):
    clip_model, processor = load_clip_model()

    inputs = processor(
        text=ALL_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = clip_model(**inputs)
        prompt_scores = outputs.logits_per_image[0]

    group_scores = {}

    for group in ALL_GROUPS:
        indices = [
            i
            for i, prompt_group in enumerate(PROMPT_GROUPS)
            if prompt_group == group
        ]

        values = prompt_scores[indices]

        top_values = torch.topk(
            values,
            k=min(2, len(indices)),
        ).values

        group_scores[group] = float(top_values.mean().item())

    best_supported_breed = max(
        SUPPORTED_PROMPTS.keys(),
        key=lambda breed: group_scores[breed],
    )

    best_supported_score = group_scores[best_supported_breed]
    other_score = group_scores["Other / Unsupported Breed"]
    non_dog_score = group_scores["Not a Dog"]

    supported_scores = sorted(
        [group_scores[breed] for breed in SUPPORTED_PROMPTS],
        reverse=True,
    )

    supported_margin = supported_scores[0] - supported_scores[1]

    return {
        "best_supported_breed": best_supported_breed,
        "best_supported_score": best_supported_score,
        "other_score": other_score,
        "non_dog_score": non_dog_score,
        "supported_margin": supported_margin,
    }


# ==================================================
# 7. IMAGE PREPROCESSING
# ==================================================
weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()

breed_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=preprocess.mean,
        std=preprocess.std,
    ),
])


# ==================================================
# 8. IMAGE UPLOAD AND BREED IDENTIFICATION
# ==================================================
uploaded_file = st.file_uploader(
    "Choose a dog image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "Invalid image. Please upload a valid JPG or PNG."
        )
        st.stop()

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True,
    )

    if st.button("Identify Breed", type="primary"):

        try:
            # --------------------------------------
            # STAGE 1: CLIP SCREENING
            # --------------------------------------
            with st.spinner(
                "Checking whether the breed is supported..."
            ):
                screening = check_uploaded_image(image)

            supported_breed = screening["best_supported_breed"]
            supported_score = screening["best_supported_score"]
            other_score = screening["other_score"]
            non_dog_score = screening["non_dog_score"]
            supported_margin = screening["supported_margin"]

            # Do not display similarity scores to the user.

            if (
                non_dog_score > supported_score
                and non_dog_score > other_score
            ):
                st.error(
                    "❌ No dog confidently identified. "
                    "Please upload a clear dog photo."
                )
                st.stop()

            unsupported = (
                other_score >= supported_score
                or supported_score < MIN_SUPPORTED_SCORE
                or supported_margin < MIN_SUPPORTED_MARGIN
            )

            if unsupported:
                st.error("❌ Breed Not Supported")

                st.warning(
                    "Our dataset does not contain this breed, "
                    "or the image does not match one of our supported "
                    "breeds clearly enough."
                )

                st.info(
                    "Our dataset contains only these five breeds:\n\n"
                    "- Golden Retriever\n"
                    "- Labrador Retriever\n"
                    "- Bichon Frise\n"
                    "- Toy Poodle\n"
                    "- Border Collie\n\n"
                    "Please upload a clear photo of one of these breeds."
                )
                st.stop()

            # --------------------------------------
            # STAGE 2: MOBILENETV2 BREED PREDICTION
            # --------------------------------------
            with st.spinner("Analyzing the dog's breed..."):
                model = load_breed_model()

                input_tensor = (
                    breed_transform(image)
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.inference_mode():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]

            sorted_indices = (
                probabilities.argsort(descending=True).tolist()
            )

            predicted_index = sorted_indices[0]

            confidence = float(
                probabilities[predicted_index].item()
            )

            second_score = float(
                probabilities[sorted_indices[1]].item()
            )

            model_margin = confidence - second_score

            # Both checks must agree before displaying a prediction.
            if (
                class_names[predicted_index] != supported_breed
                or confidence < MIN_BREED_CONFIDENCE
                or model_margin < MIN_BREED_MARGIN
            ):
                st.error("❌ Breed Not Supported or Not Confirmed")

                st.warning(
                    "The image could not be matched confidently "
                    "to one of our five supported breeds. Please "
                    "upload a clearer photo of a supported breed."
                )
                st.stop()

            # --------------------------------------
            # STAGE 3: DISPLAY THE RESULT
            # --------------------------------------
            st.success(
                f"Predicted breed: **{class_names[predicted_index]}**"
            )

            st.metric(
                "Breed model confidence",
                f"{confidence:.1%}",
            )

            st.subheader("Breed prediction scores")

            for index in sorted_indices:
                score = float(probabilities[index].item())

                st.write(
                    f"**{class_names[index]}** — {score:.1%}"
                )

                st.progress(score)

            st.caption(
                "The model supports five breeds only. Predictions "
                "and unsupported-breed screening are estimates and "
                "may make mistakes."
            )

        except Exception as exc:
            st.error(
                f"An error occurred while analyzing the image: {exc}"
            )

st.divider()

st.caption(
    "MobileNetV2 five-breed classifier with pretrained CLIP screening"
)
