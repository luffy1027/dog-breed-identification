# Dog Breed Identification

A deep learning web application built with **PyTorch**, **MobileNetV2**, **CLIP**, and **Streamlit** that identifies dog breeds from uploaded images.

## Features
- **Deep Learning Classification:** Uses a fine-tuned MobileNetV2 model (`best_dog_breed_model.pth`) alongside CLIP for breed recognition.
- **Interactive Web Interface:** Simple image upload and instant prediction built with Streamlit.

## Project Structure
- `app.py` — Main Streamlit application
- `best_dog_breed_model.pth` — Trained PyTorch model weights
- `class_names.json` — Mapping of model output indices to dog breed names
- `requirements.txt` — Python dependencies

## How to Run Locally

1. **Clone the repository:**
   `git clone https://github.com/luffy1027/dog-breed-identification.git`
   `cd dog-breed-identification`

2. **Install dependencies:**
   `pip install -r requirements.txt`

3. **Run the Streamlit app:**
   `streamlit run app.py`
