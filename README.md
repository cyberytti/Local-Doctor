# Local Doctor

<div align="center">
  <img src="https://github.com/cyberytti/Local-Doctor/blob/main/assets/logo.png" alt="Local Doctor Logo" width="200"/>
</div>

**Local Doctor** is a privacy-first, local AI tool designed for disease analysis. By providing a list of symptoms, the system predicts potential diseases and suggests basic precautions. All processing occurs locally on your machine, ensuring that sensitive health data remains private and secure.

> **⚠️ Disclaimer:** This project is currently in an experimental phase and is intended for educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 🎥 Demo

<div align="center">
  <img src="https://github.com/cyberytti/Local-Doctor/blob/main/assets/demo_use_smaller.gif" alt="Local Doctor Demo" width="600"/>
</div>

## 🏗️ Architecture & Model Details

This repository provides the Streamlit inference interface for the [local_doctor-360M](https://huggingface.co/Ahahajij182u2/local_doctor-360M) model. The core intelligence is driven by a custom-trained Large Language Model (LLM).

### Model Specifications
*   **Base Model:** [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
*   **Architecture:** LoRA Fine-tuned
*   **Model Name:** `local_doctor-360M`
*   **Creator:** [cyberytti](https://github.com/cyberytti)

### Training Data
The model was trained on a combined dataset derived from two primary sources:
1.  `Diseases_Symptoms.csv`
2.  `disease_sympts_prec_full.csv`

These were merged into a final training dataset: `combined_disease_dataset.csv`.
You can find the datasets here:  
[📂 Datasets](https://github.com/cyberytti/Local-Doctor/tree/main/datasets)

### Training Environment
*   **Framework:** Unsloth
*   **Hardware:** Single NVIDIA L4 GPU
*   **Status:** Mid-training phase. While the current 350M parameter model shows promising results, further fine-tuning is planned to enhance reliability for production use.

You can view the training scripts and notebooks here:  
[📂 Training Code](https://github.com/cyberytti/Local-Doctor/tree/main/train)

## 🚀 Installation & Setup

Follow these steps to set up and run Local Doctor locally.

### Prerequisites
*   Python 3.10+
*   Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/cyberytti/Local-Doctor.git
cd Local-Doctor
```

### Step 2: Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

Launch the Streamlit user interface:

```bash
streamlit run streamlit_app.py
```

> **Note:** On the first run, the application will automatically download the `local_doctor-360m` model from Hugging Face. Please ensure you have a stable internet connection during this initial setup.

This project and the underlying `local_doctor-360M` model were created by **cyberytti**. This repository serves to share the complete recipe, including data preparation, training methodology, and inference implementation.

For inquiries or contributions, please refer to the GitHub repository issues section.
