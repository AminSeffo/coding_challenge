# TinyBERT Sentiment Analysis API

## Overview
This project provides a **REST API** for sentiment analysis using a fine-tuned **TinyBERT** model. The API is built with **FastAPI** and deployed using **Podman**.

## Features
- **Text sentiment prediction** (positive/negative/neutral/irrelevant)
- **FastAPI for REST API**
- **Runs in a containerized environment using Podman**
- **TinyBERT for lightweight NLP inference**

---

## Jupyter Notebook Analysis
The included Jupyter Notebook contains steps for:

### **1. Device Selection**
- Uses `torch.device` to select `mps` (for Mac GPUs) or CPU.
- Ensures compatibility with different hardware.

### **2. Data Loading**
- Reads training (`training.csv`) and validation (`validation.csv`) datasets.
- Uses `pandas` for data handling.
- Renames dataset columns to `ID`, `Topic`, `Sentiment`, and `Text`.

### **3. Preprocessing**
- Tokenization and stopword removal using NLTK.
- Label encoding of sentiment classes using `sklearn`.
- Data splitting into training and testing sets with `train_test_split`.

### **4. Visualization**
- Uses `matplotlib` and `seaborn` to analyze and visualize data distributions.

### **5. Model Training**
- Fine-tunes TinyBERT on the labeled sentiment dataset.
- Utilizes PyTorch for training the classification model.

### **6. Evaluation**
- Evaluates model performance using standard metrics.
- Runs inference on the validation dataset to check accuracy and predictions.

---

## Setup Instructions

### **1. Train and Save the Model**
Ensure you have trained and saved your TinyBERT model:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "tinybert_sentiment_model_v2"
model = AutoModelForSequenceClassification.from_pretrained("google/bert-tiny")
tokenizer = AutoTokenizer.from_pretrained("google/bert-tiny")

# Save the trained model
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
```
Ensure `tinybert_sentiment_model_v2/` exists in your project directory before proceeding.

---

### **2. Install Podman (If Not Installed)**
#### **Mac**:
```bash
brew install podman
podman machine init
podman machine start
```

#### **Ubuntu/Debian**:
```bash
sudo apt update && sudo apt install -y podman
```

#### **Windows (via WSL2)**:
Use **Podman Desktop** or install via WSL:
```bash
winget install --id=redhat.Podman-Desktop
```

---

### **3. Build the API**

#### **Create `requirements.txt`**
```txt
fastapi
torch
transformers
uvicorn
```

#### **Create `Dockerfile`**
```dockerfile
# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code and model
COPY app.py .
COPY tinybert_sentiment_model_v2/ tinybert_sentiment_model_v2/

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### **Build the Container**
```bash
podman build -t sentiment-api .
```

#### **Run the API in a Container**
```bash
podman run -p 8000:8000 sentiment-api
```

---

### **4. Testing the API**
Once the container is running, test the API using `curl`:
```bash
curl -X 'POST' 'http://localhost:8000/predict' -H 'Content-Type: application/json' -d '{"text": "I love this product!"}'
```

Expected Response:
```json
{
    "label": "positive",
    "score": 0.95
}
```

---

### **5. Managing the Podman Container**
#### **Check Running Containers**
```bash
podman ps
```

#### **Stop the Container**
```bash
podman stop <container_id>
```

#### **Remove the Container**
```bash
podman rm <container_id>
```

---

### **6. Deployment Considerations**
- To deploy this API on a remote server, use **Podman systemd service**:
  ```bash
  podman generate systemd --name sentiment-api --files --new
  systemctl --user enable --now container-sentiment-api.service
  ```
- To scale it, use **Podman pods** for multi-container orchestration.

---

## Conclusion
This project provides a simple and efficient way to deploy a **TinyBERT Sentiment Analysis API** using **FastAPI** and **Podman**. The Jupyter Notebook further details data preprocessing, visualization, and training steps to fine-tune the sentiment analysis model.

