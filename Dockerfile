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