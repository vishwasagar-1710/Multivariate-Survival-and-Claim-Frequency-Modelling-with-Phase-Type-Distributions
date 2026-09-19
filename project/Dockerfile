FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Artifacts in models/ are expected to already be trained (see README).
# To retrain inside the image instead, uncomment:
# RUN python src/train_model.py

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
