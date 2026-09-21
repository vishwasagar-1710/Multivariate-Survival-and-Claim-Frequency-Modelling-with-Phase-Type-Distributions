FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model and generate all artifacts/plots at image build time so
# the container starts ready to serve.
RUN python data/generate_data.py && python scripts/run_pipeline.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
