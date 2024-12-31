# Dockerfile
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the flow script
COPY prefect .

# Set the default command to run the flow
CMD ["python", "deployment.py"]
