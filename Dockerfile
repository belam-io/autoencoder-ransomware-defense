FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install streamlit explicitly
RUN pip install streamlit

# Add this line to your Dockerfile, preferably after the pip installs:
RUN apt-get update && apt-get install -y netcat-openbsd

COPY . .

ENV PYTHONPATH=/app

# Default overridden by docker-compose
CMD ["python3"]
