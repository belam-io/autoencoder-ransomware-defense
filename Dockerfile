# Start from the base image
FROM python:3.11-slim

# Create a non-root user and set permissions
# This addresses the root warning and improves security
RUN useradd -m appuser
RUN mkdir /home/appuser/.local
RUN chown -R appuser:appuser /home/appuser

# Set the working directory
WORKDIR /app

# Copy requirements before installing to leverage Docker caching
COPY requirements.txt .

# ----------------------------------------------------
# 🌟 VIRTUAL ENVIRONMENT FIX 🌟
# ----------------------------------------------------

# Create the virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV

# Activate the virtual environment
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install core requirements inside the venv
RUN pip install --no-cache-dir -r requirements.txt

# Install streamlit explicitly (if not in requirements.txt)
RUN pip install streamlit

# ----------------------------------------------------
# Install system packages (needs root access)
# netcat is needed for the Docker readiness checks
# ----------------------------------------------------
RUN apt-get update && apt-get install -y netcat-openbsd

# Switch to the non-root user for all subsequent operations
USER appuser

# Copy the application code
COPY . .

# Set PYTHONPATH for the application
ENV PYTHONPATH=/app

# Default command (will be overridden by docker-compose)
CMD ["python3"]