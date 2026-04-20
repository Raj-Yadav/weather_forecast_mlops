# Use a minimal Python runtime as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# --- LAYER CACHING OPTIMIZATION ---
# Copy requirements BEFORE the rest of the code.
# Docker caches layers. If only your Python code changes
# (but not requirements.txt), Docker reuses the pip install layer —
# making subsequent builds significantly faster.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Document that the container listens on port 5000
EXPOSE 5000

# Start the Flask application
CMD ["python", "backend/app.py"]