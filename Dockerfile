# Use the official, lightweight Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files to disk and ensure console logs are real-time
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for PostgreSQL connections
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first (this optimizes Docker's build cache)
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the default port (Northflank will override this dynamically, but it's good practice)
EXPOSE 10000

# Run the app using Gunicorn for production-grade concurrency
# Notice we use mulity_hes:app based on your filename
CMD ["sh", "-c", "gunicorn mulity_hes:app --workers 2 --threads 2 --bind 0.0.0.0:${PORT:-10000}"]
