# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables to prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Set the working directory
WORKDIR /app

# Copy only requirements first (for caching layers)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Ensure proper permissions (avoids issues in Docker)
RUN chmod -R 755 /app

# Expose the port Render uses
EXPOSE 8080

# Start Gunicorn with 4 workers
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "server:app"]
