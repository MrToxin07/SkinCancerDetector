# Use a Python base image that has better compatibility
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
# The build will now use the constraints to install compatible versions
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (app.py, models, static, templates)
COPY . .

# Use Gunicorn to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
