# Use a secure and lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
# This is where the older NumPy will be installed, resolving the conflict.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (app.py, models, static, templates)
COPY . .

# Use Gunicorn to run the application (Render's standard web server)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
