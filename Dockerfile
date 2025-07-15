# Use specific Python version (e.g. 3.11)
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Run Streamlit app (if applicable)
CMD ["streamlit", "run", "your_app.py"]

