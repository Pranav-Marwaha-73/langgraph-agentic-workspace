# 1. The Base OS: We start with a lightweight Linux server running Python 3.11
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system-level tools (prevents errors when installing database drivers)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY the requirements first (Smart caching: makes future rebuilds 10x faster)
COPY requirements.txt .

# 5. Install all your Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your app's code into the container
COPY . .

# 7. Tell the container which port Streamlit uses
EXPOSE 8501

# 8. Set Streamlit's production environment variables (stops annoying popups and forces cloud mode)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 9. The Ignition Switch: The exact command to boot your app
CMD ["streamlit", "run", "front2.py"]