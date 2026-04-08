# 1. The Base OS
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system-level tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY requirements first (Smart caching)
COPY requirements.txt .

# 5. Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your app
COPY . .

# 7. Render Environment Setup
ENV PORT=10000
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 8. Expose the fallback port (Render ignores this, but it's good practice)
EXPOSE 10000

# 9. The Ignition Switch
# Using strict exec-shell form to guarantee $PORT evaluation, with CORS disabled for Cloudflare.
CMD ["streamlit", "run", "ReAct_Agent.py", "--server.port=10000", "--server.address=0.0.0.0"]