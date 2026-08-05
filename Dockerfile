FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD [
    "streamlit",
    "run",
    "app.py",
    "--server.address=0.0.0.0",
    "--server.port=7860",
    "--server.headless=true",
    "--browser.gatherUsageStats=false"
]
