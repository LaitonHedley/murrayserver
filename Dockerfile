
FROM python:3.12-slim

RUN pip install --no-cache-dir aiohttp numpy

WORKDIR /tmp

COPY murrayserver murrayserver
COPY setup.py .
RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["python3", "-u", "-m", "murrayserver"]
