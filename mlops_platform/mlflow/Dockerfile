FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow psycopg2-binary

EXPOSE 5000

# command is overridden by docker-compose.yml; this default lets the image
# also be run standalone for debugging
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
