# Gunakan image python yang stabil
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Salin requirements dan install dependencies
COPY requirement.txt ./
RUN pip install --upgrade pip && pip install -r requirement.txt

# Salin semua file project ke container
COPY . .

# Debug: cek isi folder webapp/webapp di dalam container
RUN ls -lR /app/webapp/webapp

# Set PYTHONPATH agar Python bisa menemukan webapp package
ENV PYTHONPATH="/app:/app/src:/app/webapp"

# Jalankan collectstatic dari webapp directory
WORKDIR /app/webapp
RUN python manage.py collectstatic --noinput || true

# Expose port Railway (Railway pakai $PORT env var)
EXPOSE 8000

# Perintah start (run from webapp directory)
CMD gunicorn webapp.wsgi:application --bind 0.0.0.0:8000
