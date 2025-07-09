# Gunakan image python yang stabil
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Salin requirements dan install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Salin semua file project ke container
COPY . .

# Jalankan collectstatic jika pakai Django staticfiles
RUN python manage.py collectstatic --noinput || true

# Expose port Railway (Railway pakai $PORT env var)
EXPOSE 8000

# Perintah start (ganti sesuai struktur project Anda)
CMD gunicorn webapp.wsgi:application --bind 0.0.0.0:8000
