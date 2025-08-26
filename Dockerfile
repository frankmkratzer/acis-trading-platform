# ACIS Trading Platform Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /opt/acis-trading

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    cron \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create log directory
RUN mkdir -p /var/log/acis-trading

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -r -s /bin/false acis && \
    chown -R acis:acis /opt/acis-trading /var/log/acis-trading

# Set timezone
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=5m --retries=3 \
    CMD python -c "import psycopg2; psycopg2.connect('$POSTGRES_URL')" || exit 1

# Switch to non-root user
USER acis

# Set Python path
ENV PYTHONPATH=/opt/acis-trading

# Default command
CMD ["python", "smart_scheduler.py"]