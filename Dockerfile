FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY rlwatch.yaml ./
COPY examples/ ./examples/

# Install the package
RUN pip install --no-cache-dir -e .

# Expose Streamlit dashboard port
EXPOSE 8501

# Default: run the simulated demo then launch dashboard
CMD ["sh", "-c", "python examples/simulate_grpo_run.py && rlwatch dashboard --log-dir ./rlwatch_logs"]
