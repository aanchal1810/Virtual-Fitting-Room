# Makefile for Virtual Dressing Room

# Activate virtual environment
VENV = venv

# Run the FastAPI app
run:
	@echo "🚀 Starting FastAPI with Uvicorn..."
	$(VENV)/Scripts/uvicorn app.main:app --reload || $(VENV)/bin/uvicorn app.main:app --reload

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Save dependencies
freeze:
	pip freeze > requirements.txt

# Clean pycache
clean:
	@echo "🧹 Cleaning __pycache__ folders..."
	find . -type d -name "__pycache__" -exec rm -r {} +
