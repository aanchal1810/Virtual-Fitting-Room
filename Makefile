# Makefile for Virtual Dressing Room

VENV = venv
REQ = requirements.txt
PYTHON = python3
UVICORN = $(VENV)/bin/uvicorn

# Default target
run: $(VENV)/bin/activate
	@echo "🚀 Starting FastAPI with Uvicorn..."
	$(VENV)/bin/pip install -r $(REQ)
	$(UVICORN) app.main:app --reload

# Create virtual environment if it doesn't exist
$(VENV)/bin/activate: $(REQ)
	@echo "🐍 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment ready!"

install: $(VENV)/bin/activate
	@echo "📦 Installing dependencies..."
	$(VENV)/bin/pip install -r $(REQ)

freeze:
	$(VENV)/bin/pip freeze > $(REQ)

clean:
	@echo "🧹 Cleaning __pycache__ folders..."
	find . -type d -name "__pycache__" -exec rm -r {} +
