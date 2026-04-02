.PHONY: install run test lint clean

# Install dependencies
install:
	pip install -r requirements.txt

# Download required NLTK data
nltk-data:
	python3 -c "import nltk; nltk.download('stopwords')"

# Run the app (development)
run:
	streamlit run app.py

# Run the app (production / headless)
start:
	streamlit run app.py --server.headless true --server.port 8501

# Run tests
test:
	python3 -m pytest test_pipeline.py test_qa_parser.py test_topic_grouper.py -v

# Run tests quietly
test-q:
	python3 -m pytest test_pipeline.py test_qa_parser.py test_topic_grouper.py -q

# Clear embedding cache (forces rebuild on next start)
clear-cache:
	rm -f database/qa_embeddings.pkl

# Clean Python cache files
clean:
	find . -type d -name __pycache__ -not -path "*/venv*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "*/venv*" -delete 2>/dev/null || true
	rm -rf .pytest_cache
