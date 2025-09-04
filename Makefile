# Makefile for RAG MS MARCO template
# Provides convenient commands for development and deployment

.PHONY: help install build-index serve stop clean logs health test eval loadtest lint format check-env

# Default target
help: ## Show this help message
	@echo "RAG MS MARCO Template - Available Commands"
	@echo "=========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  1. cp configs/.env.example .env"
	@echo "  2. Edit .env with your configuration"
	@echo "  3. make build-index"
	@echo "  4. make serve"
	@echo "  5. make health"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

check-env: ## Check if .env file exists and is configured
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found. Copy configs/.env.example to .env and configure it."; \
		exit 1; \
	fi
	@echo "✅ .env file found"
	@if grep -q "your-secure-bearer-token-here" .env; then \
		echo "⚠️  Please update AUTH_BEARER_TOKEN in .env"; \
	fi
	@if grep -q "your-api-key-here" .env; then \
		echo "⚠️  Please update OPENAI_API_KEY in .env"; \
	fi

install: ## Install Python dependencies
	@echo "Installing dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		uv sync; \
	elif [ -f "requirements.txt" ]; then \
		pip install -r requirements.txt; \
	else \
		echo "❌ No uv or requirements.txt found"; \
		exit 1; \
	fi

# ============================================================================
# INDEX MANAGEMENT
# ============================================================================

build-index: check-env ## Build Qdrant index from MS MARCO data
	@echo "Building index from MS MARCO dataset..."
	@python tools/build_index.py --verbose

build-index-small: check-env ## Build index with small sample (10k passages)
	@echo "Building small index for testing..."
	@python tools/build_index.py --sample-size 10000 --verbose

build-index-recreate: check-env ## Rebuild index (delete existing collection)
	@echo "Rebuilding index (deleting existing collection)..."
	@python tools/build_index.py --recreate --verbose

build-index-dry-run: check-env ## Dry run index build (no Qdrant operations)
	@echo "Dry run index build..."
	@python tools/build_index.py --dry-run --verbose --output-manifest build_manifest.json

# ============================================================================
# SERVICE MANAGEMENT
# ============================================================================

serve: check-env ## Start all services with Docker Compose
	@echo "Starting RAG services..."
	@cd ops && docker compose up -d

serve-with-monitoring: check-env ## Start services with monitoring stack
	@echo "Starting RAG services with monitoring..."
	@cd ops && docker compose --profile monitoring up -d

stop: ## Stop all services
	@echo "Stopping services..."
	@cd ops && docker compose down

restart: stop serve ## Restart all services

clean: ## Stop services and remove volumes
	@echo "Cleaning up services and data..."
	@cd ops && docker compose down -v
	@docker system prune -f

# ============================================================================
# DEVELOPMENT
# ============================================================================

serve-api: check-env ## Start only the API server (for development)
	@echo "Starting API server..."
	@python -m app serve --host 0.0.0.0 --port 8000

serve-api-dev: check-env ## Start API server with auto-reload
	@echo "Starting API server in development mode..."
	@uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

validate: check-env ## Validate all system components
	@echo "Validating system components..."
	@python -m app validate

# ============================================================================
# MONITORING AND HEALTH
# ============================================================================

logs: ## Show logs from all services
	@cd ops && docker compose logs -f

logs-api: ## Show logs from API service only
	@cd ops && docker compose logs -f rag_api

logs-nginx: ## Show logs from Nginx service only
	@cd ops && docker compose logs -f nginx

health: check-env ## Check health of all services
	@echo "Checking service health..."
	@echo "1. API Health:"
	@curl -s -H "Authorization: Bearer $$(grep AUTH_BEARER_TOKEN .env | cut -d'=' -f2)" \
		http://localhost:8080/v1/health | python -m json.tool || echo "❌ API health check failed"
	@echo ""
	@echo "2. Nginx Status:"
	@curl -s http://localhost:8080/nginx/health || echo "❌ Nginx health check failed"
	@echo ""
	@echo "3. Qdrant Health:"
	@curl -s http://localhost:6333/health | python -m json.tool || echo "❌ Qdrant health check failed"

status: ## Show Docker Compose service status
	@cd ops && docker compose ps

# ============================================================================
# TESTING AND EVALUATION
# ============================================================================

test: check-env ## Run basic system tests
	@echo "Running system tests..."
	@python -m app validate

eval: check-env ## Run retrieval evaluation
	@echo "Running evaluation..."
	@python tools/evaluate.py --max-queries 1000 --output evaluation_results

eval-quick: check-env ## Run quick evaluation (100 queries)
	@echo "Running quick evaluation..."
	@python tools/evaluate.py --max-queries 100 --verbose

eval-full: check-env ## Run full evaluation (5000 queries)
	@echo "Running full evaluation..."
	@python tools/evaluate.py --max-queries 5000 --output full_evaluation_results --detailed

loadtest: check-env ## Run load test
	@echo "Running load test..."
	@python tools/load_test.py \
		--token "$$(grep AUTH_BEARER_TOKEN .env | cut -d'=' -f2)" \
		--users 5 --requests 10 --output loadtest_results.json

loadtest-stress: check-env ## Run stress load test
	@echo "Running stress load test..."
	@python tools/load_test.py \
		--token "$$(grep AUTH_BEARER_TOKEN .env | cut -d'=' -f2)" \
		--users 20 --requests 25 --output stress_test_results.json

# ============================================================================
# CODE QUALITY
# ============================================================================

lint: ## Run code linting
	@echo "Running linters..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check app/ tools/; \
	else \
		echo "⚠️  ruff not found, skipping lint"; \
	fi

format: ## Format code
	@echo "Formatting code..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format app/ tools/; \
	else \
		echo "⚠️  ruff not found, skipping format"; \
	fi

typecheck: ## Run type checking
	@echo "Running type checking..."
	@if command -v mypy >/dev/null 2>&1; then \
		mypy app/; \
	else \
		echo "⚠️  mypy not found, skipping typecheck"; \
	fi

# ============================================================================
# UTILITY COMMANDS
# ============================================================================

requirements: ## Generate requirements.txt from pyproject.toml
	@echo "Generating requirements.txt..."
	@if command -v uv >/dev/null 2>&1; then \
		uv export --format requirements-txt --output-file requirements.txt; \
	else \
		echo "⚠️  uv not found, cannot generate requirements.txt"; \
	fi

build-docker: ## Build Docker images
	@echo "Building Docker images..."
	@cd ops && docker compose build

pull-images: ## Pull latest Docker images
	@echo "Pulling latest images..."
	@cd ops && docker compose pull

backup-data: ## Backup Qdrant data
	@echo "Backing up Qdrant data..."
	@docker run --rm -v rag-msmarco-vllm_qdrant_data:/data -v $$(pwd):/backup alpine \
		tar czf /backup/qdrant_backup_$$(date +%Y%m%d_%H%M%S).tar.gz -C /data .

restore-data: ## Restore Qdrant data from backup (set BACKUP_FILE=filename)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "❌ Please specify BACKUP_FILE=filename.tar.gz"; \
		exit 1; \
	fi
	@echo "Restoring Qdrant data from $(BACKUP_FILE)..."
	@docker run --rm -v rag-msmarco-vllm_qdrant_data:/data -v $$(pwd):/backup alpine \
		tar xzf /backup/$(BACKUP_FILE) -C /data

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs: ## Open documentation in browser
	@echo "Opening documentation..."
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:8080/docs; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:8080/docs; \
	else \
		echo "📖 Open http://localhost:8080/docs in your browser"; \
	fi

notebook: ## Start Jupyter for running the cookbook notebook
	@echo "Starting Jupyter..."
	@if command -v jupyter >/dev/null 2>&1; then \
		jupyter notebook cookbook/; \
	else \
		echo "❌ Jupyter not installed. Install with: pip install jupyter"; \
	fi

# ============================================================================
# CI/CD TARGETS
# ============================================================================

ci-test: check-env build-index-small eval-quick ## Run CI tests
	@echo "Running CI test suite..."

ci-lint: lint typecheck ## Run CI linting

ci-build: build-docker ## Build for CI

# ============================================================================
# EXAMPLES AND DEMOS
# ============================================================================

demo-query: check-env ## Run a demo query
	@echo "Running demo query..."
	@curl -X POST http://localhost:8080/v1/query \
		-H "Authorization: Bearer $$(grep AUTH_BEARER_TOKEN .env | cut -d'=' -f2)" \
		-H "Content-Type: application/json" \
		-d '{"query": "What is machine learning?", "stream": false}' | python -m json.tool

demo-ingest: check-env ## Run demo text ingestion
	@echo "Running demo ingestion..."
	@curl -X POST http://localhost:8080/v1/ingest \
		-H "Authorization: Bearer $$(grep AUTH_BEARER_TOKEN .env | cut -d'=' -f2)" \
		-H "Content-Type: application/json" \
		-d '{"source_type": "text", "payload": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."}' | python -m json.tool

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

debug: ## Show debug information
	@echo "=== Debug Information ==="
	@echo "Environment file:"
	@ls -la .env 2>/dev/null || echo ".env not found"
	@echo ""
	@echo "Docker services:"
	@cd ops && docker compose ps
	@echo ""
	@echo "Docker networks:"
	@docker network ls | grep rag
	@echo ""
	@echo "Docker volumes:"
	@docker volume ls | grep rag
	@echo ""
	@echo "Port usage:"
	@netstat -ln | grep -E "(6333|8080|8000|9090|3000)" || true

clean-docker: ## Clean all Docker resources
	@echo "Cleaning Docker resources..."
	@cd ops && docker compose down -v --remove-orphans
	@docker system prune -af
	@docker volume prune -f