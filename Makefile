.PHONY: local-setup
local-setup:
	@echo Creating virtual environment
	@poetry env activate
	@$(MAKE) install

install:
	@echo Installing all dev dependencies
	@poetry install --with dev

.PHONY: lint
lint:
	@echo "Linting code"
	@poetry run pre-commit run -a

.PHONY: test
test:
	@echo "Running tests"
	@poetry run pytest -v