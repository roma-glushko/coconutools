.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

flake:
	flake8 ./coconutools ./tests

isort:
	isort ./coconutools ./tests

black:
	black ./coconutools ./tests

mypy:
	mypy ./coconutools ./tests

lint:  # Lint the source code
	make isort && make black && make flake  && make mypy

test: # Run tests
	pytest -vvv ./tests
