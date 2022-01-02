flake:
	flake8 ./coconutools ./tests

isort:
	isort ./coconutools ./tests

black:
	black ./coconutools ./tests

mypy:
	mypy ./coconutools ./tests

lint:
	make isort && make black && make flake  && make mypy

test:
	pytest -vvv ./tests
