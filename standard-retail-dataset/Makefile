.PHONE: help
help:
	@echo "🛠  make: The okay-est command line tool 🛠"
	@echo
	@echo "usage: make <command> [CMD_ARGS='command args']"
	@echo
	@echo "<command> is one of:"
	@echo
	@printf "%24s: %s\n" "bash" "Opens bash inside Docker (accepts CMD_ARGS)"
	@printf "%24s: %s\n" "pip-compile-basepython" "Compile all *.in requirements files in base-python cookiecutter (accepts CMD_ARGS)"
	@printf "%24s: %s\n" "pip-compile-drf" "Compile all *.in requirements files in DRF cookiecutter (accepts CMD_ARGS)"
	@printf "%24s: %s\n" "pip-compile-flask" "Compile all *.in requirements files in flask cookiecutter (accepts CMD_ARGS)"
	@printf "%24s: %s\n" "test" "Runs the test suite (accepts CMD_ARGS)"
	@echo
	@echo "Examples:"
	@echo "        make test"
	@echo "        make pip-compile CMD_ARGS='--upgrade-package=django'"
	@echo

.PHONY: test
test:
	@docker-compose run --rm app ./run-tests ${CMD_ARGS}

.PHONY: clean
clean:
	@find . -name "*.pyc" -exec rm -f {} \;

.PHONY: pip-compile-basepython
pip-compile-basepython:
	@docker-compose run --rm -w /app/cookiecutters/base-python/\{\{cookiecutter.project_name\}\} app pip-compile --allow-unsafe --generate-hashes --output-file=./requirements/base.txt ./requirements/base.in ${CMD_ARGS}
	@docker-compose run --rm -w /app/cookiecutters/base-python/\{\{cookiecutter.project_name\}\} app pip-compile --allow-unsafe --generate-hashes --output-file=./requirements/local.txt ./requirements/local.in ${CMD_ARGS}
	@docker-compose run --rm -w /app/cookiecutters/base-python/\{\{cookiecutter.project_name\}\} app pip-compile --allow-unsafe --generate-hashes --output-file=./requirements/cloud.txt ./requirements/cloud.in ${CMD_ARGS}

.PHONY: pip-compile-drf
pip-compile-drf:
	@docker-compose run --rm -w /app/cookiecutters/djangorestframework/\{\{cookiecutter.project_name\}\} app pip-compile --allow-unsafe --generate-hashes --output-file=./requirements/base.txt ./requirements/base.in ${CMD_ARGS}
	@docker-compose run --rm -w /app/cookiecutters/djangorestframework/\{\{cookiecutter.project_name\}\} app pip-compile --allow-unsafe --generate-hashes --output-file=./requirements/local.txt ./requirements/local.in ${CMD_ARGS}
	@docker-compose run --rm -w /app/cookiecutters/djangorestframework/\{\{cookiecutter.project_name\}\} app pip-compile --allow-unsafe --generate-hashes --output-file=./requirements/cloud.txt ./requirements/cloud.in ${CMD_ARGS}

.PHONY: pip-compile-flask
pip-compile-flask:
	@docker-compose run --rm -w /app/cookiecutters/flask/\{\{cookiecutter.project_name\}\} app pip-compile --allow-unsafe --generate-hashes --output-file=./requirements/base.txt ./requirements/base.in ${CMD_ARGS}
	@docker-compose run --rm -w /app/cookiecutters/flask/\{\{cookiecutter.project_name\}\} app pip-compile --allow-unsafe --generate-hashes --output-file=./requirements/local.txt ./requirements/local.in ${CMD_ARGS}
	@docker-compose run --rm -w /app/cookiecutters/flask/\{\{cookiecutter.project_name\}\} app pip-compile --allow-unsafe --generate-hashes --output-file=./requirements/cloud.txt ./requirements/cloud.in ${CMD_ARGS}

.PHONY: bash
bash:
	@docker-compose run --rm app bash ${CMD_ARGS}
