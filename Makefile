.PHONY: check test test-all

check:
	uv run ruff check src tests tools
	uv run ruff format --check src tests tools
	uv run mypy src/unisupermimic
	uv run pyright src/unisupermimic

test:
	uv run pytest

test-all:
	uv run pytest -m ''

# ROCm hosts: swap in the ROCm dependency profile and install its lockfile.
# This leaves pyproject.toml/uv.lock modified; `make sync-default` restores
# the default profile.
.PHONY: sync-rocm
sync-rocm:
	cp pyproject.rocm.toml pyproject.toml
	cp uv.rocm.lock uv.lock
	uv sync --locked

.PHONY: sync-default
sync-default:
	git checkout pyproject.toml uv.lock
	uv sync --locked

# Maintainer path: regenerate uv.rocm.lock after dependency changes. The
# refresh is scoped to torch so the ROCm lock stays identical to the default
# lock outside the torch/triton-rocm fork. Keep pyproject.rocm.toml in sync
# with pyproject.toml in the same change.
.PHONY: lock-rocm
lock-rocm:
	cp pyproject.rocm.toml pyproject.toml
	uv lock --refresh-package torch
	cp uv.lock uv.rocm.lock
	git checkout pyproject.toml uv.lock
