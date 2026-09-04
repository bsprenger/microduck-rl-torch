SHELL := /bin/bash

PYTHON ?= uv run python
PYTEST ?= uv run pytest
SCRIPT_LAUNCHER ?= scripts/run_with_mjpython.py
POLICY ?= alpha_walking
POLICY_DIR ?= artifacts/hf
STEPS ?= 8
SOLVER_ITERATIONS ?= 4
LINE_SEARCH_ITERATIONS ?= 4
CONTACTS ?= disabled

CONTACT_ARGS = $(if $(filter enabled,$(CONTACTS)),,--disable-contacts)

.PHONY: help install reinstall-mujoco-torch lock lock-check format lint typecheck deptry check test coverage \
	fetch-golden-policy validate-policy validate-env verify-quick \
	render-golden render-golden-torch render-golden-native render-golden-ray convert-gif render-golden-gif \
	build clean

help:
	@echo "MicroDuck RL Torch"
	@echo "  make install              Sync the uv environment and dev tools"
	@echo "  make reinstall-mujoco-torch Rebuild the sibling local simulator wheel"
	@echo "  make check                Run formatting, lint, type, and dependency checks"
	@echo "  make test                 Run the test suite"
	@echo "  make fetch-golden-policy Download and verify the official HF ONNX policy"
	@echo "  make validate-env         Run native-vs-mujoco-torch env validation"
	@echo "  make verify-quick         Fetch the policy and validate the environment"
	@echo "  make render-golden        Render the HF policy in the Torch env to MP4"
	@echo "  make render-golden-gif    Render the HF policy to MP4 and GIF"
	@echo "  make render-golden-torch  Render the Torch env with full CAD to MP4 and GIF"
	@echo "  make render-golden-ray    Use the pure mujoco-torch ray renderer"

install:
	uv sync --all-groups

reinstall-mujoco-torch:
	uv sync --reinstall-package mujoco-torch --all-groups

lock:
	uv lock

format:
	uv run ruff format src tests scripts

lint:
	uv run ruff check src tests scripts

typecheck:
	uv run ty check src tests scripts

deptry:
	uv run deptry . -kf microduck_rl_torch -kf microduck_rl_torch_verification

lock-check:
	uv lock --locked

check: lock-check lint typecheck deptry

test:
	$(PYTEST)

coverage:
	$(PYTEST) --cov=microduck_rl_torch --cov-report=term-missing --cov-report=html

fetch-golden-policy:
	$(PYTHON) scripts/fetch_hf_policy.py --policy $(POLICY) --output-dir $(POLICY_DIR)

validate-policy: fetch-golden-policy
	$(PYTHON) scripts/validate_policy.py --policy-dir $(POLICY_DIR) --policy $(POLICY)

validate-env: fetch-golden-policy
	$(PYTHON) scripts/validate_environment.py --policy-dir $(POLICY_DIR) --policy $(POLICY) --steps $(STEPS) --fixed-iterations --solver-iterations $(SOLVER_ITERATIONS) --line-search-iterations $(LINE_SEARCH_ITERATIONS) $(CONTACT_ARGS)

verify-quick: validate-policy validate-env

RENDER_OUTPUT ?= artifacts/render/microduck-alpha-walking.mp4
RENDER_GIF ?= artifacts/render/microduck-alpha-walking.gif
RENDER_STEPS ?= 250
RENDER_FPS ?= 25
RENDER_EVERY ?= 2
RENDER_WIDTH ?= 320
RENDER_HEIGHT ?= 240
RENDER_BACKEND ?= mujoco
RENDER_CAMERA ?= free
RENDER_DEVICE ?= cpu
RENDER_VX ?= 0.15
RENDER_VY ?= 0.0
RENDER_VTHETA ?= 0.0
RENDER_GIF_FPS ?= 12
RENDER_GIF_WIDTH ?= 720
RENDER_GIF_COLORS ?= 48
RENDER_CONTACTS ?= enabled
RENDER_SOLVER_ITERATIONS ?= 4
RENDER_LINE_SEARCH_ITERATIONS ?= 4
RENDER_RAY_CHUNK_SIZE ?= 256

RENDER_ARGS = \
	--output "$(RENDER_OUTPUT)" \
	--steps "$(RENDER_STEPS)" \
	--fps "$(RENDER_FPS)" \
	--render-every "$(RENDER_EVERY)" \
	--width "$(RENDER_WIDTH)" \
	--height "$(RENDER_HEIGHT)" \
	--device "$(RENDER_DEVICE)" \
	--render-backend "$(RENDER_BACKEND)" \
	--camera "$(RENDER_CAMERA)" \
	--vx "$(RENDER_VX)" --vy "$(RENDER_VY)" --vtheta "$(RENDER_VTHETA)" \
	--solver-iterations "$(RENDER_SOLVER_ITERATIONS)" \
	--line-search-iterations "$(RENDER_LINE_SEARCH_ITERATIONS)" \
	--fixed-iterations \
	--contacts "$(RENDER_CONTACTS)"

.PHONY: render-golden
render-golden: fetch-golden-policy ## Render the HF golden policy in the Torch env to MP4
	@uv run python $(SCRIPT_LAUNCHER) --module microduck_rl_torch.rendering.cli -- \
		$(RENDER_ARGS)

.PHONY: render-golden-native
render-golden-native: fetch-golden-policy ## Render through the native MuJoCo OpenGL context
	@uv run python $(SCRIPT_LAUNCHER) --module microduck_rl_torch.rendering.cli -- \
		$(RENDER_ARGS) --render-backend mujoco

.PHONY: render-golden-torch
render-golden-torch: fetch-golden-policy ## Render the Torch env with full CAD to MP4 and GIF
	@uv run python $(SCRIPT_LAUNCHER) --module microduck_rl_torch.rendering.cli -- \
		$(RENDER_ARGS) --render-backend mujoco --camera free \
		--output "$(RENDER_OUTPUT:.mp4=-torch.mp4)" \
		--gif "$(RENDER_GIF:.gif=-torch.gif)" --gif-fps "$(RENDER_GIF_FPS)" \
		--gif-width "$(RENDER_GIF_WIDTH)" --gif-colors "$(RENDER_GIF_COLORS)"

.PHONY: render-golden-ray
render-golden-ray: fetch-golden-policy ## Render with the pure mujoco-torch ray renderer
	@uv run microduck-render \
		--policy "$(POLICY)" --policy-dir "$(POLICY_DIR)" \
		--output "$(RENDER_OUTPUT:.mp4=-ray.mp4)" \
		--steps "$(RENDER_STEPS)" --fps "$(RENDER_FPS)" \
		--render-every "$(RENDER_EVERY)" \
		--width "$(RENDER_WIDTH)" --height "$(RENDER_HEIGHT)" \
		--device "$(RENDER_DEVICE)" \
		--render-backend mujoco-torch --camera head_camera \
		--vx "$(RENDER_VX)" --vy "$(RENDER_VY)" --vtheta "$(RENDER_VTHETA)" \
		--solver-iterations "$(RENDER_SOLVER_ITERATIONS)" \
		--line-search-iterations "$(RENDER_LINE_SEARCH_ITERATIONS)" \
		--fixed-iterations --contacts "$(RENDER_CONTACTS)" \
		--ray-chunk-size "$(RENDER_RAY_CHUNK_SIZE)" \
		--gif "$(RENDER_GIF:.gif=-ray.gif)" \
		--gif-fps "$(RENDER_GIF_FPS)" --gif-width "$(RENDER_GIF_WIDTH)" \
		--gif-colors "$(RENDER_GIF_COLORS)"

.PHONY: convert-gif
convert-gif: ## Convert RENDER_OUTPUT into a looping palette-optimized GIF
	@uv run microduck-convert-gif \
		--input "$(RENDER_OUTPUT)" --output "$(RENDER_GIF)" \
		--fps "$(RENDER_GIF_FPS)" --width "$(RENDER_GIF_WIDTH)" --colors "$(RENDER_GIF_COLORS)"

.PHONY: render-golden-gif
render-golden-gif: fetch-golden-policy ## Render the HF golden policy to MP4 and GIF
	@uv run python $(SCRIPT_LAUNCHER) --module microduck_rl_torch.rendering.cli -- \
		$(RENDER_ARGS) \
		--gif "$(RENDER_GIF)" --gif-fps "$(RENDER_GIF_FPS)" \
		--gif-width "$(RENDER_GIF_WIDTH)" --gif-colors "$(RENDER_GIF_COLORS)"

build:
	uv build

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf dist build htmlcov
