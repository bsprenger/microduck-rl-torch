SHELL := /bin/bash

PYTHON ?= PYTHONPATH=src uv run python
PYTEST ?= PYTHONPATH=src uv run pytest
SCRIPT_LAUNCHER ?= scripts/run_with_mjpython.py
POLICY ?= alpha_walking
POLICY_DIR ?= artifacts/hf
STEPS ?= 8
SOLVER_ITERATIONS ?= 4
LINE_SEARCH_ITERATIONS ?= 4
CONTACTS ?= disabled

CONTACT_ARGS = $(if $(filter enabled,$(CONTACTS)),,--disable-contacts)

.PHONY: help install install-upstream install-benchmark reinstall-mujoco-torch lock lock-check format lint typecheck deptry check test coverage \
	fetch-golden-policy fetch-all-policies validate-policy validate-env verify-quick \
	warp-parity validate-warp-parity \
	benchmark-physics \
	render-golden render-golden-torch render-golden-native render-golden-ray convert-gif render-golden-gif render-golden-10s \
	generate-golden-trajectory build clean

help:
	@echo "MicroDuck RL Torch"
	@echo "  make install              Sync the uv environment, dev, and training groups"
	@echo "  make install-upstream     Add the locked upstream Warp/mjlab reference stack"
	@echo "  make install-benchmark    Add plotting dependencies for physics benchmarks"
	@echo "  make reinstall-mujoco-torch Rebuild the sibling local simulator wheel"
	@echo "  make check                Run formatting, lint, type, and dependency checks"
	@echo "  make test                 Run the test suite"
	@echo "  make fetch-golden-policy Download and verify the official HF ONNX policy"
	@echo "  make fetch-all-policies Download every ONNX policy declared by the HF manifest"
	@echo "  make validate-env         Run native-vs-mujoco-torch env validation"
	@echo "  make warp-parity          Compare 500 Torch steps against upstream MuJoCo-Warp"
	@echo "  make benchmark-physics    Benchmark single-environment physics throughput"
	@echo "  make verify-quick         Fetch the policy and validate the environment"
	@echo "  make render-golden        Render the HF policy in the Torch env to MP4"
	@echo "  make render-golden-gif    Render the HF policy to MP4 and GIF"
	@echo "  make render-golden-torch  Render the Torch env with full CAD to MP4 and GIF"
	@echo "  make render-golden-ray    Use the pure mujoco-torch ray renderer"
	@echo "  make render-golden-10s    Render a 10-second golden-policy MP4 and GIF"
	@echo "  make generate-golden-trajectory Regenerate the native BAM parity fixture"

install:
	uv sync --group dev --group training

install-upstream:
	uv sync --group upstream

install-benchmark:
	uv sync --group benchmark

reinstall-mujoco-torch:
	uv sync --reinstall-package mujoco-torch --group dev --group training

lock:
	uv lock

format:
	uv run ruff format src tests scripts

lint:
	uv run ruff check src tests scripts

typecheck:
	uv run ty check src tests scripts

deptry:
	uv run deptry . -kf microduck_rl_torch -kf microduck_rl_torch_verification \
		--non-dev-dependency-groups benchmark,upstream \
		--per-rule-ignores 'DEP002=mjlab|better-actuator-models|rustypot|scipy,DEP004=warp|matplotlib'

lock-check:
	uv lock --locked

check: lock-check lint typecheck deptry

test:
	$(PYTEST)

coverage:
	$(PYTEST) --cov=microduck_rl_torch --cov-report=term-missing --cov-report=html

fetch-golden-policy:
	$(PYTHON) scripts/fetch_hf_policy.py --policy $(POLICY) --output-dir $(POLICY_DIR)

fetch-all-policies:
	$(PYTHON) scripts/fetch_hf_policy.py --all --output-dir $(POLICY_DIR)

validate-policy: fetch-golden-policy
	$(PYTHON) scripts/validate_policy.py --policy-dir $(POLICY_DIR) --policy $(POLICY)

validate-env: fetch-golden-policy
	$(PYTHON) scripts/validate_environment.py --policy-dir $(POLICY_DIR) --policy $(POLICY) --steps $(STEPS) --fixed-iterations --solver-iterations $(SOLVER_ITERATIONS) --line-search-iterations $(LINE_SEARCH_ITERATIONS) $(CONTACT_ARGS)

verify-quick: validate-policy validate-env

generate-golden-trajectory:
	@$(PYTHON) scripts/generate_golden_trajectory.py

WARP_PARITY_STEPS ?= 500
WARP_PARITY_OUTPUT ?= artifacts/parity/microduck-warp-500.md
UPSTREAM_ROOT ?=
UPSTREAM_PYTHON ?= .venv/bin/python
UPSTREAM_DEVICE ?= cpu
WARP_PARITY_FAIL_ON ?= state

.PHONY: warp-parity validate-warp-parity
warp-parity: fetch-golden-policy ## Compare the local Torch rollout with upstream Warp
	@test -n "$(UPSTREAM_ROOT)" || (echo "Set UPSTREAM_ROOT to the upstream microduck_rl checkout"; exit 2)
	@test -x "$(UPSTREAM_PYTHON)" || (echo "Missing $(UPSTREAM_PYTHON); run make install-upstream first"; exit 2)
	@$(PYTHON) scripts/validate_warp_parity.py \
		--policy "$(POLICY)" --policy-dir "$(POLICY_DIR)" \
		--upstream-root "$(UPSTREAM_ROOT)" --upstream-python "$(UPSTREAM_PYTHON)" \
		--upstream-device "$(UPSTREAM_DEVICE)" --steps "$(WARP_PARITY_STEPS)" \
		--output "$(WARP_PARITY_OUTPUT)" --fail-on "$(WARP_PARITY_FAIL_ON)"

validate-warp-parity: warp-parity

BENCHMARK_STEPS ?= 50
BENCHMARK_WARMUP_STEPS ?= 10
BENCHMARK_REPEATS ?= 3
BENCHMARK_DEVICES ?= cpu,mps
BENCHMARK_BACKEND ?= both
BENCHMARK_OUTPUT ?= artifacts/benchmarks/physics-single-env
BENCHMARK_README_GRAPH ?= docs/assets/microduck-physics-throughput.png
BENCHMARK_UPSTREAM_ROOT ?= ../microduck_rl
BENCHMARK_SOLVER_ITERATIONS ?= 4
BENCHMARK_LINE_SEARCH_ITERATIONS ?= 4
BENCHMARK_MESH_MESH_CONTACTS ?= disabled

.PHONY: benchmark-physics
benchmark-physics: ## Benchmark direct physics stepping without policy inference
	@PYTHONPATH=src uv run --group benchmark python scripts/benchmark_physics.py \
		--backend "$(BENCHMARK_BACKEND)" --devices "$(BENCHMARK_DEVICES)" \
		--steps "$(BENCHMARK_STEPS)" --warmup-steps "$(BENCHMARK_WARMUP_STEPS)" \
		--repeats "$(BENCHMARK_REPEATS)" \
		--solver-iterations "$(BENCHMARK_SOLVER_ITERATIONS)" \
		--line-search-iterations "$(BENCHMARK_LINE_SEARCH_ITERATIONS)" \
		--mesh-mesh-contacts "$(BENCHMARK_MESH_MESH_CONTACTS)" \
		--upstream-root "$(BENCHMARK_UPSTREAM_ROOT)" --output "$(BENCHMARK_OUTPUT)" \
		--readme-graph "$(BENCHMARK_README_GRAPH)"

RENDER_OUTPUT ?= artifacts/render/microduck-alpha-walking.mp4
RENDER_GIF ?= artifacts/render/microduck-alpha-walking.gif
RENDER_STEPS ?= 250
RENDER_SECONDS ?=
RENDER_FPS ?= 25
RENDER_EVERY ?= 2
RENDER_WIDTH ?= 320
RENDER_HEIGHT ?= 240
RENDER_BACKEND ?= mujoco
RENDER_CAMERA ?= free
RENDER_DEVICE ?= cpu
RENDER_ACTUATOR_MODE ?= xml
RENDER_VX ?= 0.3
RENDER_VY ?= 0.0
RENDER_VTHETA ?= 0.0
RENDER_GIF_FPS ?= $(RENDER_FPS)
RENDER_GIF_WIDTH ?= 720
RENDER_GIF_COLORS ?= 48
RENDER_MESH_MESH_CONTACTS ?= disabled
RENDER_SOLVER_ITERATIONS ?= 4
RENDER_LINE_SEARCH_ITERATIONS ?= 4
RENDER_RAY_CHUNK_SIZE ?= 256

RENDER_DURATION_ARGS = $(if $(strip $(RENDER_SECONDS)),--seconds "$(RENDER_SECONDS)",--steps "$(RENDER_STEPS)")

RENDER_ARGS = \
	--output "$(RENDER_OUTPUT)" \
	$(RENDER_DURATION_ARGS) \
	--fps "$(RENDER_FPS)" \
	--render-every "$(RENDER_EVERY)" \
	--width "$(RENDER_WIDTH)" \
	--height "$(RENDER_HEIGHT)" \
	--device "$(RENDER_DEVICE)" \
	--actuator-mode "$(RENDER_ACTUATOR_MODE)" \
	--render-backend "$(RENDER_BACKEND)" \
	--camera "$(RENDER_CAMERA)" \
	--vx "$(RENDER_VX)" --vy "$(RENDER_VY)" --vtheta "$(RENDER_VTHETA)" \
	--solver-iterations "$(RENDER_SOLVER_ITERATIONS)" \
	--line-search-iterations "$(RENDER_LINE_SEARCH_ITERATIONS)" \
	--fixed-iterations \
	--mesh-mesh-contacts "$(RENDER_MESH_MESH_CONTACTS)"

.PHONY: render-golden
render-golden: fetch-golden-policy ## Render the HF golden policy in the Torch env to MP4
	@PYTHONPATH=src uv run python $(SCRIPT_LAUNCHER) --module microduck_rl_torch.rendering.cli -- \
		$(RENDER_ARGS)

.PHONY: render-golden-native
render-golden-native: fetch-golden-policy ## Render through the native MuJoCo OpenGL context
	@PYTHONPATH=src uv run python $(SCRIPT_LAUNCHER) --module microduck_rl_torch.rendering.cli -- \
		$(RENDER_ARGS) --render-backend mujoco

.PHONY: render-golden-torch
render-golden-torch: fetch-golden-policy ## Render the Torch env with full CAD to MP4 and GIF
	@PYTHONPATH=src uv run python $(SCRIPT_LAUNCHER) --module microduck_rl_torch.rendering.cli -- \
		$(RENDER_ARGS) --render-backend mujoco --camera free \
		--output "$(RENDER_OUTPUT:.mp4=-torch.mp4)" \
		--gif "$(RENDER_GIF:.gif=-torch.gif)" --gif-fps "$(RENDER_GIF_FPS)" \
		--gif-width "$(RENDER_GIF_WIDTH)" --gif-colors "$(RENDER_GIF_COLORS)"

.PHONY: render-golden-ray
render-golden-ray: fetch-golden-policy ## Render with the pure mujoco-torch ray renderer
	@PYTHONPATH=src uv run microduck-render \
		--policy "$(POLICY)" --policy-dir "$(POLICY_DIR)" \
		--output "$(RENDER_OUTPUT:.mp4=-ray.mp4)" \
		$(RENDER_DURATION_ARGS) --fps "$(RENDER_FPS)" \
		--render-every "$(RENDER_EVERY)" \
		--width "$(RENDER_WIDTH)" --height "$(RENDER_HEIGHT)" \
		--device "$(RENDER_DEVICE)" \
		--actuator-mode "$(RENDER_ACTUATOR_MODE)" \
		--render-backend mujoco-torch --camera head_camera \
		--vx "$(RENDER_VX)" --vy "$(RENDER_VY)" --vtheta "$(RENDER_VTHETA)" \
		--solver-iterations "$(RENDER_SOLVER_ITERATIONS)" \
		--line-search-iterations "$(RENDER_LINE_SEARCH_ITERATIONS)" \
		--fixed-iterations \
		--mesh-mesh-contacts "$(RENDER_MESH_MESH_CONTACTS)" \
		--ray-chunk-size "$(RENDER_RAY_CHUNK_SIZE)" \
		--gif "$(RENDER_GIF:.gif=-ray.gif)" \
		--gif-fps "$(RENDER_GIF_FPS)" --gif-width "$(RENDER_GIF_WIDTH)" \
		--gif-colors "$(RENDER_GIF_COLORS)"

.PHONY: convert-gif
convert-gif: ## Convert RENDER_OUTPUT into a looping palette-optimized GIF
	@PYTHONPATH=src uv run microduck-convert-gif \
		--input "$(RENDER_OUTPUT)" --output "$(RENDER_GIF)" \
		--fps "$(RENDER_GIF_FPS)" --width "$(RENDER_GIF_WIDTH)" --colors "$(RENDER_GIF_COLORS)"

.PHONY: render-golden-gif
render-golden-gif: fetch-golden-policy ## Render the HF golden policy to MP4 and GIF
	@PYTHONPATH=src uv run python $(SCRIPT_LAUNCHER) --module microduck_rl_torch.rendering.cli -- \
		$(RENDER_ARGS) \
		--gif "$(RENDER_GIF)" --gif-fps "$(RENDER_GIF_FPS)" \
		--gif-width "$(RENDER_GIF_WIDTH)" --gif-colors "$(RENDER_GIF_COLORS)"

.PHONY: render-golden-10s
render-golden-10s: RENDER_SECONDS=10
render-golden-10s: render-golden-gif ## Render a 10-second HF golden-policy MP4 and GIF

build:
	uv build

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf dist build htmlcov
