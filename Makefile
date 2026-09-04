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

.PHONY: help install install-microduck-rl-warp install-benchmark reinstall-mujoco-torch lock lock-check format format-check lint typecheck deptry check archive-check test coverage \
	fetch-golden-policy fetch-all-policies validate-policy validate-env verify-quick \
	warp-parity validate-warp-parity \
	benchmark-physics \
	render-golden render-golden-torch render-golden-native render-golden-ray convert-gif render-golden-gif render-golden-10s render-golden-rough-10s \
	generate-golden-trajectory build clean

help:
	@echo "microduck-rl-torch"
	@echo "  make install              Sync the uv environment, dev, and training groups"
	@echo "  make install-microduck-rl-warp Install optional microduck_rl MuJoCo-Warp dependencies"
	@echo "  make install-benchmark    Install plotting dependencies for physics benchmarks"
	@echo "  make reinstall-mujoco-torch Reinstall the locked mujoco-torch dependency"
	@echo "  make check                Run lock, format, lint, type, and dependency checks"
	@echo "  make test                 Run the test suite"
	@echo "  make fetch-golden-policy Download and verify the official HF ONNX policy"
	@echo "  make fetch-all-policies Download every ONNX policy declared by the HF manifest"
	@echo "  make validate-env         Run native-vs-mujoco-torch env validation"
	@echo "  make warp-parity          Compare 500 Torch steps against microduck_rl MuJoCo-Warp"
	@echo "  make benchmark-physics    Benchmark single-environment physics throughput"
	@echo "  make verify-quick         Fetch the policy and validate the environment"
	@echo "  make render-golden        Render the HF policy in the Torch env to MP4"
	@echo "  make render-golden-gif    Render the HF policy to MP4 and GIF"
	@echo "  make render-golden-torch  Render the Torch env with full CAD to MP4 and GIF"
	@echo "  make render-golden-ray    Run the pure-Torch ray renderer for diagnostics"
	@echo "  make render-golden-10s    Render a 10-second golden-policy MP4 and GIF"
	@echo "  make render-golden-rough-10s Render a 10-second rough-terrain MP4 and GIF"
	@echo "  make generate-golden-trajectory Regenerate the native BAM fixture"

install:
	uv sync --group dev --group training

install-microduck-rl-warp:
	uv sync --group microduck-rl-warp

install-benchmark:
	uv sync --group benchmark

reinstall-mujoco-torch:
	uv sync --reinstall-package mujoco-torch --group dev --group training

lock:
	uv lock

format:
	uv run ruff format src tests scripts

format-check:
	uv run ruff format --check src tests scripts

lint:
	uv run ruff check src tests scripts

typecheck:
	uv run ty check src tests scripts --extra-search-path typings

deptry:
	uv run deptry . -kf microduck_rl_torch -kf microduck_rl_torch_verification \
		--non-dev-dependency-groups benchmark,microduck-rl-warp \
		--package-module-name-map 'warp-lang=warp' \
		--per-rule-ignores 'DEP001=hatchling,DEP002=mjlab|better-actuator-models|rustypot|scipy,DEP004=hatchling|matplotlib'

lock-check:
	uv lock --locked

check: lock-check format-check lint typecheck deptry

archive-check: build
	uv run python scripts/check_distribution.py

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
MICRODUCK_RL_ROOT ?=
MICRODUCK_RL_PYTHON ?= .venv/bin/python
MICRODUCK_RL_DEVICE ?= cpu
WARP_PARITY_FAIL_ON ?= state

.PHONY: warp-parity validate-warp-parity
warp-parity: fetch-golden-policy ## Compare the local Torch rollout with microduck_rl MuJoCo-Warp
	@test -n "$(MICRODUCK_RL_ROOT)" || (echo "Set MICRODUCK_RL_ROOT to the microduck_rl checkout"; exit 2)
	@test -x "$(MICRODUCK_RL_PYTHON)" || (echo "Missing $(MICRODUCK_RL_PYTHON); run make install-microduck-rl-warp first"; exit 2)
	@$(PYTHON) scripts/validate_warp_parity.py \
		--policy "$(POLICY)" --policy-dir "$(POLICY_DIR)" \
		--microduck-rl-root "$(MICRODUCK_RL_ROOT)" --microduck-rl-python "$(MICRODUCK_RL_PYTHON)" \
		--microduck-rl-device "$(MICRODUCK_RL_DEVICE)" --steps "$(WARP_PARITY_STEPS)" \
		--output "$(WARP_PARITY_OUTPUT)" --fail-on "$(WARP_PARITY_FAIL_ON)"

validate-warp-parity: warp-parity

BENCHMARK_STEPS ?= 50
BENCHMARK_WARMUP_STEPS ?= 10
BENCHMARK_REPEATS ?= 3
BENCHMARK_DEVICES ?= cpu,mps
BENCHMARK_BACKEND ?= both
BENCHMARK_OUTPUT ?= artifacts/benchmarks/physics-single-env
BENCHMARK_README_GRAPH ?= docs/assets/microduck-physics-throughput.png
BENCHMARK_MICRODUCK_RL_ROOT ?= ../microduck_rl
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
		--microduck-rl-root "$(BENCHMARK_MICRODUCK_RL_ROOT)" --output "$(BENCHMARK_OUTPUT)" \
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
RENDER_ROUGH ?= false
RENDER_PLAY ?= false
RENDER_TERRAIN_ROWS ?=
RENDER_TERRAIN_COLS ?=
RENDER_GIF_FPS ?= $(RENDER_FPS)
RENDER_GIF_WIDTH ?= 720
RENDER_GIF_COLORS ?= 48
RENDER_MESH_MESH_CONTACTS ?= auto
RENDER_SOLVER_ITERATIONS ?=
RENDER_LINE_SEARCH_ITERATIONS ?=
RENDER_RAY_CHUNK_SIZE ?= 256

RENDER_DURATION_ARGS = $(if $(strip $(RENDER_SECONDS)),--seconds "$(RENDER_SECONDS)",--steps "$(RENDER_STEPS)")
RENDER_ROUGH_ARGS = $(if $(filter true 1 yes,$(RENDER_ROUGH)),--rough,)
RENDER_PLAY_ARGS = $(if $(filter true 1 yes,$(RENDER_PLAY)),--play,)
RENDER_TERRAIN_ROWS_ARGS = $(if $(strip $(RENDER_TERRAIN_ROWS)),--terrain-rows "$(RENDER_TERRAIN_ROWS)",)
RENDER_TERRAIN_COLS_ARGS = $(if $(strip $(RENDER_TERRAIN_COLS)),--terrain-cols "$(RENDER_TERRAIN_COLS)",)
RENDER_SOLVER_ARGS = $(if $(strip $(RENDER_SOLVER_ITERATIONS)),--solver-iterations "$(RENDER_SOLVER_ITERATIONS)",)
RENDER_LINE_SEARCH_ARGS = $(if $(strip $(RENDER_LINE_SEARCH_ITERATIONS)),--line-search-iterations "$(RENDER_LINE_SEARCH_ITERATIONS)",)

RENDER_ARGS = \
	--output "$(RENDER_OUTPUT)" \
	$(RENDER_DURATION_ARGS) \
	--fps "$(RENDER_FPS)" \
	--render-every "$(RENDER_EVERY)" \
	--width "$(RENDER_WIDTH)" \
	--height "$(RENDER_HEIGHT)" \
	--device "$(RENDER_DEVICE)" \
	--actuator-mode "$(RENDER_ACTUATOR_MODE)" \
	$(RENDER_ROUGH_ARGS) \
	$(RENDER_PLAY_ARGS) \
	$(RENDER_TERRAIN_ROWS_ARGS) \
	$(RENDER_TERRAIN_COLS_ARGS) \
	--render-backend "$(RENDER_BACKEND)" \
	--camera "$(RENDER_CAMERA)" \
	--vx "$(RENDER_VX)" --vy "$(RENDER_VY)" --vtheta "$(RENDER_VTHETA)" \
	$(RENDER_SOLVER_ARGS) \
	$(RENDER_LINE_SEARCH_ARGS) \
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
		$(RENDER_SOLVER_ARGS) \
		$(RENDER_LINE_SEARCH_ARGS) \
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

.PHONY: render-golden-rough-10s
render-golden-rough-10s: RENDER_ROUGH=true
render-golden-rough-10s: RENDER_PLAY=true
render-golden-rough-10s: RENDER_TERRAIN_ROWS=1
render-golden-rough-10s: RENDER_TERRAIN_COLS=1
render-golden-rough-10s: RENDER_SECONDS=10
render-golden-rough-10s: RENDER_FPS=2
render-golden-rough-10s: RENDER_EVERY=25
render-golden-rough-10s: RENDER_WIDTH=160
render-golden-rough-10s: RENDER_HEIGHT=120
render-golden-rough-10s: RENDER_ACTUATOR_MODE=bam
render-golden-rough-10s: RENDER_MESH_MESH_CONTACTS=disabled
render-golden-rough-10s: RENDER_SOLVER_ITERATIONS=4
render-golden-rough-10s: RENDER_LINE_SEARCH_ITERATIONS=4
render-golden-rough-10s: RENDER_OUTPUT=artifacts/render/microduck-alpha-walking-rough.mp4
render-golden-rough-10s: RENDER_GIF=artifacts/render/microduck-alpha-walking-rough.gif
render-golden-rough-10s: render-golden-gif ## Render a 10-second rough-terrain MP4 and GIF

build:
	uv build

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf dist build htmlcov
