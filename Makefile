
SHELL := /bin/bash

PRE_RUN="source .bazel_setup.sh"

TOPDIR=$(shell pwd)
BAZEL=bazel/bazel.sh

all: print_targets

.PHONY: print_targets perf perf-rocm perf-cuda debug debug-rocm debug-cuda \
	develop develop-rocm develop-cuda wheel wheel-rocm wheel-cuda \
	test test-rocm test-cuda docs clean distclean expunge hm-ui hmtrack-rust-ui

define run_bazel_with_backend
	@set -euo pipefail; \
	if [ -n "$(1)" ]; then \
		export HM_FORCE_TORCH_BACKEND="$(1)"; \
	else \
		unset HM_FORCE_TORCH_BACKEND; \
	fi; \
	$(BAZEL) $(2)
endef

define run_develop_with_backend
	@set -euo pipefail; \
	if [ -n "$(1)" ]; then \
		export HM_FORCE_TORCH_BACKEND="$(1)"; \
	else \
		unset HM_FORCE_TORCH_BACKEND; \
	fi; \
	source ./.bazel_setup.sh; \
	if [ "$${TORCH_BACKEND}" = "rocm" ] || [ "$${TORCH_BACKEND}" = "cuda" ] || command -v nvcc >/dev/null 2>&1 || [ -f /usr/local/cuda/include/cuda_runtime.h ]; then \
		$(BAZEL) run --config=release //hockeymon:link_ext; \
	else \
		echo "Skipping hockeymon native extension link: neither ROCm nor CUDA toolkit/backend detected"; \
	fi; \
	$(BAZEL) run --config=release //hmlib:develop -- --workspace=$(TOPDIR)
endef

hm-ui:
	$(BAZEL) build --config=release //hm-ui:hm-ui

perf:
	$(call run_bazel_with_backend,,build --config=release //...)

perf-rocm:
	$(call run_bazel_with_backend,rocm,build --config=release //...)

perf-cuda:
	$(call run_bazel_with_backend,cuda,build --config=release //...)

debug:
	$(call run_bazel_with_backend,,build --config=debug //...)

debug-rocm:
	$(call run_bazel_with_backend,rocm,build --config=debug //...)

debug-cuda:
	$(call run_bazel_with_backend,cuda,build --config=debug //...)

test:
	$(call run_bazel_with_backend,,test --config=release //...)

test-rocm:
	$(call run_bazel_with_backend,rocm,test --config=release //...)

test-cuda:
	$(call run_bazel_with_backend,cuda,test --config=release //...)

wheel:
	$(call run_bazel_with_backend,,run --config=release //hockeymon:bdist_wheel)
	$(call run_bazel_with_backend,,run --config=release //hmlib:bdist_wheel)

wheel-rocm:
	$(call run_bazel_with_backend,rocm,run --config=release //hockeymon:bdist_wheel)
	$(call run_bazel_with_backend,rocm,run --config=release //hmlib:bdist_wheel)

wheel-cuda:
	$(call run_bazel_with_backend,cuda,run --config=release //hockeymon:bdist_wheel)
	$(call run_bazel_with_backend,cuda,run --config=release //hmlib:bdist_wheel)

docs:
	$(BAZEL) build //:all_doxygen_docs

clean:
	$(BAZEL) clean

distclean expunge:
	$(BAZEL) clean --expunge

develop: hm-ui
	$(call run_develop_with_backend,)

develop-rocm: hm-ui
	$(call run_develop_with_backend,rocm)

develop-cuda: hm-ui
	$(call run_develop_with_backend,cuda)

hmtrack-rust-ui: hm-ui
	HM_UI_BIN=$(TOPDIR)/bazel-bin/hm-ui/hm-ui-bin PYTHONPATH=$(TOPDIR) python hmlib/cli/hmtrack.py --camera-ui=1 $(ARGS)

deps:
	cd external/hugin && $(TOPDIR)/$(BAZEL) run --config=release //:install_tree -- --prefix=$(CONDA_PREFIX)
	cd -
	touch .hugin_built
print_targets:
	@printf '%s\n' \
		"Available make targets (run 'make <target>'):" \
		'' \
		'Build Outputs' \
		'-------------' \
		'hm-ui        Build the Rust hm-ui sidecar binary used by --camera-ui=1.' \
		'perf         Build every Bazel target with --config=release using the auto-detected torch backend.' \
		'perf-rocm    Build every Bazel target with --config=release while forcing the ROCm torch backend.' \
		'perf-cuda    Build every Bazel target with --config=release while forcing the CUDA torch backend.' \
		'debug        Build every Bazel target with --config=debug using the auto-detected torch backend.' \
		'debug-rocm   Build every Bazel target with --config=debug while forcing the ROCm torch backend.' \
		'debug-cuda   Build every Bazel target with --config=debug while forcing the CUDA torch backend.' \
		'' \
		'Documentation' \
		'--------------' \
		'docs         Builds both hockeymon and hmlib Doxygen archives via //:all_doxygen_docs; run when you need refreshed API docs.' \
		'' \
		'Developer Workflow' \
		'------------------' \
		'develop      Builds hm-ui, refreshes hockeymon extension symlinks when the detected backend is ROCm or CUDA, then installs hmlib for development.' \
		'develop-rocm Same as develop, but forces the ROCm torch backend.' \
		'develop-cuda Same as develop, but forces the CUDA torch backend.' \
		'hmtrack-rust-ui  Build hm-ui, then run hmtrack with the Rust camera UI. Pass hmtrack args with ARGS="--game-id chicago-3 ...".' \
		'test         Runs the release-configured Bazel test suite using the auto-detected torch backend.' \
		'test-rocm    Runs the release-configured Bazel test suite while forcing the ROCm torch backend.' \
		'test-cuda    Runs the release-configured Bazel test suite while forcing the CUDA torch backend.' \
		'wheel        Builds release wheels for hockeymon and hmlib using the auto-detected torch backend.' \
		'wheel-rocm   Builds release wheels for hockeymon and hmlib while forcing the ROCm torch backend.' \
		'wheel-cuda   Builds release wheels for hockeymon and hmlib while forcing the CUDA torch backend.' \
		'' \
		'Maintenance & Cleanup' \
		'---------------------' \
		'clean        bazel clean to drop cached outputs when builds behave strangely or you switch branches.' \
		'distclean    bazel clean --expunge (also aliased as expunge); run for a fully fresh Bazel state if clean is insufficient.' \
		'expunge      Same as distclean; provided for convenience.' \
		'' \
		'Dependencies' \
		'------------' \
		'deps         Installs the external hugin tree into your active conda prefix and marks it as built; run after installing a new environment or when hugin headers go missing.' \
		'' \
		'Meta' \
		'----' \
		'Explicit backend targets require a matching PyTorch environment; the helper will fail fast if the forced backend disagrees with the detected torch build.' \
		'' \
		'print_targets  Shows this help text.'
