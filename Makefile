.PHONY: setup download export infer validate bench bench-inspiron run run-headless pack pack-md clean

VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python)
PIP ?= $(if $(wildcard $(VENV)/bin/pip),$(VENV)/bin/pip,pip)

setup:
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements-cpu.txt

download:
	$(PYTHON) scripts/download_pretrained.py

export:
	$(PYTHON) scripts/export.py

infer:
	$(PYTHON) scripts/infer.py --profile inspiron3521_cpu --save validation/bus_detected.jpg

validate:
	$(PYTHON) scripts/validate.py

validate-inspiron:
	$(PYTHON) scripts/validate.py --profile inspiron3521_cpu

bench:
	$(PYTHON) scripts/bench.py --profile dev_pt

bench-inspiron:
	$(PYTHON) scripts/bench.py --profile inspiron3521_cpu --frames 30

run:
	$(PYTHON) scripts/run_cameras.py --profile dev_pt

run-headless:
	$(PYTHON) scripts/run_cameras.py --profile dev_pt --headless

pack:
	npx --yes repomix

pack-md:
	npx --yes repomix --style markdown -o repomix-output.md

clean:
	rm -rf runs/ \
	       exports/ppe.onnx \
	       exports/ppe_ncnn_model \
	       validation/bus.jpg \
	       repomix-output.*
