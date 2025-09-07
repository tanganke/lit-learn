SHELL = /bin/sh

SPHINXBUILD = sphinx-build
SPHINXOPTS = -q
SPHINXBUILDOPTS = -b html
SPHINXBUILDDIR = build/html

.PHONY: help clean html

help:
	@echo "Makefile for building the documentation"
	@echo "Usage:"
	@echo "  make html      Build HTML documentation"
	@echo "  make clean     Clean up build files"

clean:
	rm -rf $(SPHINXBUILDDIR)/*

html:
	$(SPHINXBUILD) $(SPHINXOPTS) $(SPHINXBUILDOPTS) source $(SPHINXBUILDDIR)