# kbforge developer tasks.
#
# This file is intentionally thin: every target hands off to makefile.py, which does the real work with
# rich-coloured output, timings and "what to do next" hints. Run `make` (or `make help`) to see them all.
# Pass extra arguments to the underlying tool with ARGS, e.g.  make test ARGS="-k plan -x"

UV     ?= uv
RUNNER := $(UV) run --quiet python makefile.py
ARGS   ?=

.DEFAULT_GOAL := help

TASKS := help info install install-docs test cov lint format format-check typecheck fix check \
         build cli docs-screens docs-video docs-serve clean clean-all

.PHONY: $(TASKS)

$(TASKS):
	@$(RUNNER) $@ $(ARGS)

# Anything else (e.g. a typo) still goes to makefile.py, which suggests the closest real task.
%:
	@$(RUNNER) $@ $(ARGS)
