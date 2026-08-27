.PHONY: help clean smoke humanscale fineweb ablations figures figures-acl figures-clean loss-curves sync-manifests test test-ficarra test-fast

help:
	@echo "beetle-analyze Makefile targets:"
	@echo "  make clean           Remove __pycache__/.pyc/.pytest_cache"
	@echo "  make smoke           End-to-end sanity check on one model"
	@echo "  make humanscale      Full pipeline over 20 HumanScale models"
	@echo "  make fineweb         Full pipeline over Beetle-FineWeb models"
	@echo "  make ablations       24-repo minimal ablation sweep"
	@echo "  make figures         ACL main-body figures only (fast)"
	@echo "  make figures-acl     Full ACL-quality figure pack (Fig 1-12, 300 DPI)"
	@echo "  make figures-clean   Delete all plots/fig*.pdf artefacts"
	@echo "  make loss-curves     FLORES + held-out trajectories + 8 PDFs"
	@echo "  make sync-manifests  Refresh FineWeb + ablation JSON manifests"
	@echo "  make test            pytest tests/ + aoa/tests/"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

smoke:
	bash scripts/smoke_test.sh

humanscale:
	bash scripts/run_humanscale_full.sh

fineweb:
	bash scripts/run_fineweb_full.sh

ablations:
	bash scripts/run_ablations_full.sh

figures:
	bash scripts/run_paper_figures.sh

# ACL camera-ready figure pack: regenerates the full Fig 1–12 set from the
# evals_checkpoints/ dataset at 300 DPI with ACL-column sizing.
figures-acl: figures-clean
	PYTHONPATH=. python -m plots.make_plots
	PYTHONPATH=. python -m plots.make_cross_eval_plots
	PYTHONPATH=. python -m plots.make_learning_dynamics
	PYTHONPATH=. python -m plots.make_plots_acl --palette-check
	PYTHONPATH=. python -m plots.make_plots_acl --verify

figures-clean:
	@find plots -maxdepth 1 -name 'fig*.pdf' -delete
	@rm -f 'plots/fig2_blimp_convex_hulls 2.pdf' \
	       'plots/fig4_crosslingual_heatmap 2.pdf' \
	       'plots/fig5_flores_delta_nll 2.pdf'

# Compute FLORES + held-out NLL trajectories per checkpoint, then render
# the four loss-curve comparison figures (×2 x-axes = 8 PDFs total).
loss-curves:
	bash scripts/eval_loss_curves_beetle_humanscale.sh
	bash scripts/eval_loss_curves_beetle_fineweb.sh
	bash scripts/eval_loss_curves_bgpt.sh
	PYTHONPATH=. python -m plots.make_loss_curves --tokens-axis both

sync-manifests:
	python3 scripts/sync_fineweb_manifest.py --beetlelm ../beetlelm
	python3 scripts/sync_ablation_manifest.py --beetlelm ../beetlelm

test:
	pytest tests/ aoa/tests/ -v

test-ficarra:
	pytest aoa/tests/test_ficarra_convergence.py aoa/tests/test_ficarra_signatures.py -v

test-fast:
	bash scripts/test_main_fast.sh
