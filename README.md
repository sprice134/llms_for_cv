# SEM powder characterization benchmark

Code, prompts, task definitions, model responses, and numerical results for **Evaluating LLMs for Automated SEM Powder Characterization Workflows** (AI4Mat at NeurIPS 2026). The rendered manuscript is `paper/camera_ready.pdf`.

## Results

The release contains 7,183 tasks for each of 13 model configurations (93,379 evaluations): 5,120 direct-inference, 380 materials-specific, 555 analysis-routing, 768 repeatability, and 360 comparative-characterization tasks. The final paper reports 92.0% accuracy for retrieval from precomputed morphology summaries versus 48.8% for inference from raw particle records; 10 of 13 configurations reach at least 95% accepted routing accuracy. Failed and unparsed requests remain in the denominator.

`results/evaluation/*.csv.gz` contains the scored task-level model responses, including errors, parsed answers, reference values, routing decisions, and token counts. These files are sufficient to audit the reported model results without repeating paid inference. `results/manifests/*.jsonl.gz` contains the exact prompts, reference answers, image split, routing schemas, and accepted equivalent calls. `paper/generated/potential/*.csv` contains the paper's final model and appendix summaries; `paper/generated/tables/table1.tex` and `paper/generated/figures/fig2.pdf` are the main table and Figure 2. `paper/fig1_workflow.png` is the workflow figure extracted losslessly from the rendered manuscript.

## Verify and reproduce

```bash
python3 scripts/verify_results.py
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 scripts/materialize.py
.venv/bin/python src/build_paper_artifacts.py --no-refresh --artifact-set main --format pdf
```

The verifier reads the compressed files directly and checks suite coverage and headline statistics against the released paper summaries. The artifact builder regenerates the main table and figure from the scored outputs. To regenerate task manifests from the 20 held-out images and particle measurements:

```bash
.venv/bin/python src/build_task_manifest.py --suite all --image-limit 20 --output results/manifests/tasks_image20.jsonl
.venv/bin/python src/build_task_manifest.py --suite tool_routing --image-limit 15 --output results/manifests/tool_routing_image15.jsonl
.venv/bin/python src/build_task_manifest.py --suite case_study --image-limit 15 --output results/manifests/case_study_image15.jsonl
.venv/bin/python src/build_reliability_manifest.py --source-manifest results/manifests/tasks_image15.jsonl --output results/manifests/reliability_image15_repeat3.jsonl
```

For repeatability, first generate the 15-image base manifest with `--suite all --image-limit 15 --output results/manifests/tasks_image15.jsonl`. The archived manifests remain the reference used for the reported scores. `src/run_inference.py` runs the same prompts against a configured API or local model, and `src/evaluate_numeric.py` scores raw output JSON against a selected manifest; both expose CLI help. API access and local model weights are required only for new inference.

The benchmark starts from extracted per-particle morphology measurements and the 20 held-out SEM images. Segmentation model training and weights belong to the upstream DualSight pipeline and are outside this evaluation release.
