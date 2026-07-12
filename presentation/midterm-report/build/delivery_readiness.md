# Delivery Readiness

- Workspace: `D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report`
- Delivery status: `needs_attention`
- Blocking reasons: `none`
- Warning reasons: `planning_warnings, planning_warnings_not_blocking, source_readiness_needs_attention`
- Readiness status: `needs_attention`
- Phase proof ledger: `none` valid=`False` gates=`0` proof_paths=`0` files=`0/0` missing=`0` route_required=`none` source=`readiness.execution_plan`
- Build report: `build\build_workspace_report.json` exists=`True`
- Build status: `succeeded` returncode=`0` failed_step=``
- Build speed: total_ms=`3838` steps=`6` renderer=`pptxgenjs` fast_first_pass=`False` skip_render=`True` visual_review=`False` longest=`qa:2738` render_ms=`287` qa_ms=`2738`
- Output PPTX: `build\galileo.pptx` exists=`True` sha256=`639f9cdd3bd7c3bd16be0206dde58785ee33b00e82550f13322d99fc81aa64f2` normalized_sha256=`2202ecce3945cd933cc7b6c9554ed4d324463f1b6e589a8a3c4457421b7388af`
- Visual review: required=`False` route_ledger=`False` cli=`False` run=`False` warnings=`0` sources=`none`
- Renderer: `{'requested': 'auto', 'used': 'pptxgenjs'}`

## Gates

- `acceptance_evidence_declared`: `True`
- `acceptance_evidence_files_satisfied`: `True`
- `build_report_exists`: `True`
- `build_succeeded`: `True`
- `fast_first_pass`: `False`
- `final_build_mode`: `True`
- `layout_density_checked`: `True`
- `layout_density_contract_required`: `False`
- `layout_density_floor_satisfied`: `False`
- `output_pptx_exists`: `True`
- `phase_proof_ledger_declared`: `False`
- `phase_proof_ledger_valid`: `True`
- `planning_warnings_blocking`: `False`
- `qa_run`: `True`
- `rendered_qa`: `False`
- `skip_render_allowed`: `True`
- `source_freshness_current`: `True`
- `source_readiness_ready`: `False`
- `visual_review_required`: `False`
- `visual_review_required_by_cli`: `False`
- `visual_review_required_by_route_ledger`: `False`
- `visual_review_run`: `False`
- `whitespace_warnings_blocking`: `True`

## Data Handoff

- Data handoff: status=`none` applied=`False` selections=`0`
- Data artifact rebuild: present=`False` persisted=`False` context=`` commands=`0`
- Data artifact contracts: figure_export=`False` figure_outputs=`0` registry_updates=`0` asset_updates=`{}`
- Data scout analysis: present=`False` persisted=`False` applied=`False` tasks=`0` findings=`0` visuals=`0` bindings=`0` targets=`none` variants=`none` open_questions=`0`

## Build Data Handoff

- Build data handoff: status=`none` applied=`False` selections=`0`
- Build data artifact rebuild: present=`False` context=`` commands=`0`
- Build data artifact contracts: figure_export=`False` figure_outputs=`0` registry_updates=`0` asset_updates=`{}`
- Build data scout analysis: present=`False` persisted=`False` applied=`False` tasks=`0` findings=`0` visuals=`0` bindings=`0` targets=`none` variants=`none` open_questions=`0`

## Artifact Context

- Artifact manifest: `assets\artifacts_manifest.json` exists=`False` valid=`False` outputs=`0` templates=`0`
- Bound artifact targets: outputs=`none` slides=`none` variants=`none` treatments=`none` unbound=`none`

## Reproducibility Context

- Replay contract: `none` exists=`False` seed=`none` renderer=`none` commands=`0` locked_fields=`0`
- Replay style: preset=`none` background=`none` headers=`none` footers=`none` charts=`none` tables=`none` figures=`none`
- Replay structure: slides=`0` variants=`none`
- Replay artifacts: manifest=`none` summary=`none` script=`none`
- Source inventory: `none`
- Resolved header variants: unique=`3` counts=`{"plain": 8, "side-rail": 1, "split-rule": 4}`
- Style-reference layouts: playbook=`style_reference_layout_playbook_v1` reference=`ref-clean-assay-report` applied=`0/11` skipped=`0` recipe_signatures=`5`
- Style-reference treatments: `{"chart": 2, "comparison": 2, "dashboard": 1, "decision": 1, "table": 5}`
- Style-reference variants: `{"chart": 2, "comparison-2col": 2, "lab-run-results": 5, "standard": 1, "stats": 1}`
- Style-reference recipe versions: `{"style_reference_content_recipe_library_v1": 11}`
- Style-reference slide map: `s2:comparison->comparison-2col, s3:dashboard->stats, s4:table->lab-run-results, s7:table->lab-run-results, s8:table->lab-run-results, s9:chart->chart, s10:chart->chart, s11:comparison->comparison-2col`

## Quality Context

- Slide quality contract: `none` exists=`False` title=`None` body=`None` chart=`None` footer=`None` whitespace=`False` evidence_anchor=`False` commands=`0`
- Outline quality alignment: `none` present=`False` persisted=`False` readability=`0` layout=`0` qa=`0` commands=`0`

## Layout Density

- Layout density: slides=`14` content=`13` min=`0.4858` avg=`0.7436` max=`1.0` floor=`0.55` low=`3` source=`build\qa\report.json`
- Low-density content slides: `3:0.5402, 6:0.4858, 11:0.4858`

## Next Action

- Recommended next action: `resolve_planning_warnings`
- Readiness next action: `resolve_planning_warnings`
- Action type: `edit_sources`
- Reason: Reusable/report decks should clear source-planning warnings before render.
- Slide IDs: `none`
- Planning paths: `design_brief.readability_contract.min_body_pt, design_brief.readability_contract.min_caption_pt`
- Warning types: `readability_contract`
- Suggested fields: `readability_contract, readability_contract.min_title_pt, readability_contract.min_body_pt, readability_contract.footer_reserved_inches`
- Action command: `python3 scripts/validate_planning.py --workspace D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report --report D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report\build\planning_validation.json`
- Source-edit handoff: `python3 scripts/advance_workspace.py --workspace D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report --execute --max-steps 3`

## Source Freshness

- Checked: `True`
- Source files: `24`
- Stale source files: `0`
- none

## Acceptance Evidence

- Checked: `True`
- Evidence items: `4`
- Evidence files: `0/0` existing
- Self outputs: `0`
- Missing files: `none`
- Blocking missing files: `none`

## Report Counts

- `planning`: `{"error_count": 0, "warning_count": 2}`
- `preflight`: `{"error_count": 0, "warning_count": 0}`
- `qa`: `{"design_error_count": 0, "design_warning_count": 0, "geometry_error_count": 0, "geometry_warning_count": 0, "overflow_count": 0, "overlap_count": 0, "visual_review_warning_count": 0, "visual_warning_count": 0, "whitespace_warning_count": 0}`

## Commands

- `readiness`: `python3 scripts/report_workspace_readiness.py --workspace D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report`
- `advance`: `python3 scripts/advance_workspace.py --workspace D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report --execute --max-steps 3`
- `strict_build`: `python3 scripts/build_workspace.py --workspace D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report --qa --fail-on-planning-warnings --fail-on-whitespace-warnings --overwrite`
- `visual_review_build`: `python3 scripts/build_workspace.py --workspace D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report --qa --visual-review --fail-on-planning-warnings --fail-on-whitespace-warnings --fail-on-visual-review-warnings --overwrite`
- `repeat_build`: `python3 scripts/build_workspace.py --workspace D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report --qa --skip-render --fail-on-whitespace-warnings --strict-preflight --overwrite`
