# mini-ork: what makes a run slow, and what was changed

The vendored oracle lives in `.mini-ork/` and is re-vendored from the source
clone at `/Users/admin/ps/mini-ork` (see the mini-ork skill's "Updating
mini-ork safely"). This note records the 2026-11-23 speed pass: what the
measurements said, what changed in the source clone, and what the operator
must know when dispatching.

## Where the wall-clock went

Measured from `state.db` `llm_calls` / `node_attempts` over the runs of
2026-11-22/23 (all `recursive-validate-impl`):

| Stage | Observed | Why |
|---|---|---|
| implementer | 230–981 s per node | the node re-proved the whole repository (workspace tests, clippy, benchmarks) on top of a 2.8 M-token cached context |
| panel lens | 250–1047 s, one seat 17 min | no per-node bound existed; every lens also re-ran the repo's tests |
| post-run learning | ~3–4 min on the completion path | rubric + gradient-extract + pattern-induct + GRPO writeback + OTel run after the verdict, blocking the CLI from returning |
| panel seats | 1 failed dispatch per run | the Kimi gateway answers 403 `permission_error` for every model id (subscription lapsed), so the seat fell back to MiniMax — the family the overlay had already moved off its 429 |

## The five changes (in the source clone, re-vendored here)

1. **Implementer verification budget** —
   `recipes/recursive-validate-impl/prompts/implementer.md` now caps the node
   at four scoped commands (`cargo check -p`, the one changed test target,
   `cargo fmt --check`, the DoD demo probe) and forbids `--workspace`,
   `--all-targets`, clippy-workspace, benchmarks and repeated builds. Those
   are the verifiers' job; the node was paying for them twice.
2. **Panel: conditional literature step, no self-testing** —
   `prompts/tier4-panel-review.md` only runs the arXiv check for
   algorithm/model/research-shaped deliverables (≤2 min, else
   `- not applicable`) and says tier evidence is authoritative: a lens may not
   re-run the repository's test/lint/build commands except to replace missing
   or contradictory tier evidence.
3. **Per-node `timeout_minutes`** — the field was declared in
   `schemas/workflow.schema.json` but no compiler read it. `WorkflowNode` now
   carries it, the compiler validates it (positive float; anything else is a
   `WorkflowCompileError`), and `execute_handlers` publishes it as
   `MO_NODE_TIMEOUT_S` at that node's boundary, capped by the remaining
   dispatch deadline, and restores the operator's pinned base for every node
   that does not declare one. The four lens nodes declare `12` (observed
   p95 ≈ 8 min; a killed lens is one missing report and quorum is 3 of 4).
4. **Post-run learning is deferrable** — `MO_POST_RUN_LEARNING=0` skips the
   side-channels and prints the settle command; `mini-ork settle <run_id>`
   runs them later from the run's persisted `task_class` and verdict. This
   repo dispatches with the variable set, so the CLI returns at the verdict.
5. **Lane map** — `config/agents.yaml` routes `kimi_lens` to `deepseek_flash`
   until the Moonshot subscription is restored; the vendored
   `config/agents.local.yaml` also carries `budget.daily_cap_usd: 2000.00` so
   the operator cap survives re-vendor. Its `minimax_lens: deepseek` mapping
   dates from the 2026-11-22 MiniMax 429, but a MiniMax seat served a lens
   successfully on 2026-11-23 (437 s), so the dispatch now warns that
   kimi/minimax/deepseek all resolve to the DeepSeek family. Restoring
   `minimax_lens: minimax` buys back a fourth family; a failed seat is one
   missing report and quorum is 3 of 4. Flip it once the 429 stops recurring.

Two panels bugs were also backported from the vendored copy: the
`skill_registry` `FileExistsError` race (parallel lenses materializing the same
skill) and the tier2 verifier's feature-gated `cargo test --features …` rerun.

## Operating notes

- Dispatch command (canonical): see `docs/architecture/PARITY-LIVEKIT-AGENTS.md`
  §sequencing; it pins `MINI_ORK_ROOT`/`MINI_ORK_HOME` to `./.mini-ork`,
  `MO_TARGET_CWD` to the repo, and the budget env.
- After a deferred run: `mini-ork settle <run_id>` (needs the same
  `MINI_ORK_HOME`/`MINI_ORK_DB`; exit 2 means unreadable db, unknown run, or
  missing run dir).
- `mini-ork doctor`'s provider preflight checks **key presence only** — an
  expired subscription passes it and then 403s at dispatch. Probe the gateway
  with `curl` when a lane dies with `rc=1` and no output.
- Re-vendor: rsync `mini_ork bin recipes schemas config db` **without
  `--delete`**, excluding `state.db*`, `runs/`, `kickoffs/`, `__pycache__`,
  `config/secrets.local.sh`; back up first. `config/skills` and
  `config/task_classes` are operator registries and must survive.
