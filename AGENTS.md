# AGENTS.md — working rules for this repository

Applies to every human and every agent (Claude Code, Codex, opencode, …) that
writes here. `CLAUDE.md` imports this file, so the two never drift.

`main` is **public** (`github.com:SourceShift/parlevox`, Apache-2.0). Every
commit that reaches it is public and irrevocable: a later removal, a force-push
or a relicense cannot claw back what has already been fetched, indexed or
forked. Treat publishing as the single irreversible act in this repo and gate it
accordingly.

## 1. The one rule

Nothing enters `main` that a stranger cannot read and reproduce: no secrets, no
private names, no operator-local paths, no internal tooling state, no raw
multi-model review output. When in doubt, keep it in a git-ignored directory.

## 2. What may never be committed

| Category | Examples | Where it belongs |
|---|---|---|
| Secrets | API keys, tokens, private keys, `.env`, `*.local.*` | untracked local file, `chmod 600` |
| Operator paths | `/Users/<name>/…`, `/Volumes/<disk>/…`, `~/ps/…` | nowhere — describe the artefact, not its location |
| Tool runtime state | run databases, `runs/`, worktree state, kickoff notes | git-ignored (`/.mini-ork/`, `/runs/`, `/kickoffs/`) |
| Raw review artifacts | lens reports, panel verdicts, run ids, "Response A–E" numbering, harness commands | git-ignored (`/docs/internal/`) |
| Reference-tree pointers | "reference copy: /abs/path", paths into a local clone of another project | name the upstream project, keep anchors relative |
| Business-private vocabulary | fundraising, investor, pitch-deck, valuation, seed round, term sheet material | private repo only |
| Customer / company / person names | vendor deals, prospects, individuals | private repo or the untracked terms file (§3) |
| Credential-shaped examples | `sk-…`, `ghp_…`, `AKIA…`, `xox?…`, PEM headers | only if the value itself says `test`/`fake`/`dummy`/`example` |

A path that only works on one machine is a bug as well as a leak: it breaks the
build for every contributor.

## 3. Confidential-term scanning

Two layers, one committed and one not.

**Committed (generic, safe to read):**

```
fundrais|investor|pitch[ -]?deck|venture capital|\bvaluation\b
|seed round|term sheet|@gmail\.com
```

**Not committed:** a private terms file, one regex per line, at
`$MINI_ORK_HOME/oss-guard-terms.txt`. The names it holds are themselves
sensitive, so it is untracked and a refusal cites a match **by line number
only**, never by name. A missing file skips the private layer; it does not
disable the generic one.

**Credential shapes**, each as a named group so a refusal can say *which* shape
matched without echoing the value:

| Shape | Pattern |
|---|---|
| `openai_key` | `sk-[A-Za-z0-9_-]{24,}` |
| `github_pat` | `ghp_[A-Za-z0-9]{30,}` |
| `github_fine_grained_pat` | `github_pat_[A-Za-z0-9_]{30,}` |
| `aws_access_key` | `AKIA[0-9A-Z]{16}` |
| `slack_token` | `xox[bpa]-[A-Za-z0-9-]{10,}` |
| `private_key` | `-----BEGIN [A-Z ]*PRIVATE KEY` |

The fixture exemption is scoped to the **matched value**, not the line:
`key = "sk-test-value-never-shown"` passes; `key = "sk-live-…"  # test` is
still refused.

## 4. Where the gate runs

One choke point, immediately before publishing to `main` — after the rebase so
that `origin/main..HEAD` is exactly this branch's own commits, and before the
green test gate and the push. A refusal pushes nothing and exits non-zero.

- **Scan history, not the net diff.** Every commit's patch *and* every commit
  message in `origin/main..HEAD`. A secret added in one commit and removed in
  the next still lands in public history once pushed.
- **Pin the diff format** so local config cannot make the scan fail open:
  `--no-color --no-ext-diff --no-textconv --no-renames --src-prefix=a/ --dst-prefix=b/`.
- **Walk hunks, not lines.** Inside a hunk every `+` line is content; a line
  that itself starts with `++` is not a file header.
- **Never echo the finding.** Report `path:line matches <generic term>` or
  `path:line matches <credential shape name>` or `private term #<n>`.
  Never the matched secret or the private name.
- **Override is per-merge and reviewed.** A reviewed public mention may be
  allowed explicitly for that one merge; the override, its reason and the
  reviewer go in the PR description. It is not a standing setting.

## 5. OSS-readiness checklist

Run before any public release or push to a public remote. Each item is pass/fail
**with evidence**, not an opinion.

1. **License** — `LICENSE` exists, is a recognised OSS license, and the README
   and docs do not contradict it.
2. **No committed secrets** — scan the tree and `main...HEAD` for API keys,
   tokens, private keys, `.env`/secret files. `*.local.*` and secret env files
   are git-ignored, not committed.
3. **No proprietary IP** — no private-domain content (customer or product
   specifics, regulated financial/legal domain terms from a private vertical) in
   tracked files.
4. **No hardcoded personal absolute paths** — nothing that breaks for others.
5. **README adequacy** — says what the project is, how to install and run it,
   and its license.
6. **`.gitignore` hygiene** — covers tool state, secrets, run artifacts, caches.
7. **Dependency licences** — no GPL-incompatible vendored code under a
   permissive licence.

## 6. Open-core boundary

Draw the line by **architecture**, not by licence: a permissive licence does not
stop someone hosting the OSS.

Ask four questions before writing code. Any "yes" on 2–4 sends the feature to
the private repo:

1. Does it only make sense **across a team, over time, or under audit**?
   → private (seats, SSO, RBAC, retention, cross-run evidence, billing).
2. Is it **multi-tenant** — org/project/environment boundaries, per-tenant
   quotas, tenant-scoped secrets? → private.
3. Is it **policy rather than mechanism**? A protocol, a runner or a registry
   is mechanism → OSS. Pricing, quotas, routing preferences, retention tiers,
   approval policy → private.
4. Would a **single developer on a laptop** get value from it with no server?
   → OSS. If it only works once infrastructure is standing, it is hosted
   → private.

Corollary: when the private layer needs a hook in the OSS, upstream the
**generic seam**, never the commercial behaviour. `register_*` is OSS; the
scheduler that uses it is private.

Dependency direction is one-way: **private → OSS**, never OSS → private. The
contract layer (schemas, protocols, HTTP surfaces, plugin entry points,
certificate/portable documents) is public and versioned; the private repo's CI
runs against the *released* package, not a source checkout.

**What must stay OSS:** the runtime and its invariants, the provider traits and
manifest, the sidecar protocol, the verification and observability of the public
path, commodity adapters, the CLI and SDKs, configuration formats, and budget
*mechanism*. **What must stay private:** tenancy and placement, fleet
operations, metering and billing, enterprise access controls, and operated
data/analytics.

## 7. Local-only material in this repo

Already git-ignored, and must stay that way:

- `/.mini-ork/` — orchestrator state, run database, secrets
- `/kickoffs/` — internal build briefs
- `/docs/internal/` — raw tool output, panel verdicts, speed notes
- `/runs/` — development logs
- `reference/` — read-only clones of other projects for comparison
- `/.claude/skills/` — skills materialised per task

If a document is only meaningful to the people operating this machine, it lives
in one of those directories. If it should be readable by a contributor, it lives
in `docs/` and is scrubbed to the rules above.

## 8. Before you push

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Then scan what you are about to publish — history and messages, not just the
diff — against §3. See `CONTRIBUTING.md` for the review bar.
