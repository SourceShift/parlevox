# Contributing

Thanks for helping. This is a small runtime with strong invariants, so the bar
for a change is "does it keep the invariants in
[`docs/architecture/DESIGN.md` §4](docs/architecture/DESIGN.md) true?" — not
just "does it compile".

## Licensing of contributions

No CLA, no sign-off ritual. By opening a pull request you agree your
contribution is licensed under Apache-2.0, the same terms as the project
(inbound = outbound). That is all that is needed.

The commercial edition (hosted service, enterprise add-ons) is separate code and
is not developed in this repository, so it places no additional burden on
contributors.

## Before you open a PR

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
PYTHONPATH=sdks/python python3 -m pytest sdks/python/tests -q
cargo run -q -p voicekit-cli -- demo --fake
PYTHONPATH=sdks/python cargo run -q -p voicekit-cli -- demo --sidecar
```

Both demos must print the same event sequence; the sidecar one also reports
`sidecar_dropped_frames`.

## What reviewers look for

- **Rust idiom.** The repo follows the `rust-skills` rule set (265 rules). Cite
  the rule id in the PR description when you fix or deliberately deviate from
  one — e.g. `num-cast-try-from`, `api-must-use`, `err-doc-errors`.
- **No silent loss.** New buffers need a capacity and an `OverflowPolicy`;
  new drop paths need a counter.
- **No new dependency without a reason** in the PR description.
- **Wire compatibility.** `voicekit-ipc` frame layout changes are breaking;
  call them out explicitly.
- **Tests that fail on the old code.** A bug fix without a regression test is
  not finished.

## Reporting a bug

Include the crate, the command you ran, and the full output. For sidecar issues,
the Python version and the plugin's `Plugin` subclass help a lot.
