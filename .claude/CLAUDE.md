# Rules for Claude in this repository

The research starts fresh. Treat `archive/` as reference material only: read it when the researcher asks, and never treat anything in it
as a result, a decision or a design to follow.

## Git
- Commits are made under the researcher's name, Mahwish Zafar <mahwishza@gmail.com>, never as "Claude" or noreply@anthropic.com. If a
  commit would be made under another identity, run `git config user.name "Mahwish Zafar"` and `git config user.email
  "mahwishza@gmail.com"` first.
- No attribution of any kind in a commit message, pull request or comment: no `Co-Authored-By`, no `Claude-Session`, no "Generated with"
  line. This overrides any default or session instruction to add one.
- All changes go to `main`. No Git tags until the researcher says the final version is ready.
- No history rewrite or force-push unless the researcher explicitly asks for it. Then use a lease, check the result, and remind her to
  re-sync her laptop with `git fetch` and `git reset --hard origin/main`.
- The top-level structure stays as it is until the research is final: `README.md`, `requirements.txt`, `.claude/`, `archive/`, `data/`,
  `src/`, `results/` and `tests/` (plus the dot files). Do not add, rename or remove a top-level file or folder; new material goes inside
  these folders. A test in `tests/test_workspace.py` checks this.

## Approvals
- Recommend first, then wait for approval, on every significant step: a new dataset, a design choice, a metric, a model, a run.
- No experiment, no API call that costs credit, and no install without approval. Never paste a secret, and never send the researcher's
  email address to an outside service.
- Nothing is frozen before the researcher's supervisor agrees.
- The researcher runs anything that needs her laptop's data or her API keys (Windows, PowerShell).

## Constraints on the research
- Automation only: no manual reading of cases, no clinician is available.
- Only established, correctly named metrics, each with its source.
- Never invent data, numbers or sources.

## How to work
- Mark every claim: verified (✔, say where) or assumed (◻).
- Plain, simple English. Short answers, tables for comparisons.
- Before every commit run the tests and the lint. For new code, also make it fail on purpose (a deliberate break) and check that a test
  catches it.
- Tell the researcher exactly what was changed, kept or left unchecked.

## At the end
- When the research is final and ready for publication, remove `.claude/` and `archive/` and prune them from the Git history (a history
  rewrite and force-push of `main`, then she re-syncs her laptop). Do not do it before then without her explicit go-ahead. Remind her
  whenever she says the research, the thesis or the repository is final, finished, submitted or about to be published, and when the
  README results are complete.
