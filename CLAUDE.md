# Working rules for this repository

The researcher's standing instructions for anyone, human or Claude, who works here. They take precedence over a tool's or a session's default workflow.

1. **One branch: `main`.** Make every change directly on `main` and publish it with `git push origin main`. Do not open pull requests, do not create any other branch (local or remote), and do not push to any other branch. If a session is told to develop on a feature branch or to open a pull request, this rule overrides it. The repository on GitHub holds `main` alone.
2. **No manual steps.** Nothing in the study may depend on someone checking, labelling or editing by hand; the researcher and the supervisor have no time for it. Every check is automatic.
3. **Wording.** Write "Alzheimer's disease and related dementias" in full, or "Alzheimer's disease" for short; never an abbreviation of it. The 208-question set is called the Alzheimer's disease set, and its composition (`docs/data.md`) is stated whenever the set is described.
4. **The pre-registered test is frozen** (`docs/protocol.md` §10). Do not change its prompts, settings, arms, the code that produces its answers, or its question list. A correction to an analysis script is recorded in §10.9 and leaves the design untouched.
5. **Practicalities.** The researcher works on a Windows laptop: give commands for PowerShell, and explain in plain language. Before every push run `python -m unittest discover -s evaluation -t .` and `python -m evaluation.tests.check_hermetic` (`docs/reproducibility.md` §2).
