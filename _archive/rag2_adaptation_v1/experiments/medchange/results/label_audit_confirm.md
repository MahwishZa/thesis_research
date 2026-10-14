# Label audit, confirm split

Independent labeler: Qwen2.5-7B-Instruct-Q4_K_M.gguf; the authors' rubric; 528 items (0 unparsed).

* Agreement with the gold label (newest version): **81.4%**; Cohen's kappa **0.7161**.
* Label change between versions reproduced for 72.5% of 353 changed items.
* Label-stable items (both labelers agree): 430.

| Gold class | items | agreement |
|---|---|---|
| SUPPORTED | 234 | 86.8% |
| REFUTED | 126 | 91.3% |
| NOT ENOUGH INFORMATION | 168 | 66.7% |

Agreement measures how reproducible the labels are with another model; it is not medical truth, and two models can share a bias.
