# Label audit, dev split

Independent labeler: Qwen2.5-7B-Instruct-Q4_K_M.gguf; the authors' rubric; 226 items (0 unparsed).

* Agreement with the gold label (newest version): **83.2%**; Cohen's kappa **0.7422**.
* Label change between versions reproduced for 68.2% of 151 changed items.
* Label-stable items (both labelers agree): 188.

| Gold class | items | agreement |
|---|---|---|
| SUPPORTED | 105 | 87.6% |
| REFUTED | 49 | 95.9% |
| NOT ENOUGH INFORMATION | 72 | 68.1% |

Agreement measures how reproducible the labels are with another model; it is not medical truth, and two models can share a bias.
