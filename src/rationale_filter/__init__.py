"""Part 3 of RAG²: rationale-guided filtering (Sohn et al., NAACL 2025, section 3.2).

A small sequence-to-sequence model judges each retrieved snippet on its own, helpful or not helpful, and only the helpful
snippets reach the answering model. Its training labels come from comparing the answer, and the perplexity of the
rationale, without and with the snippet (Figure 2 of the paper).

    labels         the label rule of Figure 2, the training tokens, and the threshold of the top 25% of perplexity differences
    inference      the filter input, the decision between the two label tokens, and keeping the helpful snippets
    perplexity     the perplexity of a rationale from its token log-probabilities, without and with a snippet (D17)
    scorer         the log-probability of each token of a text under a Llama-3 prompt, with llama.cpp
    label_records  one resumable record per question-snippet pair, and the labels made from all records together
    labelling      the loop that turns pairs into records: strict answers first, then the two perplexities
    training_data  the training file in the format of the released sample, refusing to overwrite labels
    filter_eval    accuracy per class, the majority rate and the AUC of the filter on held-out labels
    filter_model   Flan-T5 with the two label tokens: the first-step logits, windows of 512 tokens, batched judging (needs torch)
    training       the authors' fine-tuning of the filter, a memory test, and evaluation on labelled files (needs torch)
    cli            python -m rationale_filter label | labels | prepare-model | train | memory-test | evaluate

Importing the package loads nothing heavy: numpy, llama_cpp, torch and transformers are imported only inside the functions that use them.
"""
