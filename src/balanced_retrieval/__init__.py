"""Part 2 of RAG²: balanced retrieval (Sohn et al., NAACL 2025, section 3.4).

The rationale of Part 1 is encoded once and searched in four corpora (PubMed, PMC, clinical guidelines, textbooks) by exact
inner product. The same number of best passages is taken from each corpus, the four lists are joined in a fixed order
without removing duplicates, and one cross-encoder orders the pool; the best k are kept.

    pool         joining the four candidate lists, and ordering a pool by score
    retrieve     one question through the four searches, the pool and the re-ranking
    diagnostics  what the final lists look like: each corpus's share and the duplicate rate across corpora
    scope        the search string that defines the Alzheimer's-scoped PubMed corpus (and so the PMC corpus)
    ncbi         a polite E-utilities client and the date-window search that gets past the 10,000-hit limit
    documents    a resumable batch download of documents into one JSON-lines file
    pubmed       the PubMed corpus: titles and abstracts of the records in the scope
    pmc          the PMC corpus: open-access full text of those records
    guidelines   the guidelines corpus: the open guidelines of the Meditron collection
    textbooks    the textbooks corpus: the 18 MedQA textbooks, joined back into books
    chunking     cutting a document into passages of 128 words that overlap by 32
    corpus       one corpus on disk: passages, a row-aligned embedding file, and a manifest of how they were made
    embedding    the MedCPT article encoder and the resumable writer of a corpus's embeddings
    search       exact inner-product search over the embeddings, block by block, so a corpus need not fit in memory
    scorer       the MedCPT cross-encoder that scores a query against each pooled passage
    records      one resumable record per question, bound to the settings of the run
    cli          python -m balanced_retrieval ...

Importing the package loads nothing heavy: numpy, torch and transformers are imported only where used.
"""
