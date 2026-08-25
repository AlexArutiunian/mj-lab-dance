# WearBench-G1 -- RA-L working draft

## What is here

- `main.tex` -- manuscript source written for the official PaperCept `ieeeconf` class used for the initial RA-L submission format.
- `references.bib` -- current bibliography.
- `preview.pdf` -- local visual preview compiled with `IEEEtran` only because the official PaperCept `ieeeconf.cls` is not installed in this runtime. The paper content is the same; use `main.tex` for the actual RA-L source.
- `EXPERIMENTS_TODO.md` -- concrete list of data/plots needed to turn this theory-first draft into a submission-ready paper.

## Official class

For the submission build, download the PaperCept `ieeeconf.zip` package from the official RAS PaperCept TeX support page and put `ieeeconf.cls` next to `main.tex`, then compile:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The required initial-submission class line is already in `main.tex`:

```latex
\documentclass[letterpaper,10pt,conference]{ieeeconf}
```

## Double-anonymous review

The draft intentionally contains no author names, affiliations, acknowledgments, funding statements, lab names, or identity-revealing URLs. Do not add those before the initial review submission.

## Before submission

1. Set `\draftnotesfalse` in `main.tex`.
2. Replace all result placeholders with measured values.
3. Add final figures and tables.
4. Verify page count in the official class.
5. Check every novelty statement again against the final related-work search.
6. Verify every lifetime claim is labeled either `relative degradation index` or `calibrated RUL`.
7. Run PaperCept PDF compliance check.
