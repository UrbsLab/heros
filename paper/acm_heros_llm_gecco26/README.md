# HEROS-LLM ACM Draft

This directory contains an ACM `acmart` conference-style draft for the HEROS+LLM workshop paper.

Files:

- `main.tex`: manuscript source
- `references.bib`: bibliography

Suggested compile command in an environment with TeX + `acmart` installed:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Notes:

- The draft now uses the HEROS paper author list, but the affiliation block should still be aligned with the final HEROS paper before submission.
- The manuscript is written against the finalized experiment artifacts in:
  - `output/heros_llm/mux6_50_test_final2`
- `Comprehensiveness` and `Sufficiency` are intentionally not reported in the results because the perturbation-based rerun implementation is not yet available.
