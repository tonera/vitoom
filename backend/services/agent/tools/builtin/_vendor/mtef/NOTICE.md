# MTEF (MathType Equation Format) → LaTeX

## Provenance

This subpackage is a Python translation derived from
[`zhexiao/mtef-go`](https://github.com/zhexiao/mtef-go) (Apache License 2.0,
Copyright (c) Zhe Xiao). The intermediate Python translation
[`AndyQsmart/MTEF-py`](https://github.com/AndyQsmart/MTEF-py) was used as a
reference; both upstreams' license terms are respected here.

The vendored files in this directory:

- `_mtef.py`     — MTEF v5 record-stream parser & LaTeX renderer
- `_records.py`  — record / option / template / embell type tables
- `_chars.py`    — MathType character → LaTeX command map
- `LICENSE`      — full text of upstream `mtef-go` Apache License 2.0

## Local modifications

Compared to upstream we apply the following minimal changes (Apache-2.0 §4
requires modifications to be marked):

- replace upstream's hand-written OLE compound-file reader with `olefile`
  (already pulled in transitively by `markitdown`), removing ~470 lines of
  redundant code;
- strip all `print('(DEBUG)…')` / `print(err)` traces (they leak onto stdout
  during normal use and would pollute backend logs);
- expose a single thin entry point `mathtype_ole_to_latex(bytes) -> str | None`
  that swallows parser errors so the calling pipeline can fall back to the
  source WMF on any failure;
- normalise output spacing so the resulting LaTeX is suitable for inline
  embedding inside Markdown (`$…$`).

No new MTEF semantics or translation rules are added — purely plumbing /
hygiene changes that do not affect the parser's behaviour on valid MTEF v5
streams. Bug-for-bug compatibility with upstream `mtef-go` is preserved.
