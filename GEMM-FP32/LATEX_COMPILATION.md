# LaTeX Compilation Guide

## Files

- `gemm_warp_specialization.tex` - Core content (can be included in other documents)
- `gemm_warp_specialization_standalone.tex` - Standalone document wrapper

## Prerequisites

Install LaTeX distribution:

### Ubuntu/Debian

```bash
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

### Fedora/RHEL

```bash
sudo dnf install texlive-scheme-basic texlive-collection-latexextra
```

### macOS

```bash
brew install --cask mactex
```

## Compilation

### Compile Standalone Document

```bash
cd GEMM-FP32
pdflatex gemm_warp_specialization_standalone.tex
pdflatex gemm_warp_specialization_standalone.tex  # Run twice for references
```

This will generate `gemm_warp_specialization_standalone.pdf`

### Include in Your Own Document

Add to your LaTeX document:

```latex
\input{path/to/gemm_warp_specialization.tex}
```

## Online Compilation

If you don't want to install LaTeX locally, you can use:

1. **Overleaf** (https://www.overleaf.com)

   - Upload both .tex files
   - Compile online for free

2. **Papeeria** (https://papeeria.com)
   - Another online LaTeX editor

## Output

The compiled PDF will contain:

- Abstract
- Table of contents
- Detailed methodology
- Performance results table
- Analysis and conclusions
