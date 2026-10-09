# REPA-G project page

Static page for "REPA-G: Test-Time Conditioning with Representation-Aligned Visual Features" (NeurIPS 2026).

Deploy with GitHub Pages: push the content of this folder to the `gh-pages` branch of `valeoai/REPA-G`
(or to a `docs/` folder on `main`) and enable Pages in the repository settings.

- `index.html` — page
- `static/css/style.css` — styles (light/dark themes)
- `static/js/data.js` — numbers from the paper's tables, plus Figures 3 and 9 digitised from the PDF vectors
- `static/js/main.js` — interactive figures (no build step; MathJax loads from cdnjs)
- `static/img/` — images extracted from the paper PDF and the repo (WebP)

Preview locally: `python -m http.server` in this folder, then open http://localhost:8000.
