# Directory Guide — SWM Project

This document describes the repository layout (based on the uploaded screenshot) and recommendations for collaborators.

## Top-level structure (interpreted)

```
SWM PROJECT/
├─ CHARTS/
├─ Extracted_News/
│  ├─ .ipynb_checkpoints/
│  ├─ 2018_01_d157b48c57be246ec7dd80e.../
│  ├─ 2018_02_d157b48c57be246ec7dd80e.../
│  ├─ ...
│  ├─ 2019_01_d157b48c57be246ec7dd80e.../
│  ├─ saved_models/
│  └─ saved_outputs/
├─ SWM_Project.ipynb
├─ SWM_Start.ipynb
├─ News/ (folder)
├─ CHARTS.rar
├─ conda-base-packages-before-pytorch-... (installer or note)
├─ LLM Ladder Details.docx
├─ News.zip
├─ requirements.txt
├─ Sample-Proposal-1.pdf
├─ Sample-Proposal-2.pdf
├─ Sample-Proposal-3.pdf
├─ WIJ17-Guidelines.pptx
└─ WIJ17.pdf
```

## Folder explanations & collaborator instructions

- `CHARTS/` — store generated visualizations (PNG/PDF). Keep versioned charts or date-stamped names to avoid overwriting.

- `Extracted_News/` — **primary input data**. Contains subfolders per month (e.g., `2018_01_...`). Do not commit large raw data to git. Use `.gitignore` for data directories or add instructions in README for obtaining data.

  - `.ipynb_checkpoints/` — Jupyter internal checkpoints; ignore in git.

  - `saved_models/` — should contain trained model files. Consider adding this to `.gitignore` if models are large; alternatively provide a `models/` release bucket.

  - `saved_outputs/` — outputs such as CSVs or processed datasets. Keep read-only for downstream users.

- `SWM_Start.ipynb` and `SWM_Project.ipynb` — notebooks. Keep notebooks focused: `SWM_Start` for data ingestion/prep, `SWM_Project` for modeling/experiments (if that matches your workflow).

- `News/`, `News.zip` — archive copies of news data. `News.zip` can be used to quickly bootstrap `Extracted_News/`.

- `requirements.txt` — Python dependencies. Keep updated when adding packages.

- `CHARTS.rar`, `Sample-Proposal-*.pdf`, `WIJ17*` — auxiliary files and proposals. Archive large binary artifacts outside git when possible.


## Recommended `.gitignore` snippets

```
# Data
Extracted_News/
News.zip
saved_models/
saved_outputs/
.ipynb_checkpoints/
# Python
.env
.venv/
__pycache__/
```


## How collaborators should work
- Create feature branches for experiments.
- Store large artifacts (models, zipped datasets) in external storage (S3/Google Drive) and add download scripts to `scripts/`.
- Use `saved_models/` only for small demo models in repo; otherwise add pointer files with instructions to download.


## Who edits what (suggested)
- Data ingestion: person/team A — updates under `Extracted_News/` and preprocessing cells in `SWM_Start.ipynb`.
- Modeling: person/team B — uses `SWM_Project.ipynb`, stores models in `saved_models/` (external storage preferred).
- Reporting: person/team C — generates plots in `CHARTS/` and reports in `docs/`.
