# data_loading

Downloads Mendeley Data archives, extracts PCAP captures, and parses them into
a combined CSV.

## Modules

- `download.py` — lists files via the Mendeley public API and downloads `.rar`/`.zip` archives.
- `extract.py` — extracts `.rar`/`.zip` archives into a target directory.
- `parse_filename.py` — parses a PCAP filename into `FilenameMetadata`.
- `parse_pcap.py` — streams PCAP packets into a pandas DataFrame.
- `pipeline.py` — orchestrates the full chain and writes `combined.csv`.

## Config

All settings live in `config/data_loading/default.yaml`:

- `datasets` — registry of `dataset_id` → `name` / `version`.
- `selected_dataset_ids` — `null` (all), a single id, or a list of ids.
- `download` — API base URL, user agent, archive extensions, chunk size, and `max_files` (max archives per dataset; `null` = all).
- `extract` — whether to overwrite existing extractions.
- `filename_pattern` — regex with named groups
  `capture_timestamp`, `application`, `download_speed`, `upload_speed`.
- `parsing` — per-file packet limit.
- `paths` — `raw` / `interim` / `processed` directories and the combined CSV path.

## Run

```bash
pip install -e .
python main.py                 # process selected (or all) datasets
python main.py selected_dataset_ids=9frgkybxhn          # single dataset
python main.py selected_dataset_ids=[96jwzrp7fd,zsffy3j9y6]  # multiple
```

## Output

Each dataset produces `data/processed/<name>/packets.csv`, and all selected
datasets are concatenated into `data/processed/combined.csv` with a
`dataset_name` column distinguishing rows.
