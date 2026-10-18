# Demo data

This directory contains small, tracked artifacts for demonstrating the data
contract without committing a large live-news snapshot. The application fetches
fresh records at runtime and stores its working database in the ignored
`news.duckdb` file at the repository root.

## Files

- [`source_catalog.csv`](source_catalog.csv) lists every configured source,
  source type, ingestion mode, and canonical endpoint.
- [`articles_sample.jsonl`](articles_sample.jsonl) contains representative
  records using the same high-level fields as `newsengine.models.Article`.
  The records are compact synthetic examples; they are not a redistribution of
  publisher article text.

## Provenance

The live pipeline uses Google Finance via Google News RSS, CNBC, WSJ,
MarketWatch, Yahoo Finance, Investing.com, Benzinga, Seeking Alpha, Reddit
(`r/stocks` and `r/wallstreetbets`), and GDELT. See
[`newsengine/fetch.py`](../newsengine/fetch.py) for the executable source
configuration.

The Hugging Face
[`financial-news-multisource`](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource)
dataset is recorded as an optional historical/reference source in the catalog;
it is not downloaded automatically and is not required to run the live
pipeline. If a snapshot is obtained, keep it outside the repository or add
only a license-compliant, size-limited sample with its retrieval date and
license documented here.

## Runtime data flow

1. `newsengine.fetch` retrieves the configured RSS, Reddit, and GDELT sources
   concurrently.
2. `newsengine.normalize` resolves entities and ticker symbols.
3. The NLP pipeline adds sentiment, event, impact, and signal fields.
4. `newsengine.store` persists normalized records in DuckDB.

For reproducibility, record the collection timestamp, source name, endpoint,
and any dataset version whenever creating a local snapshot. Do not commit
credentials, API responses containing personal information, or unbounded live
feeds.
