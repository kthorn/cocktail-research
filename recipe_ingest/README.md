# Recipe Ingest

Tools for ingesting cocktail recipes from scraped HTML (Punch, Difford's) into
the mixology.tools database.

## The e2e pipeline (day-to-day entry point)

`ingest_pipeline.py` combines recipe confirmation, ingredient mapping, and
batch auto-ingest in one web app:

```bash
# optional, enables "create new ingredient" in the database:
export MIXOLOGY_USERNAME=your-login
export MIXOLOGY_PASSWORD=your-password

python ingest_pipeline.py --source punch     # or diffords, default port 5003
```

For each recipe it shows the original HTML next to the rationalized preview
(unmapped ingredients highlighted). You can **Accept** (optionally editing the
JSON), **Reject**, and — when a recipe has unmapped ingredients — map them
immediately:

- pick an existing ingredient from the API list (fuzzy suggestions + search),
- or **Create in database**: a canonical name (prefilled from the raw name),
  optional parent/ABV/sugar/acidity, POSTed to the API and mapped with its real
  ID in one step. Credentials come from `MIXOLOGY_USERNAME` / `MIXOLOGY_PASSWORD`
  (Cognito login); creating requires editor access on mixology.tools.

Every mapping decision is written to `data/ingredient_mappings.json`
immediately, so later recipes (and auto-ingest) resolve to exactly what is in
the database. Marked-edits recipes are kept in `data/validation_log.json`
under `edited_recipes`.

When the queue is empty, use **Run batch auto-ingest** (or POST `/api/batch`)
to auto-ingest every other fully-mapped recipe and write upload-ready batch
files to `output/rationalized-recipes-<source>-batch-NNN.json`. Review and
upload these to mixology.tools yourself.

## Supporting files

| File | Purpose |
| ---- | ------- |
| `data/ingredient_mappings.json` | Raw ingredient name → `{id, name}` API mapping |
| `data/validation_log.json` | Per-source progress: accepted/rejected/auto-ingested/needs_review/edited_recipes |
| `data/new-ingredients-for-upload.json` | Legacy queue file; the pipeline creates ingredients directly now |
| `batch_rationalize_recipes.py` | Standalone CLI for the batch auto-ingest pass (also used by the pipeline) |
| `cognito_token.py` | Cognito access-token helper (USER_PASSWORD_AUTH) |
| `merge_mappings.py` | Merge two mappings files with conflict resolution |
| `migrate_validation_log.py` | One-time migration utility, not part of the normal loop |
| `templates/pipeline.html` | The pipeline UI |

## Tests

```bash
python -m pytest recipe_ingest/tests -q
```
