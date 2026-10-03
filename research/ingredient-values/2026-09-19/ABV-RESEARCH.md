# ABV research for remaining leaves

Three Luna agents researched all 158 leaves from the refreshed production inventory. No database writes were performed.

The 98 resolved values were uploaded and verified against production. The completed import CSV has been removed; all values and sources remain in the complete research CSV.
- [Complete sourced research](abv-resolved-research.csv): all 158 rows, with values, status, source URLs, evidence, and notes.
- [Remaining review CSV](abv-still-needs-review.csv): 60 entries needing a brand/market selection, recipe decision, or stronger evidence. Numeric values in this file are candidates, not approved imports.

All IDs, exact ingredient names, missing ABV, and leaf status were rechecked against production after research. The completed import CSV passed the application's bulk-value parser. The coordinator reviewed all result rows and independently spot-checked 21 source/value matches. Ready sources were opened by the researching agents; credible exact-product retailer sources are accepted alongside producers and importers. Research artifacts and validation are in `abv-research-pass/`.

Bénédictine is included at 40% from its official product page. Generic categories such as Absinthe, Mezcal, and sherry styles need a selected bottling or an explicit category-default policy. Donn's Spices #2 has a 16% recipe-estimate candidate using the database's equal-parts recipe and current component values, retained for review. No new zero defaults were assigned.

At the start of this pass, all 84 entries marked ready in the original September 8 pass still had null ABV despite the earlier reported import. This pass used production values rather than assuming those updates had persisted. The user reported uploading the 98-row resolved import; production verification at 2026-09-20T19:37:54.263205+00:00 confirmed all 98 values match. The review CSV remains pending.


## Cleanup — 2026-09-20

Removed nine redundant files: three completed import CSVs, three agent input batches, and three agent result files. All agent results were verified against the complete research before removal.

Retained for future work:

- [Missing ABV leaves](missing-abv-leaves.csv): 60 remaining leaves; the [all-node inventory](missing-abv.csv) also includes two parents. [Inventory manifest](manifest.json) records the latest retrieval.
- [Upload verification](abv-research-pass/import-verification.json): all 98 resolved values match production.
- [Research validation](abv-research-pass/validation.json) and [pre-import snapshot](abv-research-pass/production-snapshot.json): historical evidence.
- [Branch zero-value history](abv-zero-branches-manifest.json) and [Fruit zero-value history](abv-fruit-zero-manifest.json): rules, exclusions, and submitted rows preserved after removing completed CSVs.

Research statuses describe the original evidence assessment, not current import status. The 60-row review CSV is the active work queue.
