# Ingredient measurement research — 2026-09-08

Retained research from two ABV passes. Neither pass changed the database. Snapshots and validation results are historical: refresh the live ingredient API before future imports or research.

## Import status and complete research

The user reported importing the 74 additional ready values and then adding the 30 review candidates to the database. Their export CSVs have been removed; all 104 original candidate values, sources, and notes remain in the complete second-pass research. This import status is user-reported, not independently checked against the live database. Historical research statuses remain unchanged.

The user also confirmed importing the original 84 ready values, bringing the reported total to 188 across both passes. The original `abv-import-ready.csv` has been removed after verifying that all 84 rows remain in the complete first-pass research.

- [Complete first-pass research](abv-research.csv): all 416 original ABV gaps, including source evidence, review items, unresolved entries, modeling defaults, and parents. Filter by status to recover the former shortlists.
- [Complete second-pass research](abv-leaf-round2-research.csv): 108 leaves that lacked ABV when researched, including four without numeric candidates. Prefer these findings over first-pass findings for the same IDs.

The second pass researched 109 rows and excluded Dolin Génépy (ID 81), which already had 35% ABV in the database. All 74 additional ready sources were independently opened and checked; both import CSVs passed the application parser. The original ready import excluded the 30 review candidates, which the user subsequently reported adding separately.

## Future work and provenance

- [Combined measurement inventory](missing-values.csv): all original measurement gaps, with leaf flags and ingredient metadata. Filter this instead of separate ABV/sugar/acidity queues. Empty cells mean missing; zero is an existing value.
- [Original snapshot](ingredients-snapshot.json) and [manifest](manifest.json): initial inventory, retrieval timestamp, and hash.
- [Final snapshot](leaf-final-snapshot.json): database state used for second-pass validation. Both snapshots omit creator identifiers.
- [First-pass source checks](coordinator-source-checks.json) and [second-pass source checks](leaf-round2-source-checks.json): retained verification evidence.
- [First-pass validation](validation.json) and [second-pass validation](leaf-round2-validation.json): historical results. First-pass validation includes hashes of intermediate files removed during cleanup; their rows remain in complete research.

The initial inventory contained 672 ingredients, including 552 leaves: 367 leaves lacked ABV, 461 lacked sugar, and 548 lacked titratable acidity. The final snapshot had 366 leaves missing ABV. Sugar and acidity research remains to be done.

## Cleanup — 2026-09-19

Removed 24 redundant batch files, subset exports, superseded summaries, and an identical intermediate snapshot. After the user reported completing the additional imports, removed three more redundant exports: `abv-leaf-additional-candidates.csv`, `abv-leaf-additional-import-ready.csv`, and `abv-leaf-review.csv`. All their rows were checked against the retained complete second-pass research before removal. Complete research preserves ingredient-level sources and unresolved work. Evidence rules and sugar-source leads below are retained from the original README.

## Evidence and update rules

- `ready`: a numeric ABV supported by an accessible product source and a sufficiently specific product match. Review the notes for the stated expression and market before using the candidate.
- `needs_review`: a plausible value or useful source exists, but product identity, formulation, market, edition, or source access prevents treating it as ready. Snippet-only evidence belongs here.
- `unresolved`: no usable numeric value established in this pass. Notes distinguish missing evidence from entries not yet searched.
- `modeling_default`: proposed nominal 0% for an ordinary food or nonalcoholic preparation. This is an explicit modeling assumption, not sourced analytical data; excluded from ready imports.
- `derived_parent`: no direct measurement proposed. Repository migration 14 calculates parent values from immediate-child averages, ignoring null values. Updating leaves can therefore also change ancestors. Parent values are category summaries, not bottle measurements.

The app's bulk-value CSV format is `ingredient_id,ingredient_name,field,value`, with at most 200 rows per request. Names must match the live database exactly. Its route rejects conflicts with existing nonmatching values. Source URLs and evidence are retained in the research files because the bulk measurement endpoint does not store them.

## Preparation-dependent cases

[Donn's Spices #2](https://kindredcocktails.com/ingredient/donns-spices-2) combines equal parts vanilla syrup and allspice dram. With additive volumes, the calculated ABV is their mean; the actual dram and syrup must be selected first. It must not receive a blanket zero.

[Nielsen-Massey's Tahitian vanilla extract](https://nielsenmassey.com/products/tahitian-pure-vanilla-extract-gallons-and-drums/) specifies 35% alcohol. This is a candidate example for generic Vanilla Extract, not proof that the unspecified database ingredient is 35%.

[Nielsen-Massey's Rose Water](https://nielsenmassey.com/products/rose-water/) lists cane alcohol. Generic floral waters need product identity before assigning zero. Brandied cherries and their syrups also require the actual preparation; packing-liquid ABV is not a measurement of drained fruit.

Sugar and titratable-acidity inventories are prepared for the next research pass. No values for those fields were proposed in this ABV pass.

## Sugar-source leads for the next pass

Useful source pages encountered during ABV verification:

- [Cointreau FAQ](https://www.cointreau.com/int/en/faq): a nutritional entry specifically for Cointreau Noir, including sugar and its serving volume (ingredient 349).
- [Ramazzotti Aperitivo Rosato](https://www.ramazzotti1815.com/en-us/products/aperitivo-rosato/): product nutrition table (ingredient 408).
- [Ramazzotti Amaro](https://www.ramazzotti1815.com/en-us/products/amaro/): product nutrition table (ingredient 500).
- [Bacardi corporate nutrition](https://www.bacardilimited.com/nutrition/bacardi/): expression-specific nutrition entries, including Reserva Ocho (ingredient 260).

These are leads only; no sugar or acidity values were added to the import.
