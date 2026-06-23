# Run the Crane CRM enrichment pipeline

This uses heuristic/fallback enrichment and skips local vision analysis. Quality will be lower than with the LLM and vision model.

## Clean website filtering

This version adds a strict official-site verification layer.

The CRM field `company_website_url` is now populated only when the URL passes official-site checks. Company profiles, social pages, marketplaces, directory listings, generic manufacturer pages, parked domains, and dead/default pages are rejected and stored in audit columns instead:

- `official_website_confidence`
- `site_status`
- `site_rejection_reason`
- `profile_urls`
- `rejected_urls`
- `official_site_debug`

Recommended clean-data config:

```env
OFFICIAL_SITE_REQUIRED=true
SITE_MIN_OFFICIAL_SCORE=60
ALLOW_PROFILE_AS_VERIFIED_URL=false
```

For very strict CRM output, raise the threshold:

```env
SITE_MIN_OFFICIAL_SCORE=70
```

To audit an existing enriched CSV without rerunning the full enrichment:

```bash
python -m src.audit_website_quality \
  --input data/output/enriched_companies.csv \
  --output data/output/website_quality_audit.csv \
  --live
```

To rerun rows that were previously fallback/unclear/profile-polluted, use a clean output file or run with `--no-resume`. For fallback-only reruns:

```bash
python -m src.enrich_pipeline \
  --input "data/input/your_file.xlsx" \
  --output "data/output/enriched_companies.csv" \
  --rerun-fallbacks \
  --log-level DEBUG
```
