# Mayo Clinic Disease Scraping Scripts

## Overview

This folder contains scripts to scrape common diseases from Mayo Clinic across 7 medical categories.

## Main Script

### scrape_common_diseases.py

**Comprehensive all-in-one scraper for all 7 categories.**

Features:
- Scrapes all diseases from all 7 medical categories
- Uses fixed parser for Mayo Clinic's div-based HTML structure
- Automatic retry of failed diseases
- Progress tracking and resume capability
- Rate limiting (3 seconds between requests)
- Detailed logging and reporting

Categories covered:
1. Cardiovascular (20 diseases)
2. Neurological (20 diseases)
3. Gastrointestinal (19 diseases)
4. Endocrine/Metabolic (17 diseases)
5. Respiratory/Pulmonary (15 diseases)
6. Kidney/Renal (12 diseases)
7. Musculoskeletal (18 diseases)

Usage:
```bash
python scrape_all_diseases.py
```

The script will:
1. Load existing progress (if any)
2. Skip already scraped diseases
3. Scrape all 7 categories
4. Automatically retry any failures
5. Generate comprehensive reports

## Legacy Scripts

### scrape_common_diseases.py
Original scraping script (modified for categories 4-7).

### retry_failed_diseases.py
Retry script for initial 7 failures from categories 1-3.

### retry_remaining_failed.py
Retry script for 14 failures from categories 4-7.

## Output Structure

```
common_diseases/
├── cardiovascular/          (20 JSON files)
├── neurological/            (20 JSON files)
├── gastrointestinal/        (19 JSON files)
├── endocrine_metabolic/     (17 JSON files)
├── respiratory_pulmonary/   (15 JSON files)
├── kidney_renal/            (12 JSON files)
└── musculoskeletal/         (18 JSON files)

metadata/
├── disease_master_list.csv    (All disease URLs)
└── scraping_progress.json     (Progress tracker)

logs/
├── scraping_report_*.txt      (Text reports)
└── scraping_stats_*.json      (JSON statistics)
```

## JSON Format

Each disease JSON file contains:
```json
{
  "metadata": {
    "disease_name": "Disease Name",
    "category": "category_name",
    "scraped_date": "ISO timestamp",
    "source_url": "Mayo Clinic URL",
    "sections_count": 7,
    "total_content_items": 49
  },
  "sections": [
    {
      "section": "Overview",
      "content": ["paragraph 1", "paragraph 2", ...]
    }
  ]
}
```

## Current Status

**Total: 121 diseases scraped (100% success rate)**

- Cardiovascular: 20 diseases
- Neurological: 20 diseases
- Gastrointestinal: 19 diseases
- Endocrine/Metabolic: 17 diseases
- Respiratory/Pulmonary: 15 diseases
- Kidney/Renal: 12 diseases
- Musculoskeletal: 18 diseases

## Key Features

### Fixed Parser
The parser uses `find_next('div')` instead of `find_next_sibling()` to correctly handle Mayo Clinic's HTML structure where content is nested in divs.

### Progress Tracking
The script tracks progress in `metadata/scraping_progress.json` and can resume from where it left off if interrupted.

### Rate Limiting
3-second delay between requests to avoid being blocked by Mayo Clinic.

### Automatic Retry
Failed diseases are automatically retried at the end of the scraping session.

## Disease Selection Criteria

Diseases were selected based on:
1. **Global prevalence** (WHO/CDC statistics)
2. **Medical impact** (mortality, morbidity, disability)
3. **Patient relevance** (commonly searched conditions)

## Documentation

- `FINAL_SCRAPING_REPORT.md` - Comprehensive report of all 121 diseases
- `SCRAPING_SUMMARY_3_CATEGORIES.md` - Report after initial 3 categories
- This README

## Notes

- All scripts avoid using emojis as requested
- Rate limiting prevents Mayo Clinic from blocking requests
- Progress is saved after each disease to prevent data loss
- The master list contains 118 diseases, 121 were successfully scraped
