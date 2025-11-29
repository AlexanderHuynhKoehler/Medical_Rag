#!/usr/bin/env python3
"""
Comprehensive Mayo Clinic scraper for all common diseases across 7 categories.
Uses fixed parser to handle Mayo Clinic's div-based HTML structure.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import csv
from datetime import datetime
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "common_diseases"
METADATA_DIR = BASE_DIR / "metadata"
LOGS_DIR = BASE_DIR / "logs"
MASTER_LIST_PATH = METADATA_DIR / "disease_master_list.csv"
PROGRESS_PATH = METADATA_DIR / "scraping_progress.json"

# Anti-blocking settings
RATE_LIMIT_SECONDS = 3
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
TIMEOUT = 15

# Scraping statistics
stats = {
    "start_time": None,
    "end_time": None,
    "total_attempted": 0,
    "successful": 0,
    "failed": 0,
    "skipped": 0,
    "by_category": {}
}

def load_progress():
    """Load existing progress or create new"""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, 'r') as f:
            return json.load(f)
    return {"scraped_diseases": [], "failed_diseases": []}

def save_progress(progress):
    """Save progress to JSON file"""
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f, indent=2)

def parse_mayo_clinic_page(html_text):
    """
    Parse Mayo Clinic disease page and extract sections/content.
    FIXED VERSION: Uses find_next('div') to handle Mayo Clinic's HTML structure.
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    sections_data = []

    # Find all h2 headings as section markers
    headings = soup.find_all('h2')

    for heading in headings:
        section_title = heading.get_text().strip()

        # Skip noise sections
        noise_keywords = ['Products and Services', 'Book:', 'Request an Appointment',
                         'Find a doctor', 'Explore Mayo Clinic', 'Newsletter', 'Research', 'Education']
        if any(keyword.lower() in section_title.lower() for keyword in noise_keywords):
            continue

        # FIXED APPROACH: Content is in the next div after h2, not in siblings
        content_items = []
        next_div = heading.find_next('div')

        if next_div:
            # Extract paragraphs from the div
            for p in next_div.find_all('p'):
                text = p.get_text().strip()
                if text and len(text) > 20:
                    content_items.append(text)

            # Extract list items from the div
            for ul_ol in next_div.find_all(['ul', 'ol']):
                for li in ul_ol.find_all('li'):
                    text = li.get_text().strip()
                    if text and len(text) > 10:
                        content_items.append(text)

        if content_items:
            sections_data.append({
                "section": section_title,
                "content": content_items
            })

    return sections_data

def scrape_disease(disease_name, url, category):
    """Scrape a single disease from Mayo Clinic"""
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=TIMEOUT)

        if response.status_code == 503:
            print(f"  Service unavailable (503)")
            return None

        response.raise_for_status()

        # Parse the page
        sections_data = parse_mayo_clinic_page(response.text)

        if not sections_data:
            print(f"  No content extracted")
            return None

        # Create output structure with metadata
        output = {
            "metadata": {
                "disease_name": disease_name,
                "category": category,
                "scraped_date": datetime.now().isoformat(),
                "source_url": url,
                "sections_count": len(sections_data),
                "total_content_items": sum(len(s.get('content', [])) for s in sections_data)
            },
            "sections": sections_data
        }

        return output

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  404 Not Found")
        elif e.response.status_code == 429:
            print(f"  RATE LIMITED! Pausing for 60 seconds...")
            time.sleep(60)
        else:
            print(f"  HTTP Error {e.response.status_code}")
        return None
    except requests.exceptions.Timeout:
        print(f"  Timeout")
        return None
    except Exception as e:
        print(f"  Error: {str(e)}")
        return None

def scrape_category(category, diseases, progress, max_diseases=None):
    """Scrape all diseases in a category"""
    print(f"\n{'='*60}")
    print(f"CATEGORY: {category.upper()}")
    print(f"{'='*60}\n")

    category_stats = {"attempted": 0, "successful": 0, "failed": 0, "skipped": 0}
    category_dir = DATA_DIR / category

    # Filter diseases for this category
    category_diseases = [d for d in diseases if d['category'] == category]

    if max_diseases:
        category_diseases = category_diseases[:max_diseases]

    print(f"Total diseases to scrape: {len(category_diseases)}\n")

    for idx, disease in enumerate(category_diseases, 1):
        disease_name = disease['disease_name']
        url = disease['mayo_url']

        # Create safe filename
        safe_filename = disease_name.lower().replace(' ', '_').replace('/', '_')
        safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c == '_')
        output_path = category_dir / f"{safe_filename}.json"

        # Skip if already scraped
        if disease_name in progress['scraped_diseases']:
            print(f"[{idx}/{len(category_diseases)}] Skipping {disease_name} (already scraped)")
            category_stats['skipped'] += 1
            continue

        print(f"[{idx}/{len(category_diseases)}] Scraping: {disease_name}")
        category_stats['attempted'] += 1
        stats['total_attempted'] += 1

        # Scrape the disease
        data = scrape_disease(disease_name, url, category)

        if data:
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Update progress
            progress['scraped_diseases'].append(disease_name)
            save_progress(progress)

            # Update stats
            category_stats['successful'] += 1
            stats['successful'] += 1

            print(f"  Saved {data['metadata']['sections_count']} sections, "
                  f"{data['metadata']['total_content_items']} items")
        else:
            # Track failure
            progress['failed_diseases'].append(disease_name)
            save_progress(progress)
            category_stats['failed'] += 1
            stats['failed'] += 1

        # Rate limiting - wait 3 seconds before next request
        if idx < len(category_diseases):
            print(f"  Waiting {RATE_LIMIT_SECONDS} seconds...")
            time.sleep(RATE_LIMIT_SECONDS)

    # Category summary
    print(f"\n{'-'*60}")
    print(f"Category Summary: {category}")
    print(f"  Attempted: {category_stats['attempted']}")
    print(f"  Successful: {category_stats['successful']}")
    print(f"  Failed: {category_stats['failed']}")
    print(f"  Skipped: {category_stats['skipped']}")
    print(f"{'-'*60}\n")

    stats['by_category'][category] = category_stats

def load_master_list():
    """Load disease master list from CSV"""
    diseases = []
    with open(MASTER_LIST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            diseases.append(row)
    return diseases

def generate_report():
    """Generate final scraping report"""
    stats['end_time'] = datetime.now().isoformat()

    # Calculate duration
    if stats['start_time']:
        start = datetime.fromisoformat(stats['start_time'])
        end = datetime.fromisoformat(stats['end_time'])
        duration = end - start
        stats['duration_minutes'] = duration.total_seconds() / 60

    # Save stats to JSON
    stats_path = LOGS_DIR / f"scraping_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Generate text report
    report_path = LOGS_DIR / f"scraping_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MAYO CLINIC SCRAPING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Start Time: {stats['start_time']}\n")
        f.write(f"End Time: {stats['end_time']}\n")
        f.write(f"Duration: {stats.get('duration_minutes', 0):.2f} minutes\n\n")
        f.write(f"Total Attempted: {stats['total_attempted']}\n")
        f.write(f"Successful: {stats['successful']}\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"Success Rate: {(stats['successful']/stats['total_attempted']*100 if stats['total_attempted'] > 0 else 0):.1f}%\n\n")

        f.write("BY CATEGORY:\n")
        f.write("-"*60 + "\n")
        for category, cat_stats in stats['by_category'].items():
            f.write(f"\n{category.upper()}:\n")
            f.write(f"  Attempted: {cat_stats['attempted']}\n")
            f.write(f"  Successful: {cat_stats['successful']}\n")
            f.write(f"  Failed: {cat_stats['failed']}\n")
            f.write(f"  Skipped: {cat_stats['skipped']}\n")

    print(f"\nReports saved:")
    print(f"  - {stats_path}")
    print(f"  - {report_path}")

def retry_failed(progress, diseases):
    """Retry all failed diseases"""
    if not progress['failed_diseases']:
        return

    print(f"\n{'='*60}")
    print(f"RETRYING FAILED DISEASES")
    print(f"{'='*60}\n")

    # Get unique failed diseases
    failed_unique = list(set(progress['failed_diseases']))
    print(f"Total unique failed diseases: {len(failed_unique)}\n")

    # Create disease lookup
    disease_lookup = {d['disease_name']: d for d in diseases}

    still_failing = []

    for idx, disease_name in enumerate(sorted(failed_unique), 1):
        if disease_name not in disease_lookup:
            print(f"[{idx}/{len(failed_unique)}] Skipping {disease_name} - not in master list")
            continue

        disease_info = disease_lookup[disease_name]
        url = disease_info['mayo_url']
        category = disease_info['category']

        print(f"[{idx}/{len(failed_unique)}] Retrying: {disease_name}")
        print(f"  Category: {category}")

        safe_filename = disease_name.lower().replace(' ', '_').replace('/', '_')
        safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c == '_')
        output_path = DATA_DIR / category / f"{safe_filename}.json"

        data = scrape_disease(disease_name, url, category)

        if data:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Remove all occurrences from failed list
            progress['failed_diseases'] = [d for d in progress['failed_diseases'] if d != disease_name]

            if disease_name not in progress['scraped_diseases']:
                progress['scraped_diseases'].append(disease_name)

            save_progress(progress)

            print(f"  SUCCESS: Saved {data['metadata']['sections_count']} sections, "
                  f"{data['metadata']['total_content_items']} items")
        else:
            still_failing.append(disease_name)
            print(f"  FAILED")

        if idx < len(failed_unique):
            print(f"  Waiting {RATE_LIMIT_SECONDS} seconds...")
            time.sleep(RATE_LIMIT_SECONDS)

    print(f"\n{'-'*60}")
    print(f"Retry Summary:")
    print(f"  Attempted: {len(failed_unique)}")
    print(f"  Successful: {len(failed_unique) - len(still_failing)}")
    print(f"  Still failed: {len(still_failing)}")
    if still_failing:
        print(f"\n  Diseases still failing:")
        for disease in still_failing:
            print(f"    - {disease}")
    print(f"{'-'*60}\n")

def main():
    """Main scraping workflow"""
    print("\nMAYO CLINIC COMMON DISEASES SCRAPER - ALL 7 CATEGORIES")
    print("="*60)

    stats['start_time'] = datetime.now().isoformat()

    # Load master list
    print("\nLoading disease master list...")
    diseases = load_master_list()
    print(f"  Loaded {len(diseases)} diseases")

    # Load progress
    print("Loading progress tracker...")
    progress = load_progress()
    print(f"  Already scraped: {len(progress['scraped_diseases'])} diseases")
    print(f"  Previously failed: {len(progress['failed_diseases'])} diseases")

    # Define all 7 categories
    categories_to_scrape = [
        ('cardiovascular', 50),
        ('neurological', 50),
        ('gastrointestinal', 50),
        ('endocrine_metabolic', 50),
        ('respiratory_pulmonary', 50),
        ('kidney_renal', 50),
        ('musculoskeletal', 50)
    ]

    # Scrape all categories
    for category, max_count in categories_to_scrape:
        try:
            scrape_category(category, diseases, progress, max_diseases=max_count)
        except KeyboardInterrupt:
            print("\n\nScraping interrupted by user")
            break
        except Exception as e:
            print(f"\nError in category {category}: {e}")
            continue

    # Retry failed diseases
    if progress['failed_diseases']:
        retry_failed(progress, diseases)

    # Generate report
    print("\nGenerating final report...")
    generate_report()

    # Final summary
    print("\n" + "="*60)
    print("SCRAPING COMPLETE")
    print("="*60)
    print(f"Total Successful: {stats['successful']}")
    print(f"Total Failed: {stats['failed']}")
    if stats['total_attempted'] > 0:
        print(f"Success Rate: {(stats['successful']/stats['total_attempted']*100):.1f}%")

    # Show final progress status
    progress = load_progress()
    print(f"\nFinal Status:")
    print(f"  Total scraped: {len(progress['scraped_diseases'])}")
    print(f"  Remaining failures: {len(set(progress['failed_diseases']))}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
