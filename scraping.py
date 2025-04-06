import requests
from bs4 import BeautifulSoup
import time
import csv

BASE_URL = "https://www.shl.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_soup(url):
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def parse_main_row(row):
    name_tag = row.select_one("td.custom__table-heading__title a")
    name = name_tag.text.strip()
    detail_url = BASE_URL + name_tag['href']

    remote_support = "Yes" if row.select("td")[1].find(class_="-yes") else "No"
    adaptive_support = "Yes" if row.select("td")[2].find(class_="-yes") else "No"

    irt_support = "No"
    keys = row.select("td")[3].select("span.product-catalogue__key")
    for key in keys:
        if key.text.strip() == "A":
            irt_support = "Yes"
            break

    return {
        "Assessment Name": name,
        "URL": detail_url,
        "Remote Testing Support": remote_support,
        "Adaptive Support": adaptive_support,
        "IRT Support": irt_support
    }

def parse_detail_page(url):
    try:
        soup = get_soup(url)

        def get_field(label):
            tag = soup.find("h4", string=label)
            if tag:
                next_p = tag.find_next_sibling("p")
                return next_p.text.strip() if next_p else "N/A"
            return "N/A"

        def get_test_type():
            keys = soup.select("span.product-catalogue__key")
            return ", ".join([k.text.strip() for k in keys]) if keys else "N/A"

        duration = get_field("Assessment length")
        if "Approximate Completion Time in minutes = " in duration:
            duration = duration.replace("Approximate Completion Time in minutes = ", "")

        return {
            "Test Type(s)": get_test_type(),
            "Duration": duration,
            "Description": get_field("Description"),
            "Job Levels": get_field("Job levels"),
            "Languages": get_field("Languages")
        }

    except Exception as e:
        print(f"❌ Error parsing detail page: {url} -> {e}")
        return {
            "Test Type(s)": "N/A",
            "Duration": "N/A",
            "Description": "N/A",
            "Job Levels": "N/A",
            "Languages": "N/A"
        }

def scrape_category():
    start = 0
    results = []

    while True:
        list_url = f"{BASE_URL}/solutions/products/product-catalog/?start={start}&type=1&type=1"
        print(f"\n🔎 Scraping list page: {list_url}")
        soup = get_soup(list_url)

        rows = soup.select("tr[data-entity-id]")
        if not rows:
            print("✅ No more rows found.")
            break

        for row in rows:
            row_data = parse_main_row(row)
            print(f"   ↪ Visiting detail page: {row_data['URL']}")
            detail_data = parse_detail_page(row_data["URL"])
            row_data.update(detail_data)
            results.append(row_data)
            time.sleep(1)  # Be polite to server

        start += 12  # Move to next page

    return results

if __name__ == "__main__":
    print(f"\n🚀 Starting scrape for Type 1 assessments only")
    data = scrape_category()

    if data:
        keys = data[0].keys()
        with open("shl_data_type1.csv", "w", newline="" , encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

        print(f"\n✅ Done! {len(data)} Type 1 records saved to 'shl_data_type1.csv'")
    else:
        print("⚠️ No data was scraped.")
