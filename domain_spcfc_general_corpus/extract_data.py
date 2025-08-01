import re
import requests
from bs4 import BeautifulSoup
import time
import json

raw_text = """
/wiki/Etruscan_religion
/wiki/Etruscan_mythology
/wiki/List_of_Etruscan_mythological_figures
/wiki/List_of_Etruscan_names_for_Greek_heroes
/wiki/Etrusca_Disciplina
/wiki/Tages
/wiki/Vegoia
/wiki/Haruspices
/wiki/Haruspicy
/wiki/Divination
/wiki/Etruscan_priest
/wiki/Etruscan_priestess
/wiki/Priestesses
/wiki/Liver_of_Piacenza
/wiki/Votive_offering
/wiki/Lead_Plaque_of_Magliano
/wiki/Liber_Linteus
/wiki/Charun
/wiki/Aita_(mythology)
/wiki/Voltumna
/wiki/Tinia
/wiki/Uni_(mythology)
/wiki/Menrva
/wiki/Fufluns
/wiki/Laran
/wiki/Turms
/wiki/Maris_(mythology)
/wiki/Turan_(mythology)
/wiki/Cel_(goddess)
/wiki/Usil_(god)
/wiki/Hercle
/wiki/Catha_(mythology)
/wiki/Leinth
/wiki/Selvans
/wiki/Thalna
/wiki/Women_in_Etruscan_society
/wiki/Women_in_Etruscan_religion
/wiki/Etruscan_society
/wiki/Etruscan_civilization
/wiki/Etruscan_architecture#Temples
/wiki/Villanovan_culture
/wiki/Corpus_Speculorum_Etruscorum
/wiki/Corpus_Inscriptionum_Etruscarum
"""

# Extract links
links = re.findall(r'^/wiki/[^\s]+', raw_text, re.MULTILINE)
links = sorted(set(link.split('#')[0] for link in links))  # Remove #section

# Wikipedia base URL
base_url = "https://en.wikipedia.org"

# Prepare storage
data = {}

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; WikipediaScraper/1.0; +https://example.com/bot)"  # be polite
}

for link in links:
    url = base_url + link
    title = link.split("/")[-1].replace("_", " ")
    print(f"Scraping {url} ...")
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        continue

    soup = BeautifulSoup(resp.text, "html.parser")
    # Extract only main content (not navigation, not infobox)
    content = soup.find("div", id="mw-content-text")
    if not content:
        print(f"No content found for {title}")
        continue
    # Collect only first paragraphs (not lists, not tables)
    paragraphs = []
    for el in content.find_all(['p', 'ul'], recursive=False):
        # Skip empty paragraphs and navigation
        text = el.get_text(strip=True)
        if text and len(text) > 30:  # Ignore super short lines
            paragraphs.append(text)
        if len(paragraphs) >= 3:  # Limit to first 3 paragraphs
            break
    if not paragraphs:
        paragraphs = [content.get_text(separator="\n", strip=True)[:800]]
    data[title] = "\n\n".join(paragraphs)
    time.sleep(1.2)  # Be gentle with Wikipedia, avoid hammering

# Optionally: save to JSON for later
with open("wikipedia_etru_content.json", "w") as f:
    json.dump(data, f, indent=2)

print("Scraped", len(data), "Wikipedia articles.")

