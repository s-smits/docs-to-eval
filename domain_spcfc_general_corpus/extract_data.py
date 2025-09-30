"""
Extract Wikipedia articles about Etruscan civilization and related topics
"""

import re
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

# Wikipedia API endpoint
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

# List of Etruscan-related Wikipedia pages to extract
ETRUSCAN_PAGES = """
/wiki/Etruscan_language
/wiki/Etruscan_alphabet
/wiki/Etruscan_religion
/wiki/Etruscan_mythology
/wiki/Etruscan_art
/wiki/Pyrgi_Tablets
/wiki/Tabula_Cortonensis
/wiki/Cippus_Perusinus
/wiki/Liber_Linteus
/wiki/Lead_Plaque_of_Magliano
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


class WikipediaExtractor:
    """Extract and process Wikipedia articles"""
    
    def __init__(self, output_dir: str = ".", delay: float = 1.0):
        """
        Initialize the extractor
        
        Args:
            output_dir: Directory to save extracted data
            delay: Delay between requests to be respectful to Wikipedia's servers
        """
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DocsToEval Educational Corpus Builder/1.0 (Research Project)'
        })
    
    def extract_page_content(self, page_title: str) -> Optional[Dict[str, str]]:
        """
        Extract content from a Wikipedia page using the MediaWiki API
        
        Args:
            page_title: Title of the Wikipedia page (without /wiki/ prefix)
            
        Returns:
            Dictionary with title, content, and metadata
        """
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': page_title,
                'prop': 'extracts|info',
                'explaintext': True,  # Get plain text instead of HTML
                'exsectionformat': 'plain',
                'inprop': 'url',
            }
            
            response = self.session.get(WIKI_API_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            # Get the first (and should be only) page
            page_data = next(iter(pages.values()))
            
            if 'missing' in page_data:
                print(f"⚠️  Page not found: {page_title}")
                return None
            
            return {
                'title': page_data.get('title', page_title),
                'content': page_data.get('extract', ''),
                'url': page_data.get('fullurl', ''),
                'pageid': page_data.get('pageid', 0)
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching {page_title}: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error for {page_title}: {e}")
            return None
    
    def parse_page_list(self, page_list_text: str) -> List[str]:
        """Parse the page list and extract page titles"""
        links = re.findall(r'^/wiki/([^\s#]+)', page_list_text, re.MULTILINE)
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        return unique_links
    
    def extract_all_pages(self, page_list_text: str) -> Dict[str, Dict]:
        """
        Extract all pages from the list
        
        Args:
            page_list_text: Raw text containing /wiki/ links
            
        Returns:
            Dictionary mapping page titles to their content
        """
        page_titles = self.parse_page_list(page_list_text)
        results = {}
        
        print(f"📚 Extracting {len(page_titles)} Wikipedia pages...")
        print("=" * 60)
        
        for i, title in enumerate(page_titles, 1):
            print(f"[{i}/{len(page_titles)}] Fetching: {title}")
            
            content = self.extract_page_content(title)
            if content:
                results[title] = content
                print(f"  ✅ Success ({len(content['content'])} characters)")
            else:
                print(f"  ⚠️  Failed")
            
            # Be respectful to Wikipedia's servers
            if i < len(page_titles):
                time.sleep(self.delay)
        
        print("=" * 60)
        print(f"✅ Extracted {len(results)}/{len(page_titles)} pages successfully")
        
        return results
    
    def save_as_json(self, data: Dict[str, Dict], filename: str = "wikipedia_etru_content.json"):
        """Save extracted data as JSON"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved JSON: {output_path}")
        return output_path
    
    def save_as_text_files(self, data: Dict[str, Dict], subdir: str = "etruscan_texts"):
        """Save each page as a separate text file"""
        text_dir = self.output_dir / subdir
        text_dir.mkdir(exist_ok=True)
        
        for title, page_data in data.items():
            # Create safe filename
            safe_filename = re.sub(r'[^\w\s-]', '', title.replace('/', '_'))
            safe_filename = re.sub(r'\s+', '_', safe_filename).lower()
            
            filepath = text_dir / f"{safe_filename}.txt"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {page_data['title']}\n\n")
                f.write(f"Source: {page_data['url']}\n\n")
                f.write("=" * 70 + "\n\n")
                f.write(page_data['content'])
            
        print(f"💾 Saved {len(data)} text files to: {text_dir}")
        return text_dir
    
    def get_corpus_stats(self, data: Dict[str, Dict]) -> Dict[str, any]:
        """Calculate statistics about the extracted corpus"""
        total_chars = sum(len(p['content']) for p in data.values())
        total_words = sum(len(p['content'].split()) for p in data.values())
        
        stats = {
            'total_pages': len(data),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_chars_per_page': total_chars // len(data) if data else 0,
            'avg_words_per_page': total_words // len(data) if data else 0,
            'titles': [p['title'] for p in data.values()]
        }
        
        return stats


def main():
    """Main extraction function"""
    print("🏺 Etruscan Wikipedia Corpus Extractor")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Initialize extractor
    extractor = WikipediaExtractor(
        output_dir=script_dir,
        delay=1.0  # 1 second between requests
    )
    
    # Extract all pages
    data = extractor.extract_all_pages(ETRUSCAN_PAGES)
    
    if not data:
        print("❌ No data extracted. Exiting.")
        return
    
    # Save as JSON
    extractor.save_as_json(data)
    
    # Save as individual text files
    extractor.save_as_text_files(data)
    
    # Print statistics
    stats = extractor.get_corpus_stats(data)
    print("\n📊 Corpus Statistics:")
    print("=" * 60)
    print(f"Total Pages:    {stats['total_pages']}")
    print(f"Total Words:    {stats['total_words']:,}")
    print(f"Total Chars:    {stats['total_characters']:,}")
    print(f"Avg Words/Page: {stats['avg_words_per_page']:,}")
    print(f"Avg Chars/Page: {stats['avg_chars_per_page']:,}")
    print("=" * 60)
    
    print("\n✅ Extraction complete!")
    print(f"📁 JSON file: {script_dir / 'wikipedia_etru_content.json'}")
    print(f"📁 Text files: {script_dir / 'etruscan_texts'}/")


if __name__ == "__main__":
    main()