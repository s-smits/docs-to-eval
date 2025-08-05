#!/usr/bin/env python3
"""
Enhanced Wikipedia Content Scraper for Etruscan Domain Corpus

This script scrapes Wikipedia articles about Etruscan topics with improved
error handling, progress tracking, and configuration options.
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment
from rich.console import Console
from rich.progress import Progress, TaskID, track
from rich.logging import RichHandler

# Configuration
@dataclass
class ScrapingConfig:
    """Configuration for the Wikipedia scraper."""
    base_url: str = "https://en.wikipedia.org"
    output_file: str = "wikipedia_etru_content.json"
    backup_file: str = "wikipedia_etru_content.backup.json"
    cache_file: str = "scraping_cache.json"
    text_output_dir: str = "etruscan_texts"  # Directory for individual text files
    
    # Rate limiting
    delay_between_requests: float = 1.2
    timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Content extraction - REMOVE LIMITS TO GET FULL ARTICLES
    max_paragraphs: int = None  # No limit - get ALL content
    min_paragraph_length: int = 10  # Lower threshold for more content
    max_content_length: int = None  # No limit - get FULL article
    
    # Request headers
    user_agent: str = "Mozilla/5.0 (compatible; EtruscanCorpusBuilder/2.0; +https://example.com/bot)"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "scraper.log"

class WikipediaScraper:
    """Enhanced Wikipedia scraper with robust error handling and progress tracking."""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.console = Console()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        
        # Setup logging
        self._setup_logging()
        
        # Load cache and existing data
        self.cache = self._load_cache()
        self.existing_data = self._load_existing_data()
        
        # Track different types of failures
        self.not_found_urls = set()
        self.failed_urls = set()
        
        # Setup text output directory
        self._setup_text_output_dir()
        
        self.logger.info(f"Initialized scraper with {len(self.existing_data)} existing articles")
        self.logger.info(f"Cache contains {len(self.cache)} entries")

    def _setup_logging(self):
        """Setup logging with both file and console handlers."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with rich formatting
        console_handler = RichHandler(console=self.console, show_path=False)
        console_handler.setLevel(logging.INFO)
        
        # File handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.addHandler(console_handler)

    def _setup_text_output_dir(self):
        """Setup the text output directory and clean old files."""
        self.text_output_path = Path(self.config.text_output_dir)
        self.text_output_path.mkdir(exist_ok=True)
        
        # Clean up old .txt files (like convert_json_to_txt.py does)
        old_files = list(self.text_output_path.glob("*.txt"))
        if old_files:
            self.logger.info(f"Cleaning up {len(old_files)} old text files")
            for file_path in old_files:
                try:
                    file_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {e}")

    def _clean_filename(self, title: str) -> str:
        """Clean title to create a valid filename (matches convert_json_to_txt.py)."""
        # Remove special characters and replace spaces with underscores
        cleaned = re.sub(r'[^\w\s-]', '', title)
        cleaned = re.sub(r'\s+', '_', cleaned).lower()
        return cleaned

    def _save_text_file(self, title: str, content: str):
        """Save content as individual text file."""
        try:
            filename = self._clean_filename(title) + '.txt'
            file_path = self.text_output_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n{content.strip()}")
            
            self.logger.debug(f"Saved text file: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save text file for '{title}': {e}")

    def _load_cache(self) -> Dict:
        """Load scraping cache to avoid re-scraping failed URLs."""
        if os.path.exists(self.config.cache_file):
            try:
                with open(self.config.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    # Convert lists back to sets for cache entries
                    if "failed_urls" in cache:
                        cache["failed_urls"] = set(cache["failed_urls"])
                    if "successful_urls" in cache:
                        cache["successful_urls"] = set(cache["successful_urls"])
                    if "not_found_urls" in cache:
                        cache["not_found_urls"] = set(cache["not_found_urls"])
                    return cache
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return {"failed_urls": set(), "successful_urls": set(), "not_found_urls": set()}

    def _save_cache(self):
        """Save scraping cache."""
        try:
            # Convert sets to lists for JSON serialization
            cache_data = {
                "failed_urls": list(self.cache.get("failed_urls", set())),
                "successful_urls": list(self.cache.get("successful_urls", set())),
                "not_found_urls": list(self.cache.get("not_found_urls", set()))
            }
            with open(self.config.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def _load_existing_data(self) -> Dict:
        """Load existing scraped data."""
        if os.path.exists(self.config.output_file):
            try:
                with open(self.config.output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load existing data: {e}")
        return {}

    def _create_backup(self):
        """Create backup of existing data."""
        if os.path.exists(self.config.output_file):
            try:
                import shutil
                shutil.copy2(self.config.output_file, self.config.backup_file)
                self.logger.info(f"Backup created: {self.config.backup_file}")
            except Exception as e:
                self.logger.error(f"Failed to create backup: {e}")

    def extract_links_from_text(self, raw_text: str) -> List[str]:
        """Extract and clean Wikipedia links from raw text."""
        # Extract links
        links = re.findall(r'^/wiki/[^\s]+', raw_text, re.MULTILINE)
        
        # Clean and deduplicate
        cleaned_links = []
        seen = set()
        
        for link in links:
            # Remove #section anchors
            clean_link = link.split('#')[0]
            if clean_link not in seen:
                seen.add(clean_link)
                cleaned_links.append(clean_link)
        
        return sorted(cleaned_links)

    def extract_content(self, html: str, title: str) -> Optional[str]:
        """Extract FULL content from Wikipedia article HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find main content
            content_div = soup.find('div', id='mw-content-text')
            if not content_div:
                self.logger.warning(f"No main content found for {title}")
                return None
            
            # Target the actual parser output - this contains the full article
            parser_output = content_div.find('div', class_='mw-parser-output')
            if parser_output:
                content_div = parser_output
                self.logger.debug(f"Using mw-parser-output for {title}")
            
            # Clean unwanted elements BEFORE extraction
            content_div = self._clean_content(content_div)
            
            # Extract ALL content elements
            content_elements = []
            element_count = 0
            
            # Get ALL content in document order - no limits!
            for element in content_div.find_all([
                'p',           # Paragraphs
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',  # Headers
                'ul', 'ol',    # Lists
                'dl',          # Definition lists
                'blockquote',  # Quotes
                'pre',         # Preformatted text
                'div'          # Content divs (filtered)
            ]):
                
                # Skip nested elements to avoid duplication
                if element.find_parent(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'dl', 'blockquote', 'pre']):
                    continue
                
                # Skip unwanted div classes
                if element.name == 'div':
                    div_classes = element.get('class', [])
                    skip_div_classes = [
                        'navbox', 'infobox', 'thumbinner', 'thumbcaption',
                        'toc', 'references', 'reflist', 'authority-control',
                        'navbox-container', 'metadata', 'hatnote', 'catlinks',
                        'printfooter', 'mw-authority-control'
                    ]
                    if any(cls in str(div_classes) for cls in skip_div_classes):
                        continue
                
                text = element.get_text(strip=True)
                
                # Include all content with minimal filtering
                if text and len(text) >= self.config.min_paragraph_length:
                    # Skip only the most obvious navigation content
                    skip_phrases = [
                        'coordinates:', 'jump to:', 'navigation menu',
                        'edit]', '[edit]', 'citation needed]',
                        'from wikipedia, the free encyclopedia',
                        'redirected from', 'article talk',
                        'tools appearance hide'
                    ]
                    
                    if not any(phrase in text.lower() for phrase in skip_phrases):
                        # Format headers with markdown-style markers
                        if element.name.startswith('h'):
                            level = int(element.name[1])
                            header_marker = '#' * level
                            text = f"\n{header_marker} {text}\n"
                        
                        content_elements.append(text)
                        element_count += 1
                
                # NO LIMIT - Keep going until end of article!
                # Remove the max_paragraphs check entirely
            
            self.logger.info(f"Extracted {element_count} elements from {title}")
            
            # Join all content
            if content_elements:
                content = '\n\n'.join(content_elements)
                
                # NO TRUNCATION - return full content
                self.logger.info(f"Extracted {len(content)} characters from {title}")
                return content
            else:
                # Fallback to get ALL text if structured extraction failed
                self.logger.debug(f"Using fallback extraction for {title}")
                
                if parser_output:
                    full_text = parser_output.get_text(separator='\n', strip=True)
                else:
                    full_text = content_div.get_text(separator='\n', strip=True)
                
                # Minimal cleanup
                lines = []
                for line in full_text.split('\n'):
                    line = line.strip()
                    if line and len(line) > 5:
                        lines.append(line)
                
                if lines:
                    return '\n\n'.join(lines)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to extract content for {title}: {e}")
            return None

    def _clean_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove unwanted elements from the soup - enhanced version."""
        # Remove elements that don't contain useful article content
        unwanted_selectors = [
            # Navigation and UI elements
            'div.navbox', 'div.navbox-container', 'table.navbox',
            'div.infobox', 'table.infobox',
            'div.hatnote', 'div.dablink',
            'div.metadata', 'div.noprint',
            'div.sister-project', 'div.sister-projects',
            
            # Media and layout elements  
            'div.thumb', 'div.tright', 'div.tleft',
            'div.thumbinner', 'div.thumbcaption',
            'figure', 'figcaption',
            
            # References and citations
            'sup.reference', 'span.reference',
            'ol.references', 'div.reflist',
            'div.mw-references-wrap',
            
            # Edit and admin elements
            'span.mw-editsection', 'span.editsection',
            'div.printfooter', 'div.catlinks',
            
            # Table of contents
            'div#toc', 'div.toc',
            
            # Authority control and similar
            'div.authority-control',
            'div.asbox', 'table.asbox',
            
            # Coordinate display
            'span#coordinates', 'div.coordinates'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        return soup

    def fetch_article(self, link: str) -> Optional[str]:
        """Fetch a single Wikipedia article with retry logic."""
        url = urljoin(self.config.base_url, link)
        title = link.split('/')[-1].replace('_', ' ')
        
        # Check caches
        if url in self.cache.get("not_found_urls", set()):
            self.logger.debug(f"Skipping {url} (404 - page not found)")
            return None
        
        if url in self.cache.get("failed_urls", set()):
            self.logger.debug(f"Skipping {url} (in failed cache)")
            return None
        
        if title in self.existing_data:
            self.logger.debug(f"Skipping {title} (already exists)")
            return self.existing_data[title]
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Fetching {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=self.config.timeout)
                
                # Handle 404 errors separately
                if response.status_code == 404:
                    self.logger.warning(f"Page not found (404): {url}")
                    self.cache.setdefault("not_found_urls", set()).add(url)
                    self.not_found_urls.add(url)
                    return None
                
                response.raise_for_status()
                
                # Extract content
                content = self.extract_content(response.text, title)
                
                if content:
                    self.cache.setdefault("successful_urls", set()).add(url)
                    self.logger.info(f"Successfully scraped: {title}")
                    return content
                else:
                    self.logger.warning(f"No content extracted for: {title}")
                    return None
                    
            except requests.exceptions.HTTPError as e:
                # Handle non-404 HTTP errors
                if response.status_code != 404:
                    self.logger.warning(f"HTTP error for {url} (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        self.cache.setdefault("failed_urls", set()).add(url)
                        self.failed_urls.add(url)
                        self.logger.error(f"Failed to fetch {url} after {self.config.max_retries} attempts")
                        
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {url} (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.cache.setdefault("failed_urls", set()).add(url)
                    self.failed_urls.add(url)
                    self.logger.error(f"Failed to fetch {url} after {self.config.max_retries} attempts")
            
            except Exception as e:
                self.logger.error(f"Unexpected error for {url}: {e}")
                self.cache.setdefault("failed_urls", set()).add(url)
                self.failed_urls.add(url)
                break
        
        return None

    def scrape_articles(self, links: List[str]) -> Dict[str, str]:
        """Scrape multiple Wikipedia articles with progress tracking."""
        self.logger.info(f"Starting to scrape {len(links)} articles")
        
        # Create backup
        self._create_backup()
        
        # Start with existing data
        results = self.existing_data.copy()
        
        # Filter links that need scraping
        links_to_scrape = []
        for link in links:
            title = link.split('/')[-1].replace('_', ' ')
            url = urljoin(self.config.base_url, link)
            
            # Skip if already scraped or in not_found cache
            if title not in results and url not in self.cache.get("not_found_urls", set()):
                links_to_scrape.append(link)
        
        if not links_to_scrape:
            self.logger.info("All articles already scraped or marked as not found!")
            # Still save existing data as text files
            self._save_all_text_files(results)
            return results
        
        self.logger.info(f"Need to scrape {len(links_to_scrape)} new articles")
        
        # Progress tracking
        with Progress(console=self.console) as progress:
            task = progress.add_task("Scraping articles...", total=len(links_to_scrape))
            
            for link in links_to_scrape:
                title = link.split('/')[-1].replace('_', ' ')
                
                # Update progress description
                progress.update(task, description=f"Scraping: {title[:30]}...")
                
                # Fetch article
                content = self.fetch_article(link)
                
                if content:
                    results[title] = content
                    # Save individual text file immediately
                    self._save_text_file(title, content)
                    
                    # Save progress periodically
                    if len(results) % 5 == 0:
                        self._save_data(results)
                        self._save_cache()
                
                # Rate limiting
                time.sleep(self.config.delay_between_requests)
                
                # Update progress
                progress.advance(task)
        
        # Final save
        self._save_data(results)
        self._save_cache()
        
        # Save any remaining text files for existing data
        self._save_all_text_files(results)
        
        # Report any failures
        if self.not_found_urls:
            self.console.print(f"\n[yellow]⚠️  {len(self.not_found_urls)} pages were not found (404):[/yellow]")
            for url in list(self.not_found_urls)[:5]:
                self.console.print(f"  - {url}")
            if len(self.not_found_urls) > 5:
                self.console.print(f"  ... and {len(self.not_found_urls) - 5} more")
        
        if self.failed_urls:
            self.console.print(f"\n[red]❌ {len(self.failed_urls)} pages failed to scrape:[/red]")
            for url in list(self.failed_urls)[:5]:
                self.console.print(f"  - {url}")
            if len(self.failed_urls) > 5:
                self.console.print(f"  ... and {len(self.failed_urls) - 5} more")
        
        return results

    def _save_all_text_files(self, data: Dict[str, str]):
        """Save all articles as individual text files."""
        if not data:
            return
            
        self.logger.info(f"Saving {len(data)} articles as individual text files...")
        
        for title, content in data.items():
            if content and content.strip():
                self._save_text_file(title, content)
        
        self.logger.info(f"Saved {len(data)} text files to {self.text_output_path}")

    def _save_data(self, data: Dict[str, str]):
        """Save scraped data to JSON file."""
        try:
            with open(self.config.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved {len(data)} articles to {self.config.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")

    def get_statistics(self, data: Dict[str, str]) -> Dict:
        """Generate statistics about the scraped data."""
        if not data:
            return {}
        
        word_counts = [len(content.split()) for content in data.values()]
        char_counts = [len(content) for content in data.values()]
        
        return {
            "total_articles": len(data),
            "total_words": sum(word_counts),
            "total_characters": sum(char_counts),
            "avg_words_per_article": sum(word_counts) / len(word_counts) if word_counts else 0,
            "avg_chars_per_article": sum(char_counts) / len(char_counts) if char_counts else 0,
            "shortest_article": min(word_counts) if word_counts else 0,
            "longest_article": max(word_counts) if word_counts else 0,
        }

def load_default_links() -> str:
    """Load the default list of Etruscan-related Wikipedia links."""
    # Note: Removed /wiki/Etrusca_Disciplina as it doesn't exist as a standalone page
    # The concept is covered in /wiki/Etruscan_religion
    return """
/wiki/Etruscan_religion
/wiki/Etruscan_mythology
/wiki/List_of_Etruscan_mythological_figures
/wiki/List_of_Etruscan_names_for_Greek_heroes
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
/wiki/Tabula_Capuana
/wiki/Daily_life_of_the_Etruscans
/wiki/Etruscan_military_history
"""

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced Wikipedia scraper for Etruscan domain corpus"
    )
    parser.add_argument(
        "--links-file", 
        type=str, 
        help="File containing Wikipedia links to scrape (one per line)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="wikipedia_etru_content.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--text-dir", 
        type=str, 
        default="etruscan_texts",
        help="Directory for individual text files"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=1.2,
        help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--max-paragraphs", 
        type=int, 
        default=None,  # No limit by default
        help="Maximum paragraphs to extract per article (None for no limit)"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=None,  # No limit by default
        help="Maximum content length in characters (None for no limit)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=15,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--stats-only", 
        action="store_true",
        help="Only show statistics for existing data"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ScrapingConfig(
        output_file=args.output,
        text_output_dir=args.text_dir,
        delay_between_requests=args.delay,
        max_paragraphs=args.max_paragraphs,
        max_content_length=args.max_length,
        timeout=args.timeout,
        log_level="DEBUG" if args.verbose else "INFO"
    )
    
    # Initialize scraper
    scraper = WikipediaScraper(config)
    
    # Show stats only if requested
    if args.stats_only:
        existing_data = scraper._load_existing_data()
        if existing_data:
            stats = scraper.get_statistics(existing_data)
            scraper.console.print("\n[bold blue]📊 Corpus Statistics[/bold blue]")
            for key, value in stats.items():
                if isinstance(value, float):
                    scraper.console.print(f"  {key.replace('_', ' ').title()}: {value:.1f}")
                else:
                    scraper.console.print(f"  {key.replace('_', ' ').title()}: {value:,}")
        else:
            scraper.console.print("[red]No existing data found[/red]")
        return
    
    # Load links
    if args.links_file and os.path.exists(args.links_file):
        with open(args.links_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    else:
        raw_text = load_default_links()
    
    # Extract links
    links = scraper.extract_links_from_text(raw_text)
    scraper.logger.info(f"Found {len(links)} unique Wikipedia links")
    
    # Scrape articles
    try:
        results = scraper.scrape_articles(links)
        
        # Show final statistics
        stats = scraper.get_statistics(results)
        scraper.console.print("\n[bold green]✅ Scraping Complete![/bold green]")
        scraper.console.print("\n[bold blue]📊 Final Statistics[/bold blue]")
        for key, value in stats.items():
            if isinstance(value, float):
                scraper.console.print(f"  {key.replace('_', ' ').title()}: {value:.1f}")
            else:
                scraper.console.print(f"  {key.replace('_', ' ').title()}: {value:,}")
        
        scraper.console.print(f"\n[bold]Outputs:[/bold]")
        scraper.console.print(f"  JSON: {config.output_file}")
        scraper.console.print(f"  Text files: {config.text_output_dir}/")
        
    except KeyboardInterrupt:
        scraper.logger.info("Scraping interrupted by user")
        scraper.console.print("\n[yellow]⚠️  Scraping interrupted. Progress has been saved.[/yellow]")
    except Exception as e:
        scraper.logger.error(f"Scraping failed: {e}")
        scraper.console.print(f"\n[red]❌ Scraping failed: {e}[/red]")

if __name__ == "__main__":
    main()

