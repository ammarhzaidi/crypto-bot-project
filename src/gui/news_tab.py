"""
News tab module for the Crypto Trading Bot GUI.
Fetches and displays crypto news with sentiment analysis.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from datetime import datetime
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import traceback
from urllib.parse import urlparse
import webbrowser
import json
import requests
from datetime import datetime, timedelta

class NewsTab:
    """
    Tab for fetching and displaying crypto news with sentiment analysis.
    """

    def analyze_crypto_sentiment(self, text):
        """
        Enhanced sentiment analysis for crypto news that considers crypto-specific terms.

        Args:
            text: The text to analyze

        Returns:
            A tuple of (sentiment_label, compound_score)
        """
        # Initialize the standard sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()

        # Get base sentiment scores
        sentiment_scores = analyzer.polarity_scores(text)
        compound = sentiment_scores["compound"]

        # Crypto-specific positive terms that might not be captured well by VADER
        crypto_positive = [
            "adoption", "reserve", "legitimize", "legitimizing", "strategic",
            "reshape", "reshaping", "institutional", "mainstream", "bullish",
            "rally", "support", "hodl", "integration", "utility", "innovation",
            "backing", "regulation", "legal", "framework", "ecosystem", "progress"
        ]

        # Crypto-specific negative terms
        crypto_negative = [
            "crash", "ban", "crackdown", "bearish", "scam", "hack", "vulnerability",
            "exploit", "fraud", "pyramid", "bubble", "dump", "selling", "fear", "fears"
        ]

        # Adjust compound score based on crypto-specific terms
        text_lower = text.lower()

        # Count occurrences of positive and negative terms
        positive_matches = sum(term in text_lower for term in crypto_positive)
        negative_matches = sum(term in text_lower for term in crypto_negative)

        # Adjust score - for each positive term add 0.05, for each negative term subtract 0.05
        # This is a simple approach and could be refined further
        adjustment = (positive_matches - negative_matches) * 0.05

        # Apply adjustment to compound score
        adjusted_compound = min(1.0, max(-1.0, compound + adjustment))

        # Determine sentiment category based on adjusted score
        if adjusted_compound >= 0.05:
            sentiment = "Bullish"
        elif adjusted_compound <= -0.05:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        # Log the adjustment for debugging
        self.log(f"Sentiment adjustment: {compound} -> {adjusted_compound} (adj: {adjustment})")
        if positive_matches > 0:
            self.log(f"Positive crypto terms: {positive_matches}")
        if negative_matches > 0:
            self.log(f"Negative crypto terms: {negative_matches}")

        return sentiment, adjusted_compound

    def __init__(self, parent):
        """
        Initialize the news tab.

        Args:
            parent: Parent frame
        """
        self.parent = parent

        # Make sure NLTK data is downloaded
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            self.download_nltk_data()

        # Store results
        self.articles = []

        # Track running state
        self.running = False
        self.search_thread = None
        self.start_time = 0

        # Create main frame
        self.frame = ttk.Frame(parent, padding="10")

        # Create widgets
        self.create_widgets()

        # Pack the main frame
        self.frame.pack(fill=tk.BOTH, expand=True)

    def download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            messagebox.showerror("NLTK Download Error", f"Failed to download NLTK data: {str(e)}")

    def create_widgets(self):
        """Create all widgets for the tab."""
        # Create control panel
        self.create_control_panel()

        # Create results panel
        self.create_results_panel()

    def create_control_panel(self):
        """Create the control panel with input options."""
        control_frame = ttk.LabelFrame(self.frame, text="Crypto News Search", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Create grid for controls
        control_frame.columnconfigure(0, weight=0)  # Label column
        control_frame.columnconfigure(1, weight=1)  # Entry column
        control_frame.columnconfigure(2, weight=0)  # Button column
        control_frame.columnconfigure(3, weight=0)  # Additional button column

        # Keyword input
        ttk.Label(control_frame, text="Search Keyword:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.keyword_var = tk.StringVar(value="Bitcoin")
        keyword_entry = ttk.Entry(control_frame, textvariable=self.keyword_var, width=30)
        keyword_entry.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Days filter
        ttk.Label(control_frame, text="Max Age (days):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.days_var = tk.IntVar(value=7)
        days_spinbox = ttk.Spinbox(control_frame, from_=1, to=30, textvariable=self.days_var, width=5)
        days_spinbox.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Add button to run search
        self.run_button = ttk.Button(control_frame, text="Search News", command=self.run_news_search)
        self.run_button.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Add button to clear results
        clear_button = ttk.Button(control_frame, text="Clear Results", command=self.clear_results)
        clear_button.grid(row=1, column=2, padx=5, pady=5)

        # Create a frame for sources
        sources_frame = ttk.LabelFrame(control_frame, text="News Sources")
        sources_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W+tk.E, padx=5, pady=5)

        # Source checkboxes
        self.sources = {
            "coindesk": tk.BooleanVar(value=True),
            "cointelegraph": tk.BooleanVar(value=True),
            "cryptonews": tk.BooleanVar(value=True),
            "bitcoinmagazine": tk.BooleanVar(value=True),
            "decrypt": tk.BooleanVar(value=True),
            "cryptoslate": tk.BooleanVar(value=True),
            "cryptobriefing": tk.BooleanVar(value=True),
            "newsbtc": tk.BooleanVar(value=True)
        }

        # Create grid for source checkboxes - 4 columns
        sources_frame.columnconfigure(0, weight=1)
        sources_frame.columnconfigure(1, weight=1)
        sources_frame.columnconfigure(2, weight=1)
        sources_frame.columnconfigure(3, weight=1)

        # Add checkboxes with nicer display names
        source_labels = {
            "coindesk": "CoinDesk",
            "cointelegraph": "CoinTelegraph",
            "cryptonews": "CryptoNews",
            "bitcoinmagazine": "Bitcoin Magazine",
            "decrypt": "Decrypt",
            "cryptoslate": "CryptoSlate",
            "cryptobriefing": "Crypto Briefing",
            "newsbtc": "NewsBTC"
        }

        # Arrange checkboxes in a grid, 4 columns
        col, row = 0, 0
        for source_key, source_label in source_labels.items():
            ttk.Checkbutton(
                sources_frame,
                text=source_label,
                variable=self.sources[source_key]
            ).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            col += 1
            if col > 3:
                col = 0
                row += 1

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=3, column=0, columnspan=4, sticky=tk.W+tk.E, padx=5, pady=5)

    def create_results_panel(self):
        """Create the panel for displaying news results."""
        results_frame = ttk.LabelFrame(self.frame, text="Search Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create notebook for results tabs
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs for results and log
        self.news_results_frame = ttk.Frame(self.results_notebook, padding="10")
        self.log_frame = ttk.Frame(self.results_notebook, padding="10")
        self.article_frame = ttk.Frame(self.results_notebook, padding="10")

        self.results_notebook.add(self.news_results_frame, text="News")
        self.results_notebook.add(self.article_frame, text="Article Preview")
        self.results_notebook.add(self.log_frame, text="Log")

        # Create news results treeview
        self.create_news_treeview()

        # Create article preview area
        self.create_article_preview()

        # Create log text area
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def create_article_preview(self):
        """Create the article preview area."""
        # Title label
        self.article_title_var = tk.StringVar(value="Select an article to preview")
        title_label = ttk.Label(self.article_frame, textvariable=self.article_title_var,
                                font=("TkDefaultFont", 12, "bold"), wraplength=700)
        title_label.pack(fill=tk.X, pady=5)

        # Source and date
        self.article_meta_var = tk.StringVar(value="")
        meta_label = ttk.Label(self.article_frame, textvariable=self.article_meta_var,
                               font=("TkDefaultFont", 9, "italic"))
        meta_label.pack(fill=tk.X, pady=2)

        # Sentiment indicator
        self.sentiment_frame = ttk.Frame(self.article_frame)
        self.sentiment_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.sentiment_frame, text="Sentiment: ").pack(side=tk.LEFT)

        # Create styles for different sentiments
        style = ttk.Style()
        style.configure('Bullish.TLabel', background='#E8F5E9')  # Light green
        style.configure('Bearish.TLabel', background='#FFEBEE')  # Light red
        style.configure('Neutral.TLabel', background='#F5F5F5')  # Light gray
        self.styles_created = True

        # Default to standard style
        self.sentiment_label = ttk.Label(self.sentiment_frame, text="N/A", width=10)
        self.sentiment_label.pack(side=tk.LEFT)

        # Open URL button
        self.open_url_button = ttk.Button(self.article_frame, text="Open in Browser",
                                          command=self.open_selected_url, state=tk.DISABLED)
        self.open_url_button.pack(anchor=tk.W, pady=10)

        # Summary text
        ttk.Label(self.article_frame, text="Summary:").pack(anchor=tk.W, pady=5)
        self.article_summary = scrolledtext.ScrolledText(self.article_frame, height=15, wrap=tk.WORD)
        self.article_summary.pack(fill=tk.BOTH, expand=True)

    def create_news_treeview(self):
        """Create a treeview to display news results."""
        # Create frame for treeview and scrollbar
        frame = ttk.Frame(self.news_results_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollbars
        y_scrollbar = ttk.Scrollbar(frame)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        x_scrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create treeview
        columns = ("Title", "Source", "Sentiment", "Date", "URL")
        self.news_treeview = ttk.Treeview(
            frame, columns=columns, show="headings",
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set
        )

        # Configure columns
        self.news_treeview.heading("Title", text="Title")
        self.news_treeview.heading("Source", text="Source")
        self.news_treeview.heading("Sentiment", text="Sentiment")
        self.news_treeview.heading("Date", text="Published")
        self.news_treeview.heading("URL", text="URL")

        # Set column widths
        self.news_treeview.column("Title", width=300, minwidth=200)
        self.news_treeview.column("Source", width=100, minwidth=80)
        self.news_treeview.column("Sentiment", width=80, minwidth=80)
        self.news_treeview.column("Date", width=150, minwidth=100)
        self.news_treeview.column("URL", width=300, minwidth=200)

        # Configure scrollbars
        y_scrollbar.config(command=self.news_treeview.yview)
        x_scrollbar.config(command=self.news_treeview.xview)

        # Pack treeview
        self.news_treeview.pack(fill=tk.BOTH, expand=True)

        # Configure tags for sentiment colors
        self.news_treeview.tag_configure('bullish', background='#E8F5E9')  # Light green
        self.news_treeview.tag_configure('bearish', background='#FFEBEE')  # Light red
        self.news_treeview.tag_configure('neutral', background='#F5F5F5')  # Light gray

        # Bind double-click to open URL
        self.news_treeview.bind("<Double-1>", self.open_article_url)

        # Bind single-click to preview article
        self.news_treeview.bind("<<TreeviewSelect>>", self.preview_article)

    def log(self, message):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        # Insert log message in a thread-safe way
        self.parent.after(0, lambda: self.log_text.insert(tk.END, log_message))
        self.parent.after(0, lambda: self.log_text.see(tk.END))

    def update_status(self, message):
        """Update the status bar."""
        self.status_var.set(message)

    def extract_source_from_url(self, url):
        """Extract the website name from a URL."""
        try:
            parsed_url = urlparse(url)
            hostname = parsed_url.netloc

            # Remove www. if present
            if hostname.startswith('www.'):
                hostname = hostname[4:]

            # Extract the domain name (e.g., coindesk.com -> CoinDesk)
            parts = hostname.split('.')
            if len(parts) >= 2:
                source = parts[-2]
                # Capitalize and make it readable
                return source.capitalize()

            return hostname
        except:
            return "Unknown"

    def get_rss_feeds(self):
        """Get the list of RSS feeds based on selected sources."""
        feeds = []

        # Map of source keys to their RSS feed URLs
        feed_urls = {
            "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "cointelegraph": "https://cointelegraph.com/rss",
            "cryptonews": "https://cryptonews.com/news/feed/",
            "bitcoinmagazine": "https://bitcoinmagazine.com/feed",
            "decrypt": "https://decrypt.co/feed",
            "cryptoslate": "https://cryptoslate.com/feed/",
            "cryptobriefing": "https://cryptobriefing.com/feed/",
            "newsbtc": "https://www.newsbtc.com/feed/"
        }

        # Add selected feeds
        for source, enabled in self.sources.items():
            if enabled.get() and source in feed_urls:
                feeds.append(feed_urls[source])

        return feeds

    def clear_results(self):
        """Clear the results view."""
        # Clear treeview
        for item in self.news_treeview.get_children():
            self.news_treeview.delete(item)

        # Clear article preview
        self.article_title_var.set("Select an article to preview")
        self.article_meta_var.set("")

        # For ttk widgets, we need to use the style system instead of background
        # Reset the sentiment label to default style
        self.sentiment_label.config(text="N/A")
        self.sentiment_label['style'] = 'TLabel'  # Reset to default style

        self.article_summary.delete("1.0", tk.END)
        self.open_url_button.config(state=tk.DISABLED)

        # Reset articles list
        self.articles = []

        # Update status
        self.update_status("Results cleared")
        self.log("Results cleared")

    def run_news_search(self):
        """Run the news search in a separate thread."""
        if self.running:
            messagebox.showinfo("Running", "Search is already running. Please wait.")
            return

        # Check if at least one source is selected
        if not any(source.get() for source in self.sources.values()):
            messagebox.showinfo("No Sources", "Please select at least one news source.")
            return

        # Clear previous results
        self.clear_results()

        # Get search keyword
        keyword = self.keyword_var.get().strip()
        if not keyword:
            messagebox.showinfo("Input Error", "Please enter a search keyword.")
            return

        # Start timer
        self.start_time = time.time()

        # Start search thread
        self.running = True
        self.update_status(f"Searching for '{keyword}'...")
        self.run_button.config(state=tk.DISABLED)

        self.search_thread = threading.Thread(target=self.search_task, args=(keyword,))
        self.search_thread.daemon = True
        self.search_thread.start()

    def search_task(self, keyword):
        """
        Perform the news search in a separate thread.

        Args:
            keyword: The search keyword
        """
        try:
            self.log(f"Starting news search for '{keyword}'")

            # Get selected RSS feeds
            rss_feeds = self.get_rss_feeds()

            if not rss_feeds:
                self.log("No news sources selected. Please select at least one source.")
                self.update_status("No sources selected")
                return

            self.log(f"Using {len(rss_feeds)} news sources")

            articles = []
            feed_count = 0
            article_count = 0

            # Get maximum age for articles
            max_age_days = self.days_var.get()
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            self.log(f"Filtering for articles newer than {max_age_days} days")

            # Parse each RSS feed
            for feed_url in rss_feeds:
                try:
                    self.log(f"Fetching feed: {feed_url}")
                    feed = feedparser.parse(feed_url)
                    feed_count += 1

                    # Check feed validity
                    if not feed.entries:
                        self.log(f"No entries found in feed: {feed_url}")
                        continue

                    self.log(f"Found {len(feed.entries)} articles in feed")
                    matching_in_feed = 0

                    # Process each entry
                    for entry in feed.entries:
                        # Check if article is within date range
                        try:
                            # Try to parse the published date
                            if 'published_parsed' in entry and entry.published_parsed:
                                pub_date = datetime(*entry.published_parsed[:6])
                            elif 'published' in entry:
                                pub_date = self.parse_date(entry.published)
                            else:
                                # If no date, assume it's recent
                                pub_date = datetime.now()

                            # Skip if older than cutoff
                            if pub_date < cutoff_date:
                                continue

                        except Exception as e:
                            self.log(f"Error parsing date, assuming recent: {str(e)}")
                            # Continue anyway if date parsing fails

                        # Check keyword presence in title or summary (case-insensitive)
                        if (keyword.lower() in entry.title.lower() or
                                keyword.lower() in entry.get("summary", "").lower() or
                                keyword.lower() in entry.get("description", "").lower()):

                            # Extract source from URL
                            source = self.extract_source_from_url(entry.link)

                            # Get the summary/description text
                            summary = entry.get("summary", entry.get("description", "No summary available"))

                            # Try to clean HTML from summary if present
                            try:
                                from html import unescape
                                import re
                                # Unescape HTML entities
                                summary = unescape(summary)
                                # Remove HTML tags
                                summary = re.sub(r'<[^>]+>', ' ', summary)
                                # Remove extra whitespace
                                summary = re.sub(r'\s+', ' ', summary).strip()
                            except:
                                # If any error in cleaning, use as is
                                pass

                            articles.append({
                                "title": entry.title,
                                "link": entry.link,
                                "published": entry.get("published", "No Date"),
                                "summary": summary,
                                "source": source
                            })
                            article_count += 1
                            matching_in_feed += 1

                    self.log(f"Found {matching_in_feed} matching articles in this feed")

                except Exception as e:
                    self.log(f"Error parsing feed {feed_url}: {str(e)}")

            self.log(f"Processed {feed_count} feeds, found {article_count} matching articles")

            # Analyze sentiment for each article using our improved method
            for article in articles:
                # Combine title and summary for sentiment evaluation
                text = article["title"] + " " + article["summary"]

                # Use our enhanced crypto-specific sentiment analysis
                sentiment, compound_score = self.analyze_crypto_sentiment(text)

                article["sentiment"] = sentiment
                article["compound_score"] = compound_score

            # Sort articles by date (newest first), using complex logic to handle different date formats
            try:
                articles.sort(key=lambda x: self.parse_date(x["published"], default_recent=True), reverse=True)
            except:
                # If sorting fails, don't sort
                self.log("Error sorting articles by date")

            # Store articles for reference
            self.articles = articles

            # Update UI with results
            self.update_news_results()

            # Final status
            if articles:
                self.update_status(f"Found {len(articles)} articles for '{keyword}'")
            else:
                self.update_status(f"No articles found for '{keyword}'")

        except Exception as e:
            self.log(f"Error during search: {str(e)}")
            self.log(f"Traceback: {traceback.format_exc()}")
            self.update_status("Search failed")

        finally:
            # Reset running state
            self.running = False
            self.run_button.config(state=tk.NORMAL)

    def parse_date(self, date_str, default_recent=False):
        """Try to parse a date string using multiple formats."""
        if not date_str or date_str == "No Date":
            return datetime.now() if default_recent else datetime(1970, 1, 1)

        date_formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RSS standard
            "%a, %d %b %Y %H:%M:%S %Z",  # RSS variant
            "%Y-%m-%dT%H:%M:%S%z",       # ISO format
            "%Y-%m-%dT%H:%M:%SZ",        # ISO format (UTC)
            "%Y-%m-%d %H:%M:%S",         # Simple format
            "%d %b %Y %H:%M:%S %z",      # Another common format
            "%d %b %Y",                  # Just date
            "%B %d, %Y",                 # Month name, day, year
            "%Y/%m/%d"                   # Year/month/day
        ]

        # Try each format
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If all fails, try dateutil parser which is more flexible
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except:
            pass

        # If everything fails
        return datetime.now() if default_recent else datetime(1970, 1, 1)

    def update_news_results(self):
        """Update the news treeview with search results."""
        # Clear existing items
        for item in self.news_treeview.get_children():
            self.news_treeview.delete(item)

        if not self.articles:
            return

        # Insert articles into treeview
        for article in self.articles:
            # Determine tag based on sentiment
            if article["sentiment"] == "Bullish":
                tag = "bullish"
            elif article["sentiment"] == "Bearish":
                tag = "bearish"
            else:
                tag = "neutral"

            # Format title to prevent it from being too long
            title = article["title"]
            if len(title) > 80:
                title = title[:77] + "..."

            # Insert into treeview
            self.news_treeview.insert(
                "", tk.END,
                values=(
                    title,
                    article["source"],
                    article["sentiment"],
                    article["published"],
                    article["link"]
                ),
                tags=(tag,)
            )

    def preview_article(self, event):
        """Preview the selected article in the preview pane."""
        selected = self.news_treeview.selection()
        if not selected:
            return

        item = selected[0]
        values = self.news_treeview.item(item, "values")

        # Find the full article data
        article = None
        for a in self.articles:
            if a["link"] == values[4]:  # Check by URL
                article = a
                break

        if not article:
            return

        # Update preview pane
        self.article_title_var.set(article["title"])
        self.article_meta_var.set(f"{article['source']} • {article['published']}")

        # Set sentiment indicator
        sentiment = article["sentiment"]
        self.sentiment_label.config(text=sentiment)

        # Create styles for sentiment labels if they don't exist yet
        style = ttk.Style()
        if not hasattr(self, 'styles_created'):
            # Create styles for different sentiments
            style.configure('Bullish.TLabel', background='#E8F5E9')  # Light green
            style.configure('Bearish.TLabel', background='#FFEBEE')  # Light red
            style.configure('Neutral.TLabel', background='#F5F5F5')  # Light gray
            self.styles_created = True

        # Apply the appropriate style
        if sentiment == "Bullish":
            self.sentiment_label['style'] = 'Bullish.TLabel'
        elif sentiment == "Bearish":
            self.sentiment_label['style'] = 'Bearish.TLabel'
        else:
            self.sentiment_label['style'] = 'Neutral.TLabel'

        # Update summary
        self.article_summary.delete("1.0", tk.END)
        self.article_summary.insert("1.0", article["summary"])

        # Enable open URL button
        self.open_url_button.config(state=tk.NORMAL)

        # Focus article tab
        self.results_notebook.select(1)  # Select the Article Preview tab (index 1)

    def open_article_url(self, event):
        """Open the URL of the selected article in a browser."""
        selected = self.news_treeview.selection()
        if not selected:
            return

        item = selected[0]

        # Get the URL from the selected item
        url = self.news_treeview.item(item, "values")[4]

        if url:
            self.log(f"Opening URL: {url}")
            webbrowser.open(url)

    def open_selected_url(self):
        """Open the currently previewed article URL."""
        selected = self.news_treeview.selection()
        if not selected:
            return

        item = selected[0]
        url = self.news_treeview.item(item, "values")[4]

        if url:
            self.log(f"Opening URL: {url}")
            webbrowser.open(url)