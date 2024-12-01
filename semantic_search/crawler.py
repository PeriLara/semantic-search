import json
import os
from typing import List
from urllib.error import HTTPError, URLError

import click
import feedparser
from loguru import logger

from semantic_search.utils import ARTICLES_DIR, RSS_FEED_RESOURCES_DIR


def get_rss_urls(feed_filename: str) -> List[str]:
    """Read RSS feed URLs from a JSONL file"""
    rss_feed_path: str = os.path.join(RSS_FEED_RESOURCES_DIR, feed_filename)
    rss_urls = []
    with open(rss_feed_path, "r") as f:
        for line in f:
            feed_data = json.loads(line.strip())
            rss_urls.append(feed_data["url"])
    return rss_urls


def fetch_rss_feed(rss_url: str) -> List[dict]:
    logger.info(f"Fetching RSS feed from {rss_url}")
    try:
        feed = feedparser.parse(rss_url)
        if feed.bozo:
            logger.warning(
                f"Warning: There was an error parsing the feed: {feed.bozo_exception}"
            )
        return feed.entries
    except (
        URLError,
        HTTPError,
        TimeoutError,
    ) as e:
        logger.error(f"Network error while fetching RSS feed from {rss_url}: {str(e)}")
        return []


@click.command()
@click.option(
    "--feed-filename",
    type=click.Choice(os.listdir(RSS_FEED_RESOURCES_DIR)),
    help=f"Name of the RSS feed file in {RSS_FEED_RESOURCES_DIR} directory",
    default="public_fr.jsonl",
)
def main(feed_filename: str):
    """Fetch RSS feeds and save articles to files.

    Args:
        feed_filename: Name of the input RSS feeds file
    """
    feed_urls: list[str] = get_rss_urls(feed_filename)
    os.makedirs(ARTICLES_DIR, exist_ok=True)
    for feed_url in feed_urls:
        entries = fetch_rss_feed(feed_url)
        with open(
            os.path.join(ARTICLES_DIR, feed_filename), "a", encoding="utf-8"
        ) as f:
            for article in entries:
                f.write(json.dumps(article) + "\n")


if __name__ == "__main__":
    main()
