import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig,BrowserConfig,CacheMode,RateLimiter,CrawlerMonitor, DisplayMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import base64
import os

def process_result(result):
    print(result.metadata)  # Print the first page's content
    if result.screenshot:
        print("Screenshot captured (base64, length):", len(result.screenshot))
        # Decode the base64 screenshot and save it to a file
        with open("screenshot.png", "wb") as f:
            f.write(base64.b64decode(result.screenshot))

    
    if result.success:
        internal_links = result.links.get("internal", [])
        external_links = result.links.get("external", [])
        print(f"Found {len(internal_links)} internal links.")
        print(f"Found {len(external_links)} external links.")
        # print(f"Found {len(result.media)} media items.")

        # for link in internal_links:
        #     print(f"Internal link: {link}")
        # for link in external_links:
        #     print(f"External link: {link}")

async def main():

    # Create a chain of filters
    filter_chain = FilterChain([
        # Only follow URLs with specific patterns
        URLPatternFilter(patterns=["*/sap-ai-core/*"]),
        URLPatternFilter(patterns=["^(?:(?!#).)*"]),

        # Only crawl specific domains
        DomainFilter(
            allowed_domains=["help.sap.com"],
            # blocked_domains=["old.docs.example.com"]
        ),

        # Only include specific content types
        # ContentTypeFilter(allowed_types=["text/html"])
    ])
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=90.0,  # Pause if memory exceeds this
        check_interval=1.0,             # How often to check memory
        max_session_permit=10,          # Maximum concurrent tasks
        rate_limiter=RateLimiter(       # Optional rate limiting
            base_delay=(1.0, 2.0),
            max_delay=30.0,
            max_retries=2
        ),
        monitor=CrawlerMonitor(         # Optional monitoring
            # max_visible_rows=15,
            # display_mode=DisplayMode.DETAILED, #AGGREGATED, DETAILED
        )
    )


    downloads_path = os.path.join(os.getcwd(), "downloads")  # Custom download path
    os.makedirs(downloads_path, exist_ok=True)

    browser_conf = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=True,
        # accept_downloads=True, 
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        text_mode=True,
        # downloads_path=downloads_path
    )

    prune_filter = PruningContentFilter(
        threshold=0.5,
        threshold_type="dynamic",  # or "dynamic"
        min_word_threshold=15
    )
    
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)
    
    # Configure a 2-level deep crawl
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=1, 
            filter_chain=filter_chain,
            include_external=False,
            max_pages=10,
        ),
        markdown_generator=md_generator,
        # scraping_strategy=LXMLWebScrapingStrategy(),
        # verbose=True,
        simulate_user=True,
        override_navigator=True,
        magic=True,
        page_timeout=10000,
        exclude_social_media_links=True,
        cache_mode=CacheMode.BYPASS,
        # delay_before_return_html=5.0,
        # css_selector="#topic-comment-container",
        screenshot=True,
        # wait_until="networkidle",
        wait_for="css:#topic-title",
        stream=True,
        remove_overlay_elements=True,   # Remove popups/modals
        process_iframes=True            # Process iframe content

    )

    urls = ["https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/what-is-sap-ai-core"]

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        async for result in await crawler.arun(
            url=urls[0],
            config=run_config,
            dispatcher=dispatcher
        ):
            if result.success:
                # Process each result immediately
                process_result(result)
            else:
                print(f"Failed to crawl {result.url}: {result.error_message}")
            

if __name__ == "__main__":
    asyncio.run(main())