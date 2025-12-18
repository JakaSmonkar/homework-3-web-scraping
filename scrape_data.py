import json
import time
import re
import pandas as pd
import requests

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
from datetime import timezone
from urllib.parse import urljoin, urlparse

BASE = "https://web-scraping.dev"

@dataclass
class ScrapeConfig:
    testimonials_secret_token: str = "secret123"
    testimonials_referer: str = f"{BASE}/testimonials"
    reviews_csrf_token: str = "secret-csrf-token-123"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    sleep_s: float = 0.2
 
# Pridobivanje produktov
def scrape_products_html(max_pages: int = 50):
    cfg = ScrapeConfig()
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent})

    products = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        r = session.get(f"{BASE}/products", params={"page": page}, timeout=30)
        if r.status_code != 200:
            print(f"Products: stopping at page={page} (status {r.status_code})")
            break

        soup = BeautifulSoup(r.text, "html.parser")

        rows = soup.select("div.row.product")
        if not rows:
            break

        page_added = 0
        for row in rows:
            a = row.select_one(".description h3 a[href]")
            if not a:
                continue

            url = urljoin(BASE, a["href"].strip())
            path = urlparse(url).path
            if re.search(r"^/product/\d+/?$", path) is None:
                continue

            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = a.get_text(strip=True)

            img = row.select_one(".thumbnail img[src]")
            image_url = urljoin(BASE, img["src"].strip()) if img and img.get("src") else None

            desc_el = row.select_one(".short-description")
            description = desc_el.get_text(" ", strip=True) if desc_el else None

            price_el = row.select_one(".price-wrap .price")
            price = None
            if price_el:
                txt = price_el.get_text(strip=True).replace("$", "").replace(",", "")
                try:
                    price = float(txt)
                except:
                    price = None

            products.append({
                "title": title,
                "url": url,
                "page": page,
                "image_url": image_url,
                "price": price,
                "description": description,
            })
            page_added += 1

        if page_added == 0:
            break

        time.sleep(cfg.sleep_s)

    return products

# Pridobivanje testimonials
def scrape_testimonials_api(max_pages: int = 50):
    cfg = ScrapeConfig()
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent})

    testimonials = []

    for page in range(1, max_pages + 1):
        url = f"{BASE}/api/testimonials"
        headers = {
            "Referer": cfg.testimonials_referer,
            "X-Secret-Token": cfg.testimonials_secret_token,
        }

        r = session.get(url, params={"page": page}, headers=headers, timeout=30)

        #konca se z 403
        if r.status_code != 200:
            print(f"Testimonials: stopping at page={page} (status {r.status_code})")
            break

        soup = BeautifulSoup(r.text, "html.parser")
        cards = soup.select("div.testimonial")

        if not cards:
            break

        for idx, card in enumerate(cards):
            #username
            ident = card.select_one("identicon-svg")
            username = ident.get("username") if ident else None

            #description
            p = card.select_one("p.text")
            text = p.get_text(" ", strip=True) if p else None

            #št zvezdic
            rating = len(card.select(".rating svg"))

            if text:
                testimonials.append(
                    {
                        "page": page,
                        "idx": idx,
                        "username": username,
                        "text": text,
                        "rating": rating,
                    }
                )

        time.sleep(cfg.sleep_s)

    df = pd.DataFrame(testimonials).drop_duplicates(subset=["text"])
    return df.to_dict(orient="records")

def _parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse dates like 2023-02-10 etc.
    """
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except Exception:
        return None

def scrape_reviews_api(max_pages: int = 50, first: int = 20):
    cfg = ScrapeConfig()
    session = requests.Session()
    session.headers.update({
        "User-Agent": cfg.user_agent,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Referer": f"{BASE}/reviews",
    })

    query = """
    query GetReviews($first: Int, $after: String) {
      reviews(first: $first, after: $after) {
        edges {
          node {
            rid
            text
            rating
            date
          }
          cursor
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
    """

    all_reviews = []
    after = None

    for _ in range(max_pages):
        payload = {
            "query": query,
            "variables": {"first": first, "after": after},
        }

        r = session.post(f"{BASE}/api/graphql", json=payload, timeout=30)
        if r.status_code != 200:
            print("Reviews GraphQL stopped:", r.status_code, r.text[:200])
            break

        data = r.json()
        if "errors" in data:
            print("Reviews GraphQL errors:", data["errors"])
            break

        reviews_data = data.get("data", {}).get("reviews", {})
        edges = reviews_data.get("edges", [])

        if not edges:
            break

        for edge in edges:
            node = edge.get("node", {})
            date_str = node.get("date")
            text = node.get("text")
            rating = node.get("rating")

            if not date_str or not text:
                continue

            dt = _parse_date(date_str)
            if not dt:
                continue

            all_reviews.append({
                "rid": node.get("rid"),
                "date": dt.date().isoformat(),
                "text": text.strip(),
                "rating": rating,
            })

        page_info = reviews_data.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break

        after = page_info.get("endCursor") or edges[-1].get("cursor")
        time.sleep(cfg.sleep_s)

    df = pd.DataFrame(all_reviews).drop_duplicates(subset=["date", "text"])
    return df.to_dict(orient="records")

def main() -> None:
    out = {
        "products": scrape_products_html(),
        "testimonials": scrape_testimonials_api(),
        "reviews": scrape_reviews_api(),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
    }
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved data.json")
    print({k: (len(v) if isinstance(v, list) else v) for k, v in out.items() if k in ["products", "testimonials", "reviews"]})


if __name__ == "__main__":
    main()
