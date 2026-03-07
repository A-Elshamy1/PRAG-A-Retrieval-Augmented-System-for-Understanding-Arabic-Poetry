import os
import json
import time
import requests
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DATA_PATH      = os.getenv("DATA_PATH", "data/poems_raw.json")
BASE_URL       = "https://www.aldiwan.net"
DELAY          = 2
TARGET_POEMS   = 200
POEMS_PER_POET = 10


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ar,en;q=0.9",
    })
    return session


def get_poem_links(poet_url: str, session: requests.Session) -> list[str]:
    resp = session.get(poet_url, timeout=15)
    soup = BeautifulSoup(resp.text, "lxml")

    poem_links = list(dict.fromkeys([
        urljoin(BASE_URL, a["href"])
        for a in soup.find_all("a", href=True)
        if re.search(r'poem\d+\.html', a["href"])
    ]))

    return poem_links[:POEMS_PER_POET]


def get_poem_data(poem_url: str, poet_name: str, session: requests.Session) -> dict | None:
    resp = session.get(poem_url, timeout=15)
    soup = BeautifulSoup(resp.text, "lxml")

    title_parts = soup.title.text.strip().split(" - ") if soup.title else []
    title       = title_parts[0] if title_parts else "بلا عنوان"

    era_tag = soup.find("p", class_="main-color")
    era     = era_tag.get_text(strip=True) if era_tag else "غير محدد"

    body = soup.find("div", class_="bet-1")
    if not body:
        return None

    poem_text = body.get_text(separator="\n", strip=True)
    if len(poem_text) < 50:
        return None

    lines_tag = soup.find("span", class_="label")
    num_lines = lines_tag.get_text(strip=True) if lines_tag else "غير محدد"

    return {
        "title":     title,
        "poet_name": poet_name,
        "era":       era,
        "num_lines": num_lines,
        "poem_text": poem_text,
        "poem_url":  poem_url,
    }


def scrape_poems() -> list[dict]:
    session    = make_session()
    all_poems  = []
    poets_done = 0
    page       = 1

    print("=" * 60)
    print(f"[AR] بدء الاستخراج — الهدف {TARGET_POEMS} قصيدة")
    print(f"[EN] Starting scrape — target {TARGET_POEMS} poems")
    print("=" * 60)

    while len(all_poems) < TARGET_POEMS:
        url  = f"{BASE_URL}/authers-1?page={page}"
        resp = session.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")

        poet_anchors = [
            a for a in soup.find_all("a", href=True)
            if "cat-poet-" in a["href"] and a.get_text(strip=True)
        ]

        if not poet_anchors:
            print(f"[EN] No more poets on page {page}, stopping.")
            break

        for anchor in poet_anchors:
            if len(all_poems) >= TARGET_POEMS:
                break

            poet_url  = urljoin(BASE_URL, anchor["href"])
            poet_name = anchor.get_text(strip=True)
            poets_done += 1

            print(f"\n[EN] Poet #{poets_done}: {poet_name}")
            print(f"[AR] الشاعر #{poets_done}: {poet_name}")

            time.sleep(DELAY)
            poem_links = get_poem_links(poet_url, session)
            print(f"  [EN] Found {len(poem_links)} poem links")

            if not poem_links:
                print(f"  [EN] Skipping — no poems found")
                continue

            poet_poems_count = 0
            for poem_url in poem_links:
                if len(all_poems) >= TARGET_POEMS:
                    break

                time.sleep(DELAY)
                poem_data = get_poem_data(poem_url, poet_name, session)

                if poem_data:
                    all_poems.append(poem_data)
                    poet_poems_count += 1
                    total = len(all_poems)
                    print(f"  [{total:03d}/{TARGET_POEMS}] {poem_data['title'][:50]}")

                    if total % 10 == 0:
                        print(f"\n[AR] تم جمع {total} قصيدة حتى الآن...")
                        print(f"[EN] Collected {total} poems so far...\n")

            print(f"  [EN] Got {poet_poems_count} poems from {poet_name}")

        page += 1
        time.sleep(DELAY)

    print(f"\n[AR] اكتمل: {len(all_poems)} قصيدة من {poets_done} شاعر")
    print(f"[EN] Done: {len(all_poems)} poems from {poets_done} poets")
    return all_poems


def save_poems(poems: list[dict]) -> None:
    os.makedirs(os.path.dirname(DATA_PATH) if os.path.dirname(DATA_PATH) else ".", exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(poems, f, ensure_ascii=False, indent=2)
    print(f"[AR] تم الحفظ في: {DATA_PATH}")
    print(f"[EN] Saved to: {DATA_PATH}")


def main():
    poems = scrape_poems()

    if poems:
        save_poems(poems)
        print("\n[EN] Sample poem:")
        sample = poems[0]
        print(f"  Title    : {sample['title']}")
        print(f"  Poet     : {sample['poet_name']}")
        print(f"  Era      : {sample['era']}")
        print(f"  Lines    : {sample['num_lines']}")
        print(f"  URL      : {sample['poem_url']}")
        print(f"  Text     : {sample['poem_text'][:100]}...")
    else:
        print("[EN] No poems collected. Check your internet connection.")


if __name__ == "__main__":
    main()