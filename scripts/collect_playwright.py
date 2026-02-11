"""Playwright collector skeleton for stats.gjf.or.kr.

This script logs in via the '관계자' button and validates that key navigation is reachable.
Selector values are read from config/selectors.json (or config/selectors.example.json).

NOTE:
- Real region/indicator extraction is still pending exact selector confirmation.
- Use this as the production entrypoint once selectors are finalized.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright

BASE_URL = "https://stats.gjf.or.kr"
DATA_DIR = Path("data/snapshots")
ARTIFACT_DIR = Path("artifacts")
SELECTOR_CANDIDATES = [
    Path("config/selectors.json"),
    Path("config/selectors.example.json"),
]


class ConfigError(RuntimeError):
    pass


def load_selectors() -> dict[str, str]:
    for path in SELECTOR_CANDIDATES:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise ConfigError("No selectors file found. Create config/selectors.json.")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required env var: {name}")
    return value


async def login(page: Page, selectors: dict[str, str], user: str, password: str) -> None:
    await page.goto(BASE_URL, wait_until="domcontentloaded")
    await page.locator(selectors["observer_login_button"]).first.click(timeout=15000)

    await page.locator(selectors["login_user_input"]).fill(user, timeout=15000)
    await page.locator(selectors["login_password_input"]).fill(password, timeout=15000)
    await page.locator(selectors["login_submit_button"]).click(timeout=15000)

    await page.locator(selectors["login_success_anchor"]).first.wait_for(timeout=20000)


async def collect_minimal(page: Page, selectors: dict[str, str]) -> list[dict[str, Any]]:
    await page.locator(selectors["employment_tab_button"]).first.click(timeout=15000)
    await page.wait_for_timeout(1500)

    snapshot_month = datetime.now().strftime("%Y-%m")
    now = datetime.now().isoformat(timespec="seconds")

    # Placeholder row until exact extraction selectors are confirmed.
    return [
        {
            "snapshot_month": snapshot_month,
            "region_level": "national",
            "region_name": "전국",
            "indicator": "피보험자수",
            "current_value": 0,
            "current_signal": "정상",
            "prev_2m_value": 0,
            "prev_2m_signal": "정상",
            "collected_at": now,
            "note": "playwright skeleton run; replace with real extraction",
        }
    ]


def save_rows(rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise RuntimeError("No rows collected")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_month = str(rows[0]["snapshot_month"])
    output = DATA_DIR / f"{snapshot_month}.csv"

    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output


async def run() -> None:
    selectors = load_selectors()
    user = required_env("GJF_USER")
    password = required_env("GJF_PASSWORD")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await login(page, selectors, user, password)
            rows = await collect_minimal(page, selectors)
            output = save_rows(rows)
            print(f"Saved snapshot: {output}")
        except Exception:
            screenshot = ARTIFACT_DIR / "collector_failure.png"
            await page.screenshot(path=str(screenshot), full_page=True)
            print(f"Saved failure screenshot: {screenshot}")
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
