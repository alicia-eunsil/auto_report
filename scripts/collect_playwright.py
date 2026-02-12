"""Playwright collector for stats.gjf.or.kr.

Collector flow:
1) Login with observer account
2) Move to employment insurance tab
3) Iterate region levels/options and extract 6 indicators for each region
4) Save normalized snapshot CSV

The script is intentionally selector-driven. You must provide real selectors in
`config/selectors.json` (copy from `config/selectors.example.json`).
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import Dialog, FrameLocator, Locator, Page, async_playwright

BASE_URL = "https://stats.gjf.or.kr"
DATA_DIR = Path("data/snapshots")
ARTIFACT_DIR = Path("artifacts")
SELECTOR_CANDIDATES = [
    Path("config/selectors.json"),
    Path("config/selectors.example.json"),
]

EXPECTED_LEVEL_COUNTS = {
    "national": 1,
    "province": 17,
    "gyeonggi_city": 31,
}
EXPECTED_INDICATOR_COUNT = 6


DEFAULT_INDICATOR_NAMES = [
    "피보험자수",
    "취득자수",
    "상실자수",
    "사업장수",
    "사업장 성립",
    "사업장 소멸",
]


class ConfigError(RuntimeError):
    pass


@dataclass
class ExtractionContext:
    snapshot_month: str
    collected_at: str


def load_selectors() -> dict[str, Any]:
    for path in SELECTOR_CANDIDATES:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                selectors = json.load(f)

            # Keep the loaded path for clearer runtime diagnostics.
            selectors["__selectors_path"] = str(path)
            return selectors
    raise ConfigError("No selectors file found. Create config/selectors.json.")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required env var: {name}")
    return value


def parse_snapshot_month(text: str) -> str:
    # examples: 2026년 02월, 2026-02
    m = re.search(r"(20\d{2})\D{0,3}(\d{1,2})", text)
    if not m:
        return datetime.now().strftime("%Y-%m")
    year = int(m.group(1))
    month = int(m.group(2))
    return f"{year:04d}-{month:02d}"


def parse_number(text: str) -> str:
    cleaned = text.replace(",", "").strip()
    if not cleaned:
        return ""
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return m.group(0) if m else ""


def normalize_signal(text: str, cls: str = "") -> str:
    token = f"{text} {cls}".lower()
    if "정상" in token or "green" in token:
        return "정상"
    if "관심" in token or "yellow" in token or "orange" in token:
        return "관심"
    if "주의" in token or "red" in token:
        return "주의"
    return ""


def should_skip_region_option(name: str) -> bool:
    token = name.replace(" ", "")
    skip_tokens = {"전체", "선택", "선택하세요", "전체보기", "all"}
    return token.lower() in skip_tokens


async def open_region_dropdown_if_present(page: Page, selectors: dict[str, Any]) -> bool:
    opener = selectors.get("region_dropdown_button")
    if not opener:
        return False

    trigger = page.locator(opener).first
    if await trigger.count() == 0:
        return False

    await trigger.click(timeout=15000)
    await page.wait_for_timeout(300)
    return True


async def resolve_login_scope(page: Page, selectors: dict[str, Any]) -> Page | FrameLocator:
    frame_selector = selectors.get("login_iframe_selector", "").strip()
    if not frame_selector:
        return page

    frame_host = page.locator(frame_selector).first
    try:
        await frame_host.wait_for(state="visible", timeout=5000)
    except Exception:
        return page
    return page.frame_locator(frame_selector).first


async def click_resilient(locator: Locator, timeout: int = 15000) -> None:
    try:
        await locator.click(timeout=timeout)
        return
    except Exception as first_error:
        # Fallback for elements rendered outside viewport in headless CI.
        try:
            await locator.click(timeout=5000, force=True)
            return
        except Exception:
            pass
        try:
            await locator.evaluate("el => el.click()")
            return
        except Exception:
            raise first_error


async def login(page: Page, selectors: dict[str, Any], user: str, password: str) -> None:
    async def accept_dialog(dialog: Dialog) -> None:
        await dialog.accept()

    await page.goto(BASE_URL, wait_until="domcontentloaded")
    page.on("dialog", accept_dialog)

    await click_resilient(page.locator(selectors["observer_login_button"]).first, timeout=15000)

    login_scope = await resolve_login_scope(page, selectors)

    await login_scope.locator(selectors["login_user_input"]).fill(user, timeout=15000)
    await login_scope.locator(selectors["login_password_input"]).fill(password, timeout=15000)
    await login_scope.locator(selectors["login_submit_button"]).click(timeout=15000)

    # Some environments show a custom confirmation modal instead of JS alert.
    popup_confirm_selector = selectors.get("login_popup_confirm_button")
    if popup_confirm_selector:
        popup_confirm_button = page.locator(popup_confirm_selector).first
        try:
            await popup_confirm_button.wait_for(state="visible", timeout=4000)
            await popup_confirm_button.click(timeout=4000)
            await page.wait_for_timeout(300)
        except Exception:
            # Continue when popup is not present in this session.
            pass

    try:
        await page.locator(selectors["login_success_anchor"]).first.wait_for(timeout=20000)
    finally:
        page.remove_listener("dialog", accept_dialog)


async def region_names_from_page(page: Page, selectors: dict[str, Any], level_key: str) -> list[str]:
    configured = selectors.get("region_names", {}).get(level_key)
    if isinstance(configured, list) and configured:
        return [name for name in configured if name and not should_skip_region_option(name)]

    options_selector = selectors.get("region_option_items")
    if not options_selector:
        return []

    names: list[str] = []
    dropdown_opened = await open_region_dropdown_if_present(page, selectors)
    try:
        items = page.locator(options_selector)
        for i in range(await items.count()):
            txt = (await items.nth(i).inner_text()).strip()
            if not txt or should_skip_region_option(txt) or txt in names:
                continue
            names.append(txt)
    finally:
        if dropdown_opened:
            # close dropdown by clicking trigger again to keep UI state predictable
            await open_region_dropdown_if_present(page, selectors)

    return names


async def click_region_level(page: Page, selectors: dict[str, Any], level_key: str) -> None:
    level_selector = selectors.get("region_level_buttons", {}).get(level_key)
    if not level_selector:
        raise ConfigError(f"Missing selector: region_level_buttons.{level_key}")
    await page.locator(level_selector).first.click(timeout=15000)
    await page.wait_for_timeout(400)


async def select_region_name(page: Page, selectors: dict[str, Any], region_name: str) -> None:
    opener = selectors.get("region_dropdown_button")
    option = selectors.get("region_option_by_text")
    if not opener or not option:
        raise ConfigError("Missing region selectors: region_dropdown_button / region_option_by_text")

    opened = await open_region_dropdown_if_present(page, selectors)
    if not opened:
        raise RuntimeError("Region dropdown trigger not found. Check region_dropdown_button selector.")

    target_selector = option.replace("{name}", region_name)
    await page.locator(target_selector).first.click(timeout=15000)
    await page.wait_for_timeout(600)


async def extract_indicator_rows(
    page: Page,
    selectors: dict[str, Any],
    ctx: ExtractionContext,
    level_key: str,
    region_name: str,
) -> list[dict[str, str]]:
    row_selector = selectors.get("indicator_row")
    if not row_selector:
        raise ConfigError("Missing selector: indicator_row")

    rows: list[dict[str, str]] = []
    row_loc = page.locator(row_selector)

    required = [
        "indicator_name",
        "current_value",
        "current_signal",
        "prev_1m_signal",
        "prev_2m_value",
        "prev_2m_signal",
    ]
    missing = [k for k in required if k not in selectors]
    if missing:
        raise ConfigError(f"Missing selectors: {', '.join(missing)}")

    for i in range(await row_loc.count()):
        row = row_loc.nth(i)

        def child(sel_key: str) -> Locator:
            return row.locator(selectors[sel_key]).first

        indicator = (await child("indicator_name").inner_text()).strip()
        if not indicator:
            continue

        current_value_raw = (await child("current_value").inner_text()).strip()
        current_signal_text = (await child("current_signal").inner_text()).strip()
        current_signal_class = await child("current_signal").get_attribute("class") or ""

        prev_1m_signal_text = (await child("prev_1m_signal").inner_text()).strip()
        prev_1m_signal_class = await child("prev_1m_signal").get_attribute("class") or ""

        prev_value_raw = (await child("prev_2m_value").inner_text()).strip()
        prev_signal_text = (await child("prev_2m_signal").inner_text()).strip()
        prev_signal_class = await child("prev_2m_signal").get_attribute("class") or ""

        current_signal = normalize_signal(current_signal_text, current_signal_class)
        prev_1m_signal = normalize_signal(prev_1m_signal_text, prev_1m_signal_class)
        prev_signal = normalize_signal(prev_signal_text, prev_signal_class)

        if not current_signal or not prev_1m_signal or not prev_signal:
            # Skip noisy rows like headers or empty templates.
            continue

        rows.append(
            {
                "snapshot_month": ctx.snapshot_month,
                "region_level": level_key,
                "region_name": region_name,
                "indicator": indicator,
                "current_value": parse_number(current_value_raw),
                "current_signal": current_signal,
                "prev_1m_signal": prev_1m_signal,
                "prev_2m_value": parse_number(prev_value_raw),
                "prev_2m_signal": prev_signal,
                "collected_at": ctx.collected_at,
            }
        )

    return rows


def card_mode_enabled(selectors: dict[str, Any]) -> bool:
    return bool(selectors.get("region_cards_selector"))


def indicator_names(selectors: dict[str, Any]) -> list[str]:
    names = selectors.get("indicator_names")
    if isinstance(names, list) and names:
        return [str(n).strip() for n in names if str(n).strip()]
    return DEFAULT_INDICATOR_NAMES


async def click_indicator(page: Page, selectors: dict[str, Any], indicator: str) -> None:
    button_tpl = selectors.get("indicator_button_by_text")
    if not button_tpl:
        raise ConfigError("Missing selector: indicator_button_by_text")
    sel = button_tpl.replace("{name}", indicator)
    await page.locator(sel).first.click(timeout=15000)
    await page.wait_for_timeout(500)


async def extract_region_cards(
    page: Page,
    selectors: dict[str, Any],
    ctx: ExtractionContext,
    source_level: str,
    indicator: str,
) -> list[dict[str, str]]:
    required = [
        "region_cards_selector",
        "card_region_name",
        "card_current_value",
        "card_signal_items",
    ]
    missing = [k for k in required if k not in selectors]
    if missing:
        source = selectors.get("__selectors_path", "config/selectors.json")
        raise ConfigError(
            "Missing selectors for card mode: "
            f"{', '.join(missing)}. "
            f"Please add them to {source}."
        )

    rows: list[dict[str, str]] = []
    cards = page.locator(selectors["region_cards_selector"])
    for i in range(await cards.count()):
        card = cards.nth(i)
        region_name = (await card.locator(selectors["card_region_name"]).first.inner_text()).strip()
        if not region_name:
            continue

        current_value_raw = (await card.locator(selectors["card_current_value"]).first.inner_text()).strip()

        signals = card.locator(selectors["card_signal_items"])
        if await signals.count() < 3:
            continue

        s0 = signals.nth(0)
        s1 = signals.nth(1)
        s2 = signals.nth(2)
        current_signal = normalize_signal((await s0.inner_text()).strip(), await s0.get_attribute("class") or "")
        prev_1m_signal = normalize_signal((await s1.inner_text()).strip(), await s1.get_attribute("class") or "")
        prev_2m_signal = normalize_signal((await s2.inner_text()).strip(), await s2.get_attribute("class") or "")

        if not current_signal or not prev_1m_signal or not prev_2m_signal:
            continue

        level_key = source_level
        if source_level == "province" and region_name == "전국":
            level_key = "national"

        rows.append(
            {
                "snapshot_month": ctx.snapshot_month,
                "region_level": level_key,
                "region_name": region_name,
                "indicator": indicator,
                "current_value": parse_number(current_value_raw),
                "current_signal": current_signal,
                "prev_1m_signal": prev_1m_signal,
                "prev_2m_value": "",
                "prev_2m_signal": prev_2m_signal,
                "collected_at": ctx.collected_at,
            }
        )

    return rows


async def collect_rows_from_cards(page: Page, selectors: dict[str, Any], ctx: ExtractionContext) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    for level_key in ["province", "gyeonggi_city"]:
        await click_region_level(page, selectors, level_key)
        for indicator in indicator_names(selectors):
            await click_indicator(page, selectors, indicator)
            rows = await extract_region_cards(page, selectors, ctx, level_key, indicator)
            if not rows:
                raise RuntimeError(f"No region cards extracted for {level_key}/{indicator}")
            all_rows.extend(rows)
    return all_rows


def validate_completeness(rows: list[dict[str, str]]) -> None:
    by_level_region: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["region_level"], row["region_name"])
        by_level_region[key] = by_level_region.get(key, 0) + 1

    level_regions: dict[str, set[str]] = {k: set() for k in EXPECTED_LEVEL_COUNTS}
    for level, region in by_level_region:
        if level in level_regions:
            level_regions[level].add(region)

    missing_msgs: list[str] = []
    for level, expected_region_count in EXPECTED_LEVEL_COUNTS.items():
        region_count = len(level_regions.get(level, set()))
        if region_count != expected_region_count:
            missing_msgs.append(
                f"{level}: expected {expected_region_count} regions, got {region_count}"
            )

    for (level, region), count in sorted(by_level_region.items()):
        if count != EXPECTED_INDICATOR_COUNT:
            missing_msgs.append(
                f"{level}/{region}: expected {EXPECTED_INDICATOR_COUNT} indicators, got {count}"
            )

    if missing_msgs:
        joined = "\n - ".join(missing_msgs[:20])
        raise RuntimeError(
            "Collection incomplete. Check selectors and region mapping.\n - " + joined
        )


async def collect_rows(page: Page, selectors: dict[str, Any]) -> list[dict[str, str]]:
    early_warning_menu_sel = selectors.get("early_warning_service_menu_button")
    if early_warning_menu_sel:
        await page.locator(early_warning_menu_sel).first.click(timeout=15000)
        await page.wait_for_timeout(800)

    await page.locator(selectors["employment_tab_button"]).first.click(timeout=15000)
    await page.wait_for_timeout(1200)

    month_sel = selectors.get("snapshot_month_text")
    month_text = datetime.now().strftime("%Y-%m")
    if month_sel:
        loc = page.locator(month_sel).first
        if await loc.count() > 0:
            month_text = await loc.inner_text()

    ctx = ExtractionContext(
        snapshot_month=parse_snapshot_month(month_text),
        collected_at=datetime.now().isoformat(timespec="seconds"),
    )

    all_rows: list[dict[str, str]] = []

    if card_mode_enabled(selectors):
        all_rows = await collect_rows_from_cards(page, selectors, ctx)
    else:
        for level_key in ["national", "province", "gyeonggi_city"]:
            await click_region_level(page, selectors, level_key)
            names = await region_names_from_page(page, selectors, level_key)

            if not names and level_key == "national":
                names = ["전국"]
            if not names:
                raise RuntimeError(f"No region names found for level: {level_key}")

            for region_name in names:
                if not (level_key == "national" and region_name == "전국"):
                    await select_region_name(page, selectors, region_name)
                rows = await extract_indicator_rows(page, selectors, ctx, level_key, region_name)
                if not rows:
                    raise RuntimeError(f"No indicator rows extracted for {level_key}/{region_name}")
                all_rows.extend(rows)

    validate_completeness(all_rows)
    return all_rows


def save_rows(rows: list[dict[str, str]]) -> Path:
    if not rows:
        raise RuntimeError("No rows collected")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_month = str(rows[0]["snapshot_month"])
    output = DATA_DIR / f"{snapshot_month}.csv"

    field_order = [
        "snapshot_month",
        "region_level",
        "region_name",
        "indicator",
        "current_value",
        "current_signal",
        "prev_1m_signal",
        "prev_2m_value",
        "prev_2m_signal",
        "collected_at",
    ]

    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
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
            rows = await collect_rows(page, selectors)
            output = save_rows(rows)
            print(f"Saved snapshot: {output} rows={len(rows)}")
        except Exception:
            screenshot = ARTIFACT_DIR / "collector_failure.png"
            await page.screenshot(path=str(screenshot), full_page=True)
            print(f"Saved failure screenshot: {screenshot}")
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
