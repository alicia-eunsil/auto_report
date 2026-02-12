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
RISK_URL = f"{BASE_URL}/forecast/risk"
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

    await click_resilient(trigger, timeout=15000)
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


def _locator_candidates(page: Page, selector: str) -> list[Locator]:
    locators = [page.locator(selector).first]
    for frame in page.frames:
        locators.append(frame.locator(selector).first)
    return locators


async def has_any_selector(page: Page, candidates: list[str]) -> bool:
    for sel in candidates:
        for loc in _locator_candidates(page, sel):
            if await loc.count() > 0:
                return True
    return False


async def switch_to_new_page_if_opened(page: Page, previous_page_count: int) -> Page:
    pages = page.context.pages
    if len(pages) <= previous_page_count:
        return page
    new_page = pages[-1]
    try:
        await new_page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass
    return new_page


def selector_candidates(selectors: dict[str, Any], key: str, defaults: list[str]) -> list[str]:
    out: list[str] = []
    configured = selectors.get(key)
    if isinstance(configured, str) and configured.strip():
        out.append(configured.strip())
    elif isinstance(configured, list):
        out.extend([str(v).strip() for v in configured if str(v).strip()])

    for sel in defaults:
        if sel not in out:
            out.append(sel)
    return out


async def click_first_available(
    page: Page,
    candidates: list[str],
    *,
    timeout: int = 15000,
    label: str,
) -> str:
    last_error: Exception | None = None
    for _ in range(2):
        for sel in candidates:
            for loc in _locator_candidates(page, sel):
                if await loc.count() == 0:
                    continue
                try:
                    if not await loc.is_visible():
                        continue
                except Exception:
                    continue
                try:
                    await click_resilient(loc, timeout=timeout)
                    return sel
                except Exception as exc:
                    last_error = exc
        await page.wait_for_timeout(700)

    if last_error:
        raise RuntimeError(f"Failed to click {label}. candidates={candidates}") from last_error
    raise RuntimeError(f"No element found for {label}. candidates={candidates}")


async def click_exact_text_any_frame(page: Page, text: str) -> str | None:
    js = """
    (raw) => {
      const norm = (v) => (v || "").replace(/\\s+/g, "").trim();
      const target = norm(raw);
      if (!target) return null;

      const isVisible = (el) => {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      };

      const nodes = Array.from(document.querySelectorAll("button,a,[role='button'],li,div,span"));
      for (const el of nodes) {
        if (!isVisible(el)) continue;
        if (norm(el.textContent) !== target) continue;

        const rowLike =
          el.closest(".risk-metric-list-row") ||
          el.closest("[class*='risk-metric-list-row']") ||
          el.closest(".risk-metric-item");

        const clickable =
          el.closest("button,a,[role='button'],li,[tabindex]") ||
          rowLike ||
          (el.hasAttribute("onclick") ? el : null) ||
          el;

        clickable.click();
        const tag = clickable.tagName ? clickable.tagName.toLowerCase() : "unknown";
        const cls = clickable.className ? String(clickable.className).trim().slice(0, 60) : "";
        return cls ? `${tag}.${cls}` : tag;
      }
      return null;
    }
    """
    for frame in page.frames:
        try:
            matched = await frame.evaluate(js, text)
        except Exception:
            continue
        if matched:
            return str(matched)
    return None


async def login(page: Page, selectors: dict[str, Any], user: str, password: str) -> None:
    async def accept_dialog(dialog: Dialog) -> None:
        await dialog.accept()

    await page.goto(BASE_URL, wait_until="domcontentloaded")
    page.on("dialog", accept_dialog)

    await click_resilient(page.locator(selectors["observer_login_button"]).first, timeout=15000)

    login_scope = await resolve_login_scope(page, selectors)

    await login_scope.locator(selectors["login_user_input"]).fill(user, timeout=15000)
    await login_scope.locator(selectors["login_password_input"]).fill(password, timeout=15000)
    await click_resilient(login_scope.locator(selectors["login_submit_button"]), timeout=15000)

    # Some environments show a custom confirmation modal instead of JS alert.
    popup_confirm_selector = selectors.get("login_popup_confirm_button")
    if popup_confirm_selector:
        popup_confirm_button = page.locator(popup_confirm_selector).first
        try:
            await popup_confirm_button.wait_for(state="visible", timeout=4000)
            await click_resilient(popup_confirm_button, timeout=4000)
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
    await click_resilient(page.locator(level_selector).first, timeout=15000)
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
    await click_resilient(page.locator(target_selector).first, timeout=15000)
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


def indicator_variants(indicator: str) -> list[str]:
    base = indicator.strip()
    variants = [base]

    if base.endswith("수") and len(base) > 1:
        variants.append(base[:-1])
    if " " in base:
        variants.append(base.replace(" ", ""))

    if base == "사업장 성립":
        variants.extend(["사업장성립", "성립"])
    if base == "사업장 소멸":
        variants.extend(["사업장소멸", "소멸"])

    out: list[str] = []
    for v in variants:
        if v and v not in out:
            out.append(v)
    return out


async def click_indicator(page: Page, selectors: dict[str, Any], indicator: str) -> None:
    button_tpl = selectors.get("indicator_button_by_text")
    if not button_tpl:
        raise ConfigError("Missing selector: indicator_button_by_text")

    variants = indicator_variants(indicator)

    for name in variants:
        exact_used = await click_exact_text_any_frame(page, name)
        if exact_used:
            print(f"INFO: indicator clicked {indicator} via exact-text target: {exact_used}")
            await page.wait_for_timeout(500)
            return

    candidates: list[str] = []

    for name in variants:
        candidates.append(button_tpl.replace("{name}", name))
        candidates.extend(
            [
                f"button:text-is('{name}')",
                f"[role='button']:text-is('{name}')",
                f"a:text-is('{name}')",
                f"li:text-is('{name}')",
            ]
        )

    # Keep order while dropping duplicates.
    deduped: list[str] = []
    for c in candidates:
        if c not in deduped:
            deduped.append(c)

    used = await click_first_available(page, deduped, timeout=15000, label=f"indicator:{indicator}")
    print(f"INFO: indicator clicked {indicator} via selector: {used}")
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
    card_candidates = selector_candidates(
        selectors,
        "region_cards_selector",
        [
            "div.panel-detail-cont-row.main:has(.panel-detail-cont-row-header-tit)",
            "div.panel-detail-cont-row.main",
            ".panel-detail-cont-row.main",
            "div.panel-detail-cont > div.panel-detail-cont-row:has(.panel-detail-cont-row-header-tit)",
        ],
    )
    cards = None
    used_card_selector = ""
    for _ in range(20):
        for sel in card_candidates:
            loc = page.locator(sel)
            if await loc.count() > 0:
                cards = loc
                used_card_selector = sel
                break
        if cards is not None:
            break
        await page.wait_for_timeout(600)

    if cards is None:
        return []

    if used_card_selector != selectors.get("region_cards_selector", ""):
        print(f"INFO: card selector fallback used for {source_level}/{indicator}: {used_card_selector}")

    for i in range(await cards.count()):
        card = cards.nth(i)
        name_loc = card.locator(selectors["card_region_name"]).first
        if await name_loc.count() == 0:
            continue
        try:
            region_name = (await name_loc.inner_text(timeout=2000)).strip()
        except Exception:
            continue
        if not region_name:
            continue
        region_token = region_name.replace(" ", "").replace("\xa0", "")

        value_loc = card.locator(selectors["card_current_value"]).first
        current_value_raw = ""
        if await value_loc.count() > 0:
            try:
                current_value_raw = (await value_loc.inner_text(timeout=2000)).strip()
            except Exception:
                current_value_raw = ""

        signals = card.locator(selectors["card_signal_items"])
        if await signals.count() < 3:
            continue

        s0 = signals.nth(0)  # oldest (t-2)
        s1 = signals.nth(1)  # previous month (t-1)
        s2 = signals.nth(2)  # current month (t)
        try:
            s0_text = (await s0.inner_text(timeout=2000)).strip()
            s1_text = (await s1.inner_text(timeout=2000)).strip()
            s2_text = (await s2.inner_text(timeout=2000)).strip()
        except Exception:
            continue
        current_signal = normalize_signal(s2_text, await s2.get_attribute("class") or "")
        prev_1m_signal = normalize_signal(s1_text, await s1.get_attribute("class") or "")
        prev_2m_signal = normalize_signal(s0_text, await s0.get_attribute("class") or "")

        if not current_signal or not prev_1m_signal or not prev_2m_signal:
            continue

        level_key = source_level
        if source_level == "province" and region_name == "전국":
            level_key = "national"
        if source_level == "gyeonggi_city" and region_token in {"경기도", "경기도전체", "전체"}:
            print(f"INFO: skip aggregate card in city level: {region_name}")
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
                "prev_2m_value": "",
                "prev_2m_signal": prev_2m_signal,
                "collected_at": ctx.collected_at,
            }
        )

    return rows


def region_rows_signature(rows: list[dict[str, str]], take: int = 8) -> tuple[str, ...]:
    if not rows:
        return tuple()
    key_rows = sorted(
        rows,
        key=lambda r: (r["region_name"], r["indicator"]),
    )[:take]
    return tuple(
        f"{r['region_name']}|{r['current_value']}|{r['current_signal']}|{r['prev_1m_signal']}|{r['prev_2m_signal']}"
        for r in key_rows
    )


async def wait_for_card_refresh(
    page: Page,
    selectors: dict[str, Any],
    previous_signature: tuple[str, ...] | None,
    source_level: str,
    indicator: str,
) -> None:
    # Wait for cards to render and, when possible, change from the previous indicator snapshot.
    for _ in range(25):
        try:
            rows = await extract_region_cards(page, selectors, ExtractionContext("", ""), source_level, indicator)
        except Exception:
            await page.wait_for_timeout(500)
            continue
        sig = region_rows_signature(rows)
        if not sig:
            await page.wait_for_timeout(500)
            continue
        if previous_signature is None or sig != previous_signature:
            return
        await page.wait_for_timeout(500)

    # Do not silently continue: stale data across indicators leads to misleading reports.
    raise RuntimeError(f"Card data did not refresh after indicator click: {source_level}/{indicator}")


async def collect_rows_from_cards(page: Page, selectors: dict[str, Any], ctx: ExtractionContext) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    for level_key in ["province", "gyeonggi_city"]:
        await click_region_level(page, selectors, level_key)
        names = indicator_names(selectors)
        previous_signature: tuple[str, ...] | None = None

        # Seed a known baseline metric for each level to avoid stale first-indicator reads.
        if len(names) > 1:
            baseline_indicator = names[-1]
            await click_indicator(page, selectors, baseline_indicator)
            await wait_for_card_refresh(page, selectors, None, level_key, baseline_indicator)
            baseline_rows = await extract_region_cards(page, selectors, ctx, level_key, baseline_indicator)
            if not baseline_rows:
                raise RuntimeError(f"No region cards extracted for baseline {level_key}/{baseline_indicator}")
            previous_signature = region_rows_signature(baseline_rows)

        for indicator in names:
            await click_indicator(page, selectors, indicator)
            await wait_for_card_refresh(page, selectors, previous_signature, level_key, indicator)
            rows = await extract_region_cards(page, selectors, ctx, level_key, indicator)
            if not rows:
                raise RuntimeError(f"No region cards extracted for {level_key}/{indicator}")
            previous_signature = region_rows_signature(rows)
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

    # Guard against stale scrape where indicator switch silently fails.
    stale_regions: list[str] = []
    by_region: dict[tuple[str, str], set[tuple[str, str, str, str]]] = {}
    for r in rows:
        key = (r["region_level"], r["region_name"])
        sig = (
            str(r.get("current_value", "")),
            str(r.get("current_signal", "")),
            str(r.get("prev_1m_signal", "")),
            str(r.get("prev_2m_signal", "")),
        )
        by_region.setdefault(key, set()).add(sig)

    for (level, region), signatures in sorted(by_region.items()):
        if len(signatures) <= 1:
            stale_regions.append(f"{level}/{region}")

    stale_threshold = max(3, int(len(by_region) * 0.3)) if by_region else 9999
    if len(stale_regions) >= stale_threshold:
        sample = ", ".join(stale_regions[:10])
        missing_msgs.append(
            "indicator-switch check failed; too many regions have identical values across all indicators: "
            + sample
        )

    if missing_msgs:
        joined = "\n - ".join(missing_msgs[:20])
        raise RuntimeError(
            "Collection incomplete. Check selectors and region mapping.\n - " + joined
        )


async def collect_rows(page: Page, selectors: dict[str, Any]) -> list[dict[str, str]]:
    await page.goto(RISK_URL, wait_until="domcontentloaded")
    try:
        await page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    await page.wait_for_timeout(1200)

    employment_candidates = selector_candidates(
        selectors,
        "employment_tab_button",
        [
            "text=고용보험",
            "button:has-text('고용보험')",
            "button[role='tab']:has-text('고용보험')",
            "a:has-text('고용보험')",
            "li:has-text('고용보험')",
            "div:has-text('고용보험')",
        ],
    )

    found_employment = False
    for _ in range(20):
        if await has_any_selector(page, employment_candidates):
            found_employment = True
            break
        await page.wait_for_timeout(1000)

    if not found_employment:
        raise RuntimeError(
            f"No element found for employment tab after loading risk page. "
            f"url={page.url} candidates={employment_candidates}"
        )

    await click_first_available(page, employment_candidates, timeout=20000, label="employment tab")
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
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
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
