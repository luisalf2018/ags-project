"""
Core logic: read an Order Confirmation workbook, read the persisted
Expiration Dates list, and populate an "Expiration" column on the order
confirmation based on matching item numbers.

Kept dependency-free (openpyxl only) so the whole app installs with one
pip command.
"""
from __future__ import annotations

import datetime
from typing import Optional

import openpyxl

# --- Header name recognition -------------------------------------------------
# The real files this app was built from use these exact headers, but we
# match loosely (case/spacing-insensitive, substring-based) so small
# variations in future exports don't break the app.

ITEM_NO_HEADER_CANDIDATES = ["item no", "item number", "item#", "item #"]
EXPIRATION_HEADER_CANDIDATES = ["expiration date", "expiration", "exp date"]

EXP_LIST_KEY_HEADER_CANDIDATES = ["c&s code", "cs code", "item no", "item number", "code"]
EXP_LIST_DATE_HEADER_CANDIDATES = ["expiration date", "expiration"]

MAX_HEADER_SEARCH_ROWS = 30  # how far down to look for the header row
NOT_FOUND_VALUE = "N/A"


class MatcherError(Exception):
    """Raised for anything that should surface as a friendly error in the UI."""


def _norm(s) -> str:
    return str(s).strip().lower() if s is not None else ""


def _find_header_row_and_cols(ws, key_candidates, extra_candidates_map=None):
    """
    Scan the first MAX_HEADER_SEARCH_ROWS rows for a row that contains a
    cell matching one of key_candidates. Returns (header_row_index,
    {candidate_name: col_index}) using 1-based openpyxl indices.
    extra_candidates_map: dict of other header name lists to also resolve
    on that same row, e.g. {"expiration": EXPIRATION_HEADER_CANDIDATES}.
    """
    extra_candidates_map = extra_candidates_map or {}
    max_row = min(ws.max_row, MAX_HEADER_SEARCH_ROWS)
    for row_idx in range(1, max_row + 1):
        row_values = {}
        for col_idx in range(1, ws.max_column + 1):
            val = _norm(ws.cell(row=row_idx, column=col_idx).value)
            if val:
                row_values[col_idx] = val

        key_col = None
        for col_idx, val in row_values.items():
            if any(cand in val for cand in key_candidates):
                key_col = col_idx
                break

        if key_col is not None:
            result = {"_row": row_idx, "_key_col": key_col}
            for name, candidates in extra_candidates_map.items():
                found = None
                for col_idx, val in row_values.items():
                    if any(cand in val for cand in candidates):
                        found = col_idx
                        break
                result[name] = found
            return result

    return None


def _clean_item_key(value):
    """Normalize an item-number-like cell value to a stable dict key."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    s = str(value).strip()
    if not s:
        return None
    # Strip a trailing ".0" some spreadsheets introduce when a number
    # round-trips through text.
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def load_expiration_map(expiration_list_path: str) -> dict:
    """
    Reads the persisted expiration-dates workbook and returns
    {item_key: earliest_expiration_datetime_or_None}.
    """
    wb = openpyxl.load_workbook(expiration_list_path, data_only=True, read_only=True)
    ws = wb.worksheets[0]

    header = _find_header_row_and_cols(
        ws,
        EXP_LIST_KEY_HEADER_CANDIDATES,
        {"date": EXP_LIST_DATE_HEADER_CANDIDATES},
    )
    if header is None or header.get("date") is None:
        raise MatcherError(
            "Couldn't find the item-code and expiration-date columns in the "
            "expiration dates list. Expected headers like 'C&S Code' and "
            "'Expiration Date'."
        )

    key_col = header["_key_col"]
    date_col = header["date"]
    start_row = header["_row"] + 1

    best: dict = {}
    for row in ws.iter_rows(min_row=start_row, values_only=True):
        key_raw = row[key_col - 1] if key_col - 1 < len(row) else None
        key = _clean_item_key(key_raw)
        if key is None:
            continue
        date_val = row[date_col - 1] if date_col - 1 < len(row) else None
        if isinstance(date_val, datetime.datetime):
            d = date_val
        elif isinstance(date_val, datetime.date):
            d = datetime.datetime(date_val.year, date_val.month, date_val.day)
        else:
            d = None

        if key not in best:
            best[key] = d
        elif d is not None and (best[key] is None or d < best[key]):
            best[key] = d

    wb.close()
    return best


def process_order_confirmation(
    order_path: str,
    expiration_map: dict,
    output_path: str,
) -> dict:
    """
    Loads the order confirmation, fills in the Expiration column using
    expiration_map, and saves the result to output_path. Nothing else in
    the file is modified. Returns a small stats dict for the UI.
    """
    wb = openpyxl.load_workbook(order_path, data_only=False)
    ws = wb.worksheets[0]

    header = _find_header_row_and_cols(
        ws,
        ITEM_NO_HEADER_CANDIDATES,
        {"expiration": EXPIRATION_HEADER_CANDIDATES},
    )
    if header is None:
        raise MatcherError(
            "Couldn't find an 'Item No' column in the order confirmation."
        )

    header_row = header["_row"]
    item_col = header["_key_col"]
    exp_col = header["expiration"]

    if exp_col is None:
        # No Expiration column yet in this file -- add one right after the
        # last used column.
        exp_col = ws.max_column + 1
        ws.cell(row=header_row, column=exp_col, value="Expiration")

    # Copy a representative date number format from elsewhere in the sheet
    # if we can find one, so new values look consistent with the template.
    date_format = "mm-dd-yy"

    matched = 0
    not_found = 0
    row_idx = header_row + 1
    while row_idx <= ws.max_row:
        item_cell = ws.cell(row=row_idx, column=item_col)
        key = _clean_item_key(item_cell.value)
        if key is None:
            # First blank/non-numeric Item No after the header marks the
            # end of the line-item table (totals/footer follows).
            break

        exp_cell = ws.cell(row=row_idx, column=exp_col)
        date_val = expiration_map.get(key)
        if date_val is not None:
            exp_cell.value = date_val
            exp_cell.number_format = date_format
            matched += 1
        else:
            exp_cell.value = NOT_FOUND_VALUE
            not_found += 1

        row_idx += 1

    wb.save(output_path)
    wb.close()

    return {
        "matched": matched,
        "not_found": not_found,
        "total": matched + not_found,
    }


# --- Expiration lists from several distributors -------------------------------
# Lists come from more than one distributor and not every workbook puts the per-item dates on its first
# tab (an inventory check can have a summary tab, a lot-by-lot detail tab and a notes tab). The loader
# below looks at every tab, so nothing is hard-coded to a tab name or position.

LIST_ITEM_HEADERS = ["c&s code", "cs code", "item number", "item no", "item#", "item #", "sku", "code"]
# (header text, quality): an exact "Expiration Date" is the real thing; anything that only says
# "Expiration" may be a derived column (e.g. a summary tab's "Best Expiration" is the FURTHEST date).
LIST_DATE_HEADERS = [("expiration date", 3), ("exp date", 3), ("expiry date", 3), ("expiration", 1), ("expiry", 1)]
DATE_TEXT_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%m-%d-%Y", "%d-%b-%Y")


def _as_datetime(value):
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        for fmt in DATE_TEXT_FORMATS:
            try:
                return datetime.datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
    return None


def _find_list_columns(rows):
    """First row (within the header search window) holding both an item-code column and an expiration-date
    column. Returns (row_index_0_based, item_col_0_based, date_col_0_based, date_quality) or None."""
    for row_idx, row in enumerate(rows):
        cells = [_norm(v) for v in row]
        date_choice = None  # (quality, col) - best quality wins, leftmost on a tie
        for col, text in enumerate(cells):
            if not text:
                continue
            for header, quality in LIST_DATE_HEADERS:
                if header in text and "best" not in text:
                    if date_choice is None or quality > date_choice[0]:
                        date_choice = (quality, col)
                    break
        if date_choice is None:
            continue
        for candidate in LIST_ITEM_HEADERS:  # most specific header first
            item_col = next((col for col, text in enumerate(cells) if text and candidate in text), None)
            if item_col is not None and item_col != date_choice[1]:
                return row_idx, item_col, date_choice[1], date_choice[0]
    return None


def load_expiration_map_any_tab(source) -> tuple[dict, str]:
    """Reads an expiration list (a path or file-like object) from whichever tab holds item codes together
    with expiration dates, and returns ({item_key: earliest_expiration_datetime}, tab_name). A code that
    appears several times (one row per lot) keeps its CLOSEST date. When more than one tab qualifies, the
    one whose date column is a plain "Expiration Date" (not a derived one like "Best Expiration") and that
    has the most dated rows is used."""
    wb = openpyxl.load_workbook(source, data_only=True, read_only=True)
    try:
        best = None  # (quality, dated_rows, tab name, map)
        for ws in wb.worksheets:
            all_rows = list(ws.iter_rows(values_only=True))
            found = _find_list_columns(all_rows[:MAX_HEADER_SEARCH_ROWS])
            if found is None:
                continue
            header_idx, item_col, date_col, quality = found
            earliest: dict = {}
            for row in all_rows[header_idx + 1:]:
                key = _clean_item_key(row[item_col]) if item_col < len(row) else None
                date = _as_datetime(row[date_col]) if date_col < len(row) else None
                if key is None or date is None:
                    continue
                if key not in earliest or date < earliest[key]:
                    earliest[key] = date
            if earliest and (best is None or (quality, len(earliest)) > (best[0], best[1])):
                best = (quality, len(earliest), ws.title.strip(), earliest)
    finally:
        wb.close()
    if best is None:
        raise MatcherError(
            "Couldn't find a tab with item codes and expiration dates in that expiration dates list."
        )
    return best[3], best[2]


def merge_expiration_maps(maps) -> dict:
    """Several lists active at once: an item found in more than one (or listed several times) gets its
    CLOSEST expiration date."""
    merged: dict = {}
    for expiration_map in maps:
        for key, date in expiration_map.items():
            if date is not None and (key not in merged or date < merged[key]):
                merged[key] = date
    return merged
