import base64
import io
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv, set_key
from openai import OpenAI
from PIL import Image, ImageOps

load_dotenv()

st.set_page_config(page_title="AI Handwritten Qty Extractor", layout="wide")
st.title("AI Handwritten Qty Extractor")
st.caption("Upload photos of order sheets. Rows with a handwritten number will be pulled out into a table you can export.")

ENV_PATH = Path(__file__).parent.parent / ".env"

# Cloud deployments (no shared local drive to watch) set AGS_CLOUD_MODE=1 to hide the
# folder-monitoring tab, and AGS_DATA_DIR to point item memory / usage reports at a
# mounted persistent volume instead of the app folder.
CLOUD_MODE = os.getenv("AGS_CLOUD_MODE", "").strip().lower() in ("1", "true", "yes")
DATA_DIR = Path(os.environ["AGS_DATA_DIR"]) if os.getenv("AGS_DATA_DIR") else Path(__file__).parent

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key or api_key == "paste-your-key-here":
    st.error(
        "No API key found. Open the .env file in the project folder and paste your OpenAI API key "
        "after OPENAI_API_KEY=, or use the ⚙️ Settings tab once the app is running."
    )
    st.stop()

client = OpenAI(api_key=api_key)


def apply_new_api_key(new_key: str) -> None:
    """Saves the key to .env (so it persists across restarts) and swaps the live client
    immediately, so a key change takes effect in this running session without a restart."""
    global client
    new_key = new_key.strip()
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_PATH.exists():
        ENV_PATH.write_text("")
    set_key(str(ENV_PATH), "OPENAI_API_KEY", new_key)
    os.environ["OPENAI_API_KEY"] = new_key
    client = OpenAI(api_key=new_key)

CHEAP_MODEL = "gpt-5.6-luna"
PREMIUM_MODEL = "gpt-5.6-sol"

EXPORT_COLUMN_RENAME = {"handwritten_number": "Qty"}
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def trigger_browser_download(files: list[tuple[bytes, str, str]]) -> None:
    """Fires browser downloads with no click needed, via hidden auto-clicked links -
    Streamlit's download_button can't do this itself since it only ever acts on a real click.
    Clicks are staggered so browsers treat them as separate downloads rather than dropping
    the later ones."""
    links, clicks = [], []
    for i, (file_bytes, filename, mime_type) in enumerate(files):
        b64 = base64.b64encode(file_bytes).decode()
        links.append(f'<a id="auto-dl-{i}" href="data:{mime_type};base64,{b64}" download="{filename}"></a>')
        clicks.append(f'setTimeout(function(){{document.getElementById("auto-dl-{i}").click();}},{i * 800});')
    components.html("".join(links) + "<script>" + "".join(clicks) + "</script>", height=0)


def build_upload_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Just item_no + Qty, ready to upload into the ordering system. Quantities become real
    numbers where possible (a text-typed "4" trips up spreadsheet imports); anything that isn't
    a clean number is left exactly as read rather than dropped."""
    upload = df.reindex(columns=["item_no", "Qty"]).copy()
    numeric = pd.to_numeric(upload["Qty"], errors="coerce")
    upload["Qty"] = numeric.where(numeric.notna(), upload["Qty"])
    upload["Qty"] = upload["Qty"].map(lambda v: int(v) if isinstance(v, float) and v.is_integer() else v)
    return upload


WATCH_POLL_SECONDS = 8
WATCH_CONFIG_PATH = Path(__file__).parent / "watch_config.json"
PENDING_DIRNAME = "_pending_review"
SV_CODES = [f"SV{n}" for n in range(1, 14)]

SYSTEM_PROMPT = """You are extracting data from a photo of a printed inventory/order sheet.
The sheet has printed columns such as Brand, Pack, Size, Item No., Old Item, and Item Description.
Some rows have a NUMBER WRITTEN BY HAND in the margin, on the far left or far right of the row.

This image may be only the TOP or BOTTOM portion of a taller sheet, cropped with some overlap so nothing is missed.
If a row is cut off at the very top or bottom edge of the image (less than half of that row's text is visible),
SKIP it entirely - it will be fully visible in the neighboring crop. Only include rows that are more than half visible.

Work through the ENTIRE image methodically, top row to bottom row, left column of the table to right edge of the page.
Do not stop early or sample rows - every single printed row on the page must be checked for a handwritten mark
before you finalize your answer. Handwritten marks can be small, faint, or partially overlapping printed text -
look closely at the full margin on both sides of every row.

Some rows are visually TALL because the item description wraps onto its own line below the item-code line,
which can make a handwritten mark sit low in that row's margin, close to the NEXT row's line. Match a mark to
the row whose printed content (item code, description) it sits beside as a whole block, not just whichever
single printed line happens to be nearest to it vertically - a mark should not be shifted down onto the
following row just because it is drawn near the bottom of a tall row's cell. Pay special attention to the
VERY FIRST row on the page or crop: because there is no preceding row for context, a mark belonging to that
first row is the one most likely to be mistakenly shifted onto the second row instead. Before finalizing,
explicitly double check whether the first row has a handwritten mark that may have been attributed to the
second row by mistake.

SOME sheets are a different style: instead of a printed table with a handwritten quantity mark in the margin,
BOTH the item number AND the quantity are handwritten together as repeating pairs (item number, then quantity)
written straight onto the page. These handwritten-pair sheets are very often laid out in MULTIPLE side-by-side
columns - commonly 2 or 3 - rather than one single list running down the page. If you see this style, you MUST
scan the FULL WIDTH of the page and check every column block, not just the first one or two - entries keep
going in additional columns further to the right, and it is a serious error to stop after the first column(s).
For each handwritten pair, put the handwritten item number in item_no and the handwritten quantity in
handwritten_number; leave brand/pack/size/old_item/description empty since there is no printed table providing
those fields.

Find only the rows/pairs that have a handwritten number. Ignore every row that has no handwritten mark.

IMPORTANT: every handwritten mark on this sheet is a DIGIT (0-9), never a slash, letter, or other symbol.
A handwritten mark that looks like a forward slash "/" or a lone diagonal stroke is always the digit "1"
written in a stylized way - transcribe it as "1", not as "/".

If a code (item_no or old_item) wraps across two printed lines within the same cell because the column is
narrow, concatenate it into ONE continuous code with NO space and no line break - e.g. "S122957" on one
line and "52" on the next line beneath it in the same cell means the code is "S12295752", not "S122957 52".

For each marked row, return:
- brand
- pack
- size
- item_no
- old_item
- description
- handwritten_number (the handwritten value, as text)
- mark_side ("left" or "right")
- confidence: this field must be EXACTLY the string "high" or EXACTLY the string "low" - no other value
  (not "medium", not "high", not "certain", not a number) is allowed.
- appears_altered: true or false. Set this to true if the mark shows signs of being CROSSED OUT, SCRATCHED
  OVER, STRUCK THROUGH, or otherwise VOIDED - for example a digit with a line drawn through it, an X drawn
  over it, or messy overlapping strokes that look like correction rather than a single clean digit. This is
  different from ordinary messy handwriting: it specifically means the mark looks like someone tried to
  cancel or void it. Still give your best-guess handwritten_number even when appears_altered is true.
- row_bbox: the bounding box of this entire row (printed text plus the handwritten mark) within the image,
  as [x_min, y_min, x_max, y_max], each a fraction from 0.0 to 1.0 of the image width/height
  (0,0 is the top-left corner, 1,1 is the bottom-right corner). Be as precise as possible - a person will
  crop exactly this region out of the image to double-check your reading, so it must tightly contain the row.

Set confidence to "low" whenever you are not fully certain of the handwritten value - for example if it is faint,
smudged, partially cut off, overlapping printed text, or could plausibly be confused with a different digit
(1 vs 7, 6 vs 0, 3 vs 8, 5 vs 6, etc). Otherwise set it to "high" (never anything else). Do not default everything
to "high" - only use it when you are genuinely confident. A mark that appears_altered should virtually always
also be confidence "low", since a voided/crossed-out mark is not a value you can be truly confident in.

Respond ONLY with JSON in this exact shape:
{"total_rows_on_page": 0, "total_marked_rows_found": 0, "items": [{"brand": "", "pack": "", "size": "", "item_no": "", "old_item": "", "description": "", "handwritten_number": "", "mark_side": "", "confidence": "high", "appears_altered": false, "row_bbox": [0.0, 0.0, 1.0, 0.0]}]}

total_rows_on_page must be your count of every printed row visible on the page (marked or not).
total_marked_rows_found must equal the number of entries in items.
If a printed field is missing or unreadable, use an empty string for it. If there are no marked rows, return {"total_rows_on_page": N, "total_marked_rows_found": 0, "items": []}.
"""


OVERLAP_FRACTION = 0.25


def encode_pil_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def split_top_bottom(image: Image.Image) -> list[tuple[Image.Image, int]]:
    """Returns (crop_image, y_offset) pairs - y_offset is where this crop starts within the
    original full image, in pixels, so a row's position can later be compared across crops."""
    width, height = image.size
    overlap = int(height * OVERLAP_FRACTION)
    mid = height // 2
    top_y0 = 0
    bottom_y0 = max(0, mid - overlap)
    top = image.crop((0, top_y0, width, min(height, mid + overlap)))
    bottom = image.crop((0, bottom_y0, width, height))
    return [(top, top_y0), (bottom, bottom_y0)]


def merge_overlap_duplicates(items: list[dict]) -> list[dict]:
    """Rows sitting right at the top/bottom crop boundary can get caught by both crops - once
    cleanly, once cut off with a possibly garbled reading (including a misread item code that
    would slip past text-based duplicate checks). Detect these by physical Y-position overlap
    in the original photo instead of by text, and keep the more trustworthy of the two."""
    dropped = set()
    kept = []
    for i, item_a in enumerate(items):
        if i in dropped:
            continue
        range_a = item_a.get("_abs_bbox")
        match_j = None
        if range_a:
            for j in range(i + 1, len(items)):
                if j in dropped:
                    continue
                item_b = items[j]
                if item_b.get("_crop_side") == item_a.get("_crop_side"):
                    continue  # only cross-crop pairs can be this kind of duplicate
                range_b = item_b.get("_abs_bbox")
                if not range_b:
                    continue
                y0, y1 = max(range_a[0], range_b[0]), min(range_a[1], range_b[1])
                intersection = max(0.0, y1 - y0)
                union = max(range_a[1], range_b[1]) - min(range_a[0], range_b[0])
                if union > 0 and intersection / union > 0.3:
                    match_j = j
                    break
        if match_j is None:
            kept.append(item_a)
            continue
        item_b = items[match_j]
        dropped.add(match_j)
        a_conf = str(item_a.get("confidence", "high")).lower()
        b_conf = str(item_b.get("confidence", "high")).lower()
        if a_conf == "high" and b_conf != "high":
            kept.append(item_a)
        elif b_conf == "high" and a_conf != "high":
            kept.append(item_b)
        elif item_a.get("_edge_distance", 0) >= item_b.get("_edge_distance", 0):
            kept.append(item_a)
        else:
            kept.append(item_b)
    return kept


def dedupe_items(items: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for item in items:
        key = (
            str(item.get("item_no", "")).strip().lower(),
            str(item.get("description", "")).strip().lower(),
            str(item.get("handwritten_number", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def is_suspiciously_high(value, threshold: float = 10) -> bool:
    try:
        return float(str(value).strip()) >= threshold
    except (ValueError, TypeError):
        return False


def reads_as_seven(value) -> bool:
    """A handwritten 7 is easily misread from (or as) a handwritten 1. Misreading a 7 as a 1
    is low-stakes, but the reverse - a real 1 coming out as 7 - would inflate an order, so every
    7 reading gets a mandatory human look regardless of the model's confidence."""
    try:
        return float(str(value).strip()) == 7
    except (ValueError, TypeError):
        return False


def resolve_duplicate_item_codes(items: list[dict]) -> list[dict]:
    """Purchase orders shouldn't have the same item code twice. If the same item_no shows up
    more than once in a batch: same handwritten value on every instance -> it's a processing
    duplicate, keep just one silently. Different values -> genuine conflict, flag all instances."""
    order = []
    groups: dict[str, list[dict]] = {}
    no_code_items = []
    for item in items:
        code = str(item.get("item_no", "")).strip().lower()
        if not code:
            no_code_items.append(item)
            continue
        if code not in groups:
            groups[code] = []
            order.append(code)
        groups[code].append(item)

    result = []
    for code in order:
        group = groups[code]
        if len(group) == 1:
            result.append(group[0])
            continue
        values = {str(g.get("handwritten_number", "")).strip().lower() for g in group}
        if len(values) == 1:
            result.append(group[0])
        else:
            for g in group:
                reasons = [r for r in [g.get("review_reason", "")] if r]
                reasons.append(
                    f"item code {g.get('item_no', '')} appears more than once with different "
                    "handwritten values - please verify"
                )
                g["needs_review"] = True
                g["review_reason"] = "; ".join(reasons)
                result.append(g)
    result.extend(no_code_items)
    return result


def find_conflicting_keys(items: list[dict]) -> set:
    values_by_row = {}
    for item in items:
        row_key = (
            str(item.get("item_no", "")).strip().lower(),
            str(item.get("description", "")).strip().lower(),
        )
        values_by_row.setdefault(row_key, set()).add(str(item.get("handwritten_number", "")).strip().lower())
    return {row_key for row_key, values in values_by_row.items() if len(values) > 1}


def crop_region(image: Image.Image, bbox, pad_frac: float = 0.20) -> Image.Image | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    pad_x = (x1 - x0) * pad_frac + 0.03
    pad_y = (y1 - y0) * pad_frac + 0.03
    x0, x1 = max(0.0, x0 - pad_x), min(1.0, x1 + pad_x)
    y0, y1 = max(0.0, y0 - pad_y), min(1.0, y1 + pad_y)
    width, height = image.size
    box = (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return image.crop(box)


def crop_context_region(image: Image.Image, abs_y_range, min_pad_frac: float = 0.10) -> Image.Image | None:
    """Crop a generous full-width window from the ORIGINAL photo, centered on a row's actual
    known position - not bounded by which top/bottom crop-half it happened to be detected in.
    That's what makes this reliable as a fallback: it can't cut the target row off at a crop
    seam the way re-showing the source crop-half can."""
    if not abs_y_range:
        return None
    y0, y1 = sorted((max(0.0, min(1.0, abs_y_range[0])), max(0.0, min(1.0, abs_y_range[1]))))
    row_height = y1 - y0
    pad = max(min_pad_frac, row_height * 1.5)
    y0, y1 = max(0.0, y0 - pad), min(1.0, y1 + pad)
    width, height = image.size
    box = (0, int(y0 * height), width, int(y1 * height))
    if box[3] <= box[1]:
        return None
    return image.crop(box)


def order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_deskew(image: Image.Image) -> Image.Image | None:
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blurred, 50, 150), np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img_cv.shape[0] * img_cv.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.3 * img_area:
            rect = order_corners(approx.reshape(4, 2).astype("float32"))
            (tl, tr, br, bl) = rect
            max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            if max_width < 100 or max_height < 100:
                continue
            dst = np.array(
                [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
                dtype="float32",
            )
            matrix = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_cv, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)
            return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return None


def estimate_tilt_angle(image: Image.Image) -> float:
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]

    # search only the central region - photo edges/corners are where folded page corners,
    # torn edges, and background clutter live, and those produce diagonal lines that have
    # nothing to do with the table's actual tilt
    margin_x, margin_y = int(width * 0.12), int(height * 0.12)
    central = img_cv[margin_y : height - margin_y, margin_x : width - margin_x]

    gray = cv2.cvtColor(central, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=150, minLineLength=central.shape[1] // 3, maxLineGap=20
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = np.asarray(line).reshape(4)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # a handheld photo can genuinely be tilted quite a lot - the real defense against
        # noise (folded corners, background clutter) is the central-region restriction plus
        # the consistency check below, not a narrow angle cap
        if -40 < angle < 40:
            angles.append(angle)

    # require several roughly-agreeing lines before trusting the estimate at all; a couple
    # of stray lines (or lines that disagree wildly) are noise, not evidence of real tilt
    if len(angles) < 4:
        return 0.0
    if np.std(angles) > 3.0:
        return 0.0

    return float(np.median(angles))


def deskew_image(image: Image.Image) -> tuple[Image.Image, str]:
    warped = perspective_deskew(image)
    if warped is not None:
        return warped, "perspective-corrected"
    angle = estimate_tilt_angle(image)
    if abs(angle) > 0.3:
        rotated = image.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
        return rotated, f"tilt-corrected({angle:.1f} deg)"
    return image, "none needed"


def detect_rotation_degrees(image: Image.Image) -> int:
    data_url = encode_pil_image(image)
    response = client.chat.completions.create(
        model=CHEAP_MODEL,
        reasoning_effort="medium",
        max_completion_tokens=500,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Look at this photo of a printed document. Determine how many degrees CLOCKWISE "
                    "the image must be rotated so the printed text reads normally, upright, left to right.\n"
                    "Base your answer on recognizable PRINTED WORDS (e.g. column headers, product names) - "
                    "words have one unambiguous correct reading orientation. Do not rely on rows of digits "
                    "alone to judge orientation, since numbers can look plausible rotated or even mirrored.\n"
                    'Respond ONLY with JSON: {"rotation_degrees": 0} where the value is exactly one of 0, 90, 180, 270.'
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
    )
    try:
        parsed = json.loads(response.choices[0].message.content)
        degrees = int(parsed.get("rotation_degrees", 0))
        return degrees if degrees in (0, 90, 180, 270) else 0
    except (ValueError, TypeError):
        return 0


def fix_orientation(image: Image.Image) -> tuple[Image.Image, dict]:
    image = ImageOps.exif_transpose(image)

    # fix gross sideways/upside-down rotation BEFORE fine perspective correction - the
    # perspective corner-ordering heuristic assumes the page is roughly right-side-up
    # already, so running it on a still-sideways photo can warp it in the wrong direction
    degrees = detect_rotation_degrees(image)
    if degrees:
        image = image.rotate(-degrees, expand=True)
        # verify the fix actually took - vision models can misjudge clockwise/counterclockwise;
        # if it's still off, this catches and corrects it rather than shipping a wrong guess
        confirm_degrees = detect_rotation_degrees(image)
        if confirm_degrees:
            image = image.rotate(-confirm_degrees, expand=True)
            degrees = f"{degrees}+{confirm_degrees}"

    image, deskew_note = deskew_image(image)
    return image, {"deskew": deskew_note, "rotation_applied_degrees": degrees}


def call_vision_model(image: Image.Image, max_completion_tokens: int = 16000) -> dict:
    """A very dense page (many handwritten pairs, e.g. a multi-column notepad order) can push
    the model's combined reasoning+output tokens right up against the budget - if reasoning
    alone consumes it, the response comes back empty. Retries once with double the budget in
    that specific case rather than failing the whole photo outright."""
    data_url = encode_pil_image(image)
    response = client.chat.completions.create(
        model=CHEAP_MODEL,
        reasoning_effort="medium",
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the handwritten-marked rows from this photo."},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        if choice.finish_reason == "length" and max_completion_tokens < 64000:
            return call_vision_model(image, max_completion_tokens=max_completion_tokens * 2)
        raise RuntimeError(
            f"Vision model returned an empty response (finish_reason={choice.finish_reason}) - "
            "this photo may be too dense for the current token budget."
        )
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Vision model response wasn't valid JSON ({e}): {content[:200]!r}") from e


def escalate_uncertain_item(item: dict) -> None:
    crop_img = item.get("_crop_image")
    bbox = item.get("row_bbox")
    if crop_img is None or not bbox:
        return
    region = crop_region(crop_img, bbox, pad_frac=0.15)
    if region is None:
        return
    data_url = encode_pil_image(region)
    try:
        response = client.chat.completions.create(
            model=PREMIUM_MODEL,
            reasoning_effort="high",
            max_completion_tokens=1000,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are looking at a small cropped region of one row from a printed order sheet. "
                        "A prior automated pass flagged this row as uncertain - it is possible there is NO "
                        "handwritten mark here at all (a false detection) and mark_present should be false. "
                        "It's also possible the mark is CROSSED OUT, SCRATCHED OVER, or otherwise VOIDED rather "
                        "than a clean digit - if so set appears_altered to true and still give your best-guess value.\n"
                        "Look carefully at the whole margin before deciding.\n"
                        "If there IS a handwritten mark, it is always a digit 0-9 per character (a mark that "
                        'looks like a slash "/" is a stylized "1"). Read it as carefully as possible.\n'
                        "The confidence field must be EXACTLY \"high\" or EXACTLY \"low\" - no other value - "
                        "report your true confidence, do not default to high.\n"
                        'Respond ONLY with JSON: {"mark_present": true, "appears_altered": false, "handwritten_number": "", "confidence": "high"}'
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ],
                },
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        if not parsed.get("mark_present", True):
            item["_drop"] = True
            item["escalated"] = True
            return
        value = parsed.get("handwritten_number")
        if value not in (None, ""):
            item["handwritten_number"] = value
            item["confidence"] = parsed.get("confidence", "low")
            item["appears_altered"] = bool(parsed.get("appears_altered", False))
            item["escalated"] = True
    except Exception:
        pass


def reconcile_dual_runs(items_a: list[dict], items_b: list[dict]) -> list[dict]:
    """Two independent cheap-model readings of the same crop. A row found by both runs with
    the same handwritten value is trustworthy. A row found by only one run (a possible miss by
    the other, or a possible hallucination by the one that found it) or found by both with
    different values gets tagged for review - resolved later by premium escalation."""

    def row_key(item):
        return (
            str(item.get("item_no", "")).strip().lower(),
            str(item.get("description", "")).strip().lower(),
        )

    by_key_b = {}
    for item in items_b:
        by_key_b.setdefault(row_key(item), item)

    matched_b_keys = set()
    combined = []
    for item_a in items_a:
        key = row_key(item_a)
        item_b = by_key_b.get(key)
        if item_b is not None:
            matched_b_keys.add(key)
            val_a = str(item_a.get("handwritten_number", "")).strip().lower()
            val_b = str(item_b.get("handwritten_number", "")).strip().lower()
            if val_a != val_b:
                item_a["_reconcile_flag"] = (
                    f"two independent readings disagree on the handwritten value ({val_a} vs {val_b})"
                )
        else:
            item_a["_reconcile_flag"] = "found by only one of two independent readings - please verify it is really marked"
        combined.append(item_a)

    for item_b in items_b:
        if row_key(item_b) not in matched_b_keys:
            item_b["_reconcile_flag"] = "found by only one of two independent readings - please verify it is really marked"
            combined.append(item_b)

    return combined


def extract_from_image(
    file_name: str,
    file_bytes: bytes,
    item_memory: dict | None = None,
    item_catalog: dict | None = None,
    learned_associations: dict | None = None,
) -> tuple[list[dict], dict, dict]:
    if item_memory is None:
        item_memory = load_item_memory()
    if item_catalog is None:
        item_catalog = load_item_catalog()
    if learned_associations is None:
        learned_associations = load_learned_associations()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image, orientation_info = fix_orientation(image)
    crops = split_top_bottom(image)

    image_height = image.size[1]
    crop_sides = ["top", "bottom"]

    raw_items = []
    rows_seen_total = 0
    with ThreadPoolExecutor(max_workers=len(crops) * 2) as executor:
        futures = {
            executor.submit(call_vision_model, crop_img): (crop_idx, run)
            for crop_idx, (crop_img, _y_offset) in enumerate(crops)
            for run in ("a", "b")
        }
        results: dict[int, dict] = {}
        for future in as_completed(futures):
            crop_idx, run = futures[future]
            results.setdefault(crop_idx, {})[run] = future.result()

    for crop_idx, (crop_img, y_offset) in enumerate(crops):
        parsed_a = results[crop_idx]["a"]
        parsed_b = results[crop_idx]["b"]
        items_a = parsed_a.get("items", [])
        items_b = parsed_b.get("items", [])
        crop_height = crop_img.size[1]
        for item in items_a + items_b:
            item["_crop_image"] = crop_img
            item["_crop_side"] = crop_sides[crop_idx]
            bbox = item.get("row_bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    y0, y1 = float(bbox[1]), float(bbox[3])
                    item["_abs_bbox"] = (
                        (y_offset + y0 * crop_height) / image_height,
                        (y_offset + y1 * crop_height) / image_height,
                    )
                    item["_edge_distance"] = min(y0, 1.0 - y1)
                except (TypeError, ValueError, IndexError):
                    pass
        raw_items.extend(reconcile_dual_runs(items_a, items_b))
        rows_seen_total += max(parsed_a.get("total_rows_on_page", 0) or 0, parsed_b.get("total_rows_on_page", 0) or 0)

    def flag_review(items_list):
        conflicting = find_conflicting_keys(items_list)
        for item in items_list:
            row_key = (
                str(item.get("item_no", "")).strip().lower(),
                str(item.get("description", "")).strip().lower(),
            )
            reasons = []
            if str(item.get("confidence", "high")).lower() != "high":
                suffix = " (even after premium re-check)" if item.get("escalated") else ""
                reasons.append(f"model was not confident in this reading{suffix}")
            if item.get("appears_altered"):
                reasons.append("mark appears crossed out / scratched / voided - please verify it wasn't cancelled")
            if row_key in conflicting:
                reasons.append("same row read differently across overlapping crops - please verify")
            if is_suspiciously_high(item.get("handwritten_number")):
                reasons.append("handwritten value is 10 or higher - please verify")
            if reads_as_seven(item.get("handwritten_number")):
                reasons.append("handwritten value read as 7 - easily confused with 1, please verify")
            catalog_reason = resolve_against_catalog(item, item_catalog, learned_associations)
            if catalog_reason:
                reasons.append(catalog_reason)
            if item.get("_reconcile_flag"):
                # this flag type questions whether a mark is real at all - the one kind of
                # uncertainty it's safe to auto-resolve from history (see is_known_artifact)
                if is_known_artifact(item.get("item_no", ""), item_memory):
                    item["_drop"] = True
                    item["_auto_suppressed"] = True
                else:
                    reasons.append(item["_reconcile_flag"])
            item["needs_review"] = bool(reasons)
            item["review_reason"] = "; ".join(reasons)

    items = dedupe_items(raw_items)
    items = merge_overlap_duplicates(items)
    flag_review(items)

    escalated_count = 0
    for item in items:
        if item["needs_review"]:
            escalate_uncertain_item(item)
            if item.get("escalated"):
                escalated_count += 1

    auto_suppressed_count = sum(1 for item in items if item.get("_auto_suppressed"))
    dropped_count = sum(1 for item in items if item.get("_drop") and not item.get("_auto_suppressed"))
    items = [item for item in items if not item.get("_drop")]

    # cheap-model uncertainty may have been resolved by escalation (e.g. two overlap-crop
    # reads that disagreed now agree) - re-dedupe and re-flag with the corrected values
    items = dedupe_items(items)
    flag_review(items)

    # precise counts for the usage report - computed here, before "escalated" gets stripped below
    no_escalation_needed = sum(1 for item in items if not item.get("escalated") and not item["needs_review"])
    resolved_by_premium = sum(1 for item in items if item.get("escalated") and not item["needs_review"])

    review_crops = {}
    for idx, item in enumerate(items):
        item["source_image"] = file_name
        review_id = f"{file_name}::{idx}"
        item["review_id"] = review_id
        if item["needs_review"]:
            crop_img = item.get("_crop_image")
            preview = crop_region(crop_img, item.get("row_bbox")) if crop_img is not None else None
            # cropped from the FULL original photo using the row's actual known position, not
            # from whichever top/bottom half it was detected in - so it can't get cut off at a
            # crop seam the way re-showing the source half could
            context = crop_context_region(image, item.get("_abs_bbox"))
            if preview is not None or context is not None or crop_img is not None:
                review_crops[review_id] = {
                    "small": preview if preview is not None else (context or crop_img),
                    "full": context if context is not None else crop_img,
                }
        item.pop("_crop_image", None)
        item.pop("row_bbox", None)
        item.pop("escalated", None)
        item.pop("_reconcile_flag", None)
        item.pop("_drop", None)
        item.pop("_crop_side", None)
        item.pop("_abs_bbox", None)
        item.pop("_edge_distance", None)
        item.pop("_auto_suppressed", None)

    debug_info = {
        "file": file_name,
        "deskew": orientation_info["deskew"],
        "rotation_applied_degrees": orientation_info["rotation_applied_degrees"],
        "rows_seen_across_crops": rows_seen_total,
        "escalated_to_premium": escalated_count,
        "dropped_as_false_detection": dropped_count,
        "auto_suppressed_known_artifact": auto_suppressed_count,
        "no_escalation_needed": no_escalation_needed,
        "resolved_by_premium": resolved_by_premium,
        "items_before_dedupe": len(raw_items),
        "items_returned": len(items),
        "flagged_for_review": sum(1 for item in items if item["needs_review"]),
    }
    return items, debug_info, review_crops


# --- shared review-UI helpers (used by both the manual-upload flow and the Review Queue) ---


def render_review_row(item: dict, review_id: str, crop_img, full_img=None) -> None:
    if crop_img is not None:
        st.image(crop_img, use_container_width=True)
    else:
        st.caption("(no preview available)")

    if full_img is not None:
        with st.expander(f"🔍 Can't find item {item.get('item_no', '')}? Show the full section"):
            st.image(full_img, use_container_width=True)

    cols = st.columns([1.6, 1, 1, 1])
    with cols[0]:
        st.markdown(f"**{item.get('source_image', '')}**  \n{item.get('description', '')}")
        st.caption(item.get("review_reason", ""))
    with cols[1]:
        st.text_input(
            "Item #",
            value=str(item.get("item_no", "")),
            key=f"itemno_{review_id}",
        )
    with cols[2]:
        st.text_input(
            "Handwritten value",
            value=str(item.get("handwritten_number", "")),
            key=f"hw_{review_id}",
        )
    with cols[3]:
        st.checkbox("Ignore this row", key=f"ignore_{review_id}")
    st.divider()


def apply_review_overrides(items: list[dict]) -> list[dict]:
    resolved = []
    for item in items:
        review_id = item.get("review_id")
        if item.get("needs_review") and review_id:
            if st.session_state.get(f"ignore_{review_id}"):
                continue
            overrides = {}
            edited_value = st.session_state.get(f"hw_{review_id}")
            if edited_value is not None:
                overrides["handwritten_number"] = edited_value
            edited_item_no = st.session_state.get(f"itemno_{review_id}")
            if edited_item_no is not None:
                overrides["item_no"] = edited_item_no
            if overrides:
                item = {**item, **overrides}
        resolved.append(item)
    return resolved


def find_high_value_items(items: list[dict]) -> list[dict]:
    return [item for item in items if is_suspiciously_high(item.get("handwritten_number"))]


def render_qty_confirmation_gate(high_items: list[dict], key_prefix: str) -> bool:
    """Final safety net before a commit actually writes output - a review-time edit (or a value
    that was never flagged for any other reason) could still be a suspiciously high quantity.
    Renders a red confirm-or-correct gate for those specific rows, mutating them in place so a
    correction here is reflected in what gets committed. Returns True once the user clicks through."""
    st.markdown(
        '<div style="background-color:#f8d7da;color:#842029;padding:10px 16px;border-radius:6px;'
        f'font-weight:600;margin-bottom:8px;">⚠️ Are you sure these quantities are correct? '
        f'{len(high_items)} item(s) have a quantity of 10 or higher.</div>',
        unsafe_allow_html=True,
    )
    for idx, item in enumerate(high_items):
        cols = st.columns([2, 1])
        with cols[0]:
            st.markdown(f"**{item.get('item_no', '')}** — {item.get('description', '')}")
        with cols[1]:
            item["handwritten_number"] = st.text_input(
                "Quantity",
                value=str(item.get("handwritten_number", "")),
                key=f"{key_prefix}_qty_{idx}",
            )
    return st.button("Confirm quantities and continue", key=f"{key_prefix}_confirm_qty")


# --- learned item memory (persists across runs; scoped to "is this even a real mark",
# never to a specific handwritten value - values are quantities and legitimately change
# order to order, so trusting history there could reinforce a wrong reading) ---

ITEM_MEMORY_PATH = DATA_DIR / "item_memory.json"
ARTIFACT_CONFIRMATION_THRESHOLD = 3


def load_item_memory() -> dict:
    try:
        return json.loads(ITEM_MEMORY_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_item_memory(memory: dict) -> None:
    try:
        ITEM_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        ITEM_MEMORY_PATH.write_text(json.dumps(memory, indent=2))
    except OSError:
        pass


def is_known_artifact(item_no: str, memory: dict, threshold: int = ARTIFACT_CONFIRMATION_THRESHOLD) -> bool:
    key = str(item_no).strip().lower()
    if not key:
        return False
    return memory.get(key, {}).get("artifact_confirmations", 0) >= threshold


def record_review_outcomes(flagged_items: list[dict]) -> None:
    """Call once at commit time. A row the user checked 'Ignore' is evidence the mark at that
    item code isn't real; a row they kept (edited or not) is evidence it is - which resets any
    prior artifact streak, since that proves the item CAN carry a genuine mark."""
    memory = load_item_memory()
    changed = False
    for item in flagged_items:
        item_no = str(item.get("item_no", "")).strip().lower()
        review_id = item.get("review_id")
        if not item_no or not review_id:
            continue
        entry = memory.setdefault(item_no, {"artifact_confirmations": 0})
        if st.session_state.get(f"ignore_{review_id}"):
            entry["artifact_confirmations"] = entry.get("artifact_confirmations", 0) + 1
        else:
            entry["artifact_confirmations"] = 0
        changed = True
    if changed:
        save_item_memory(memory)


def tally_human_agreement(flagged_items: list[dict]) -> tuple[int, int]:
    """Call at commit time, on the items that were actually shown to the human for review.
    'Agreed' = committed with no edit and not ignored (the app's guess stood as-is).
    'Disagreed' = the human corrected the value/item#, or marked it ignore."""
    agreed = disagreed = 0
    for item in flagged_items:
        review_id = item.get("review_id")
        if not review_id:
            continue
        if st.session_state.get(f"ignore_{review_id}"):
            disagreed += 1
            continue
        edited_hw = st.session_state.get(f"hw_{review_id}")
        edited_item_no = st.session_state.get(f"itemno_{review_id}")
        changed_hw = edited_hw is not None and str(edited_hw) != str(item.get("handwritten_number", ""))
        changed_item_no = edited_item_no is not None and str(edited_item_no) != str(item.get("item_no", ""))
        if changed_hw or changed_item_no:
            disagreed += 1
        else:
            agreed += 1
    return agreed, disagreed


# --- item catalog cross-check (a master Item Number/Brand/Description reference the user
# uploads) - catches printed-code misreads by checking a row's item_no against what that
# code's description has always been, and auto-corrects only the narrow, high-confidence
# cases: a single confusable-digit swap or a stray/missing 0 or 1, with an exact description
# match, or (regardless of digit-diff size) an exact match on BOTH description and a
# separately-learned old_item pairing. Anything less certain is a flag, never a guess. ---

ITEM_CATALOG_PATH = DATA_DIR / "item_catalog.xlsx"
LEARNED_ASSOCIATIONS_PATH = DATA_DIR / "learned_item_associations.json"
NEW_ITEM_CONFIRMATION_THRESHOLD = 2
OLD_ITEM_CONFIRMATION_THRESHOLD = 2

# digit pairs a human (or a blurry scan) commonly confuses for one another
CONFUSABLE_DIGIT_PAIRS = {frozenset(p) for p in [("1", "7"), ("6", "0"), ("3", "8"), ("5", "6"), ("4", "9")]}
# digits known to occasionally get spuriously duplicated or dropped in a printed code
INSERTABLE_DIGITS = {"0", "1"}


def normalize_catalog_text(value) -> str:
    # the catalog file uses a stray "*" (leading, trailing, or crammed against the text with
    # no space) as some kind of internal marker unrelated to the product's identity - drop it
    return " ".join(str(value or "").replace("*", " ").strip().upper().split())


def is_single_digit_substitution(a: str, b: str) -> bool:
    if len(a) != len(b) or a == b:
        return False
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diffs) == 1 and frozenset(diffs[0]) in CONFUSABLE_DIGIT_PAIRS


def is_single_stray_digit(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) != 1:
        return False
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    for i, ch in enumerate(longer):
        if ch in INSERTABLE_DIGITS and longer[:i] + longer[i + 1:] == shorter:
            return True
    return False


def is_plausible_code_misread(extracted_code: str, candidate_code: str) -> bool:
    return is_single_digit_substitution(extracted_code, candidate_code) or is_single_stray_digit(extracted_code, candidate_code)


def load_learned_associations() -> dict:
    try:
        data = json.loads(LEARNED_ASSOCIATIONS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {}
    data.setdefault("new_items", {})
    data.setdefault("old_item_map", {})
    data.setdefault("correction_log", [])
    return data


def save_learned_associations(data: dict) -> None:
    try:
        LEARNED_ASSOCIATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEARNED_ASSOCIATIONS_PATH.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def load_item_catalog() -> dict:
    """Builds {"by_code": {code: {description, brand}}, "by_description": {desc: [candidates]}}
    from the uploaded Excel, merged with any confirmed-twice new items learned from review -
    those behave identically to a real catalog entry from then on."""
    by_code: dict[str, dict] = {}
    by_description: dict[str, list] = {}

    def add_entry(code: str, description: str, brand: str) -> None:
        if not code or not description:
            return
        by_code[code] = {"description": description, "brand": brand}
        by_description.setdefault(description, [])
        if not any(c["item_no"] == code for c in by_description[description]):
            by_description[description].append({"item_no": code, "brand": brand})

    if ITEM_CATALOG_PATH.exists():
        try:
            df = pd.read_excel(ITEM_CATALOG_PATH)
            for _, row in df.iterrows():
                add_entry(
                    normalize_catalog_text(row.get("Item Number")),
                    normalize_catalog_text(row.get("Item Description")),
                    normalize_catalog_text(row.get("Brand")),
                )
        except (ValueError, KeyError, OSError):
            pass

    learned = load_learned_associations()
    for code, entry in learned.get("new_items", {}).items():
        if entry.get("confirmed_count", 0) >= NEW_ITEM_CONFIRMATION_THRESHOLD:
            add_entry(code, normalize_catalog_text(entry.get("description")), normalize_catalog_text(entry.get("brand")))

    return {"by_code": by_code, "by_description": by_description}


def resolve_against_catalog(item: dict, catalog: dict, learned: dict) -> str:
    """Checks item_no against the catalog; auto-corrects it in place when a correction is
    unambiguous and high-confidence (see module note above), and always returns a review
    reason string ("" when there's nothing to flag)."""
    by_code = catalog.get("by_code", {})
    by_description = catalog.get("by_description", {})
    if not by_code:
        return ""

    code = normalize_catalog_text(item.get("item_no"))
    description = normalize_catalog_text(item.get("description"))
    if not code or not description:
        return ""

    known = by_code.get(code)
    if known and known["description"] == description:
        return ""  # exact match - nothing to do

    candidates = [c for c in by_description.get(description, []) if c["item_no"] != code]
    if not candidates:
        if not known:
            item["_catalog_new_item"] = True
            return "item code not found in the catalog (may be a new item) - please verify"
        return "item code's description doesn't match the catalog - please verify"

    read_brand = normalize_catalog_text(item.get("brand"))
    if len(candidates) > 1 and read_brand:
        narrowed = [c for c in candidates if c.get("brand") == read_brand]
        if narrowed:
            candidates = narrowed

    single_edit_matches = [c for c in candidates if is_plausible_code_misread(code, c["item_no"])]

    read_old_item = normalize_catalog_text(item.get("old_item"))
    old_item_map = learned.get("old_item_map", {})
    corroborated_matches = []
    if read_old_item:
        for c in candidates:
            entry = old_item_map.get(c["item_no"], {})
            if entry.get("old_item") == read_old_item and entry.get("confirmed_count", 0) >= OLD_ITEM_CONFIRMATION_THRESHOLD:
                corroborated_matches.append(c)

    resolved = single_edit_matches if len(single_edit_matches) == 1 else (
        corroborated_matches if len(corroborated_matches) == 1 else []
    )
    if len(resolved) == 1:
        item["item_no"] = resolved[0]["item_no"]
        item["_catalog_corrected_from"] = code
        return ""

    return "item code's description doesn't match the catalog - please verify"


def record_catalog_learning(items: list[dict]) -> None:
    """Call once at commit time, on the final (post-review) items. Grows the new-items list
    (an item flagged as an unrecognized code that the human kept as-is, not retyped, counts as
    a confirmation) and the item_no -> old_item association, independent of the uploaded
    catalog file so neither resets when that file gets replaced."""
    learned = load_learned_associations()
    changed = False
    for item in items:
        code = normalize_catalog_text(item.get("item_no"))
        if not code:
            continue
        corrected_from = item.get("_catalog_corrected_from")
        if corrected_from:
            learned["correction_log"].append({
                "from": corrected_from, "to": code,
                "description": normalize_catalog_text(item.get("description")),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            changed = True
        if item.get("_catalog_new_item"):
            entry = learned["new_items"].setdefault(code, {
                "description": normalize_catalog_text(item.get("description")),
                "brand": normalize_catalog_text(item.get("brand")),
                "confirmed_count": 0,
            })
            entry["confirmed_count"] = entry.get("confirmed_count", 0) + 1
            changed = True
        old_item = normalize_catalog_text(item.get("old_item"))
        if old_item:
            entry = learned["old_item_map"].setdefault(code, {"old_item": old_item, "confirmed_count": 0})
            if entry.get("old_item") == old_item:
                entry["confirmed_count"] = entry.get("confirmed_count", 0) + 1
            else:
                entry["old_item"] = old_item
                entry["confirmed_count"] = 1
            changed = True
    if changed:
        save_learned_associations(learned)


# --- monthly usage report (cumulative CSV, one row per committed batch) ---

REPORTS_DIR = DATA_DIR / "reports"
REPORT_COLUMNS = [
    "timestamp", "customer", "batch_id", "num_photos",
    "items_no_escalation", "items_resolved_by_premium", "items_reached_human_review",
    "items_human_agreed", "items_human_disagreed",
]


def report_path_for(month_str: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR / f"usage_report_{month_str}.csv"


def log_batch_report(
    customer: str,
    batch_id: str,
    num_photos: int,
    no_escalation: int,
    resolved_by_premium: int,
    reached_human_review: int,
    agreed: int,
    disagreed: int,
) -> None:
    path = report_path_for(date.today().strftime("%Y%m"))
    is_new = not path.exists()
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "customer": customer,
        "batch_id": batch_id,
        "num_photos": num_photos,
        "items_no_escalation": no_escalation,
        "items_resolved_by_premium": resolved_by_premium,
        "items_reached_human_review": reached_human_review,
        "items_human_agreed": agreed,
        "items_human_disagreed": disagreed,
    }
    pd.DataFrame([row], columns=REPORT_COLUMNS).to_csv(path, mode="a", header=is_new, index=False)


def list_available_reports() -> list[str]:
    if not REPORTS_DIR.exists():
        return []
    return sorted((p.stem.replace("usage_report_", "") for p in REPORTS_DIR.glob("usage_report_*.csv")), reverse=True)


def format_month_label(month_str: str) -> str:
    try:
        return date(int(month_str[:4]), int(month_str[4:6]), 1).strftime("%B %Y")
    except (ValueError, IndexError):
        return month_str


# --- folder-watch config persistence ---


def load_saved_parent_folder() -> str:
    try:
        return json.loads(WATCH_CONFIG_PATH.read_text()).get("parent_folder_path", "")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ""


def save_parent_folder():
    try:
        WATCH_CONFIG_PATH.write_text(json.dumps({"parent_folder_path": st.session_state.get("parent_folder_path", "")}))
    except OSError:
        pass


def sv_paths(parent: Path, sv_code: str) -> dict:
    # SVn itself is the drop-in input folder (one less step for the user), Output is shared
    # across all customers (output filenames already carry the customer code), and Processed
    # keeps a per-customer subfolder so processed photos from different customers don't mix.
    processed = parent / "Processed" / sv_code
    return {
        "input": parent / sv_code,
        "processed": processed,
        "output": parent / "Output",
        "base": processed,  # pending-review staging lives alongside this customer's processed history
    }


# --- pending-review staging (disk-backed so exceptions survive an app restart) ---


def get_pending_dir(base: Path) -> Path:
    d = base / PENDING_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def save_pending_batch(
    base: Path, batch_id: str, items: list[dict], review_crops: dict, stats: dict | None = None
) -> None:
    pending_dir = get_pending_dir(base)
    payload = {"items": items, "stats": stats or {}}
    (pending_dir / f"{batch_id}.json").write_text(json.dumps(payload))
    if review_crops:
        crops_dir = pending_dir / f"{batch_id}_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for review_id, images in review_crops.items():
            images["small"].save(crops_dir / f"{safe_filename(review_id)}.png")
            images["full"].save(crops_dir / f"{safe_filename(review_id)}_full.png")


def list_pending_batches(base: Path) -> list[dict]:
    pending_dir = get_pending_dir(base)
    batches = []
    for json_path in sorted(pending_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, list):
            items, stats = data, {}  # backward compat with the pre-stats format
        else:
            items, stats = data.get("items", []), data.get("stats", {})
        batches.append({"batch_id": json_path.stem, "items": items, "stats": stats})
    return batches


def load_pending_crop(base: Path, batch_id: str, review_id: str, variant: str = "small"):
    suffix = "_full" if variant == "full" else ""
    crop_path = get_pending_dir(base) / f"{batch_id}_crops" / f"{safe_filename(review_id)}{suffix}.png"
    if crop_path.exists():
        return Image.open(crop_path)
    return None


def delete_pending_batch(base: Path, batch_id: str) -> None:
    pending_dir = get_pending_dir(base)
    (pending_dir / f"{batch_id}.json").unlink(missing_ok=True)
    crops_dir = pending_dir / f"{batch_id}_crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir, ignore_errors=True)


def count_pending_flagged(base: Path) -> int:
    return sum(
        1
        for batch in list_pending_batches(base)
        for item in batch["items"]
        if item.get("needs_review")
    )


# --- background extraction jobs ---
# Extraction runs in a server-side thread, NOT inside the browser session's script run - a
# dropped connection (screen saver, closed tab, laptop sleep) tears down the session and used
# to kill the run mid-flight. Each job's photos, status, and results live on disk, so a job
# survives any browser disconnect, and one interrupted by a server restart can be resumed from
# its saved photos. Progress is a small in-memory registry shared across all sessions.

JOBS_DIR = DATA_DIR / "jobs"
JOB_RETENTION_SECONDS = 7 * 24 * 3600
MAX_CONCURRENT_PHOTOS = 5  # global across ALL jobs - bounds memory/API load when batches overlap


@st.cache_resource
def get_job_runtime() -> dict:
    return {
        "executor": ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PHOTOS),
        "progress": {},
        "lock": threading.Lock(),
    }


def job_dir_for(job_id: str) -> Path:
    return JOBS_DIR / job_id


def read_job_meta(job_id: str) -> dict | None:
    try:
        return json.loads((job_dir_for(job_id) / "job.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_job_meta(job_id: str, meta: dict) -> None:
    path = job_dir_for(job_id) / "job.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, path)


def run_extraction_job(job_id: str, runtime: dict, item_memory: dict, item_catalog: dict, learned: dict) -> None:
    """Worker thread body. Must never touch st.* - there is no browser session attached."""
    job_dir = job_dir_for(job_id)
    meta = read_job_meta(job_id) or {}
    all_items, debug_rows, all_crops, errors = [], [], {}, []

    def process_photo(photo: dict):
        data = (job_dir / "photos" / photo["file"]).read_bytes()
        return extract_from_image(photo["name"], data, item_memory, item_catalog, learned)

    try:
        futures = {runtime["executor"].submit(process_photo, p): p["name"] for p in meta["photos"]}
        for future in as_completed(futures):
            name = futures[future]
            try:
                items, debug_info, crops = future.result()
                all_items.extend(items)
                debug_rows.append(debug_info)
                all_crops.update(crops)
            except Exception as e:
                errors.append(f"{name}: {e}")
            with runtime["lock"]:
                progress = runtime["progress"].setdefault(job_id, {"done": 0, "total": len(meta["photos"])})
                progress["done"] += 1
                progress["last"] = name
        all_items = resolve_duplicate_item_codes(all_items)
        save_pending_batch(
            job_dir, "results", all_items, all_crops,
            {"debug_rows": debug_rows, "num_photos": len(meta["photos"]), "errors": errors},
        )
        meta["status"] = "done"
        shutil.rmtree(job_dir / "photos", ignore_errors=True)
    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = str(e)
    write_job_meta(job_id, meta)


def launch_job_thread(job_id: str) -> None:
    runtime = get_job_runtime()
    meta = read_job_meta(job_id) or {}
    with runtime["lock"]:
        runtime["progress"][job_id] = {"done": 0, "total": len(meta.get("photos", []))}
    threading.Thread(
        target=run_extraction_job,
        args=(job_id, runtime, load_item_memory(), load_item_catalog(), load_learned_associations()),
        daemon=True,
    ).start()


def cleanup_old_jobs() -> None:
    if not JOBS_DIR.exists():
        return
    cutoff = time.time() - JOB_RETENTION_SECONDS
    for d in JOBS_DIR.iterdir():
        meta = read_job_meta(d.name) if d.is_dir() else None
        if meta and meta.get("status") != "processing" and meta.get("created", 0) < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def start_extraction_job(files_payload: list[tuple[str, bytes]], customer: str) -> str:
    cleanup_old_jobs()
    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
    photos_dir = job_dir_for(job_id) / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    photos = []
    for idx, (name, data) in enumerate(files_payload):
        saved = f"{idx:03d}_{safe_filename(name)}"
        (photos_dir / saved).write_bytes(data)
        photos.append({"file": saved, "name": name})
    write_job_meta(job_id, {
        "job_id": job_id, "customer": customer, "created": time.time(),
        "created_label": time.strftime("%b %d %I:%M %p"), "status": "processing", "photos": photos,
    })
    launch_job_thread(job_id)
    return job_id


def list_jobs(limit: int = 10) -> list[dict]:
    if not JOBS_DIR.exists():
        return []
    jobs = []
    for d in sorted((p for p in JOBS_DIR.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)[:limit]:
        meta = read_job_meta(d.name)
        if meta:
            jobs.append(meta)
    return jobs


def job_display_status(meta: dict) -> str:
    """'processing' only if a live worker actually exists - a job marked processing on disk
    with no in-memory progress entry was orphaned by a server restart."""
    status = meta.get("status", "failed")
    if status == "processing" and meta["job_id"] not in get_job_runtime()["progress"]:
        return "interrupted"
    return status


def load_job_into_session(job_id: str) -> None:
    meta = read_job_meta(job_id) or {}
    job_dir = job_dir_for(job_id)
    batches = list_pending_batches(job_dir)
    batch = batches[0] if batches else {"items": [], "stats": {}}
    items, stats = batch["items"], batch["stats"]
    review_crops = {}
    for item in items:
        rid = item.get("review_id")
        if item.get("needs_review") and rid:
            small = load_pending_crop(job_dir, "results", rid)
            full = load_pending_crop(job_dir, "results", rid, variant="full")
            if small and full:
                review_crops[rid] = {"small": small, "full": full}
    # review widgets are keyed by review_id, which repeats across jobs that share photo
    # filenames - clear leftovers so one job's review state can't leak into the next
    for key in [k for k in st.session_state.keys() if k.startswith(("hw_", "ignore_", "itemno_", "manual_upload"))]:
        del st.session_state[key]
    st.session_state["results"] = items
    st.session_state["debug_rows"] = stats.get("debug_rows", [])
    st.session_state["num_photos"] = stats.get("num_photos", 0)
    st.session_state["job_errors"] = stats.get("errors", [])
    st.session_state["review_crops"] = review_crops
    st.session_state["review_customer"] = meta.get("customer")
    st.session_state["loaded_job_id"] = job_id
    st.session_state["review_committed"] = False
    st.session_state["report_logged"] = False
    st.session_state["auto_downloaded"] = False


def finish_job(job_id: str | None) -> None:
    """Called at commit: the export is done, so free the (large) saved crops and mark it finished."""
    meta = read_job_meta(job_id) if job_id else None
    if not meta:
        return
    meta["status"] = "committed"
    write_job_meta(job_id, meta)
    delete_pending_batch(job_dir_for(job_id), "results")


# --- folder-watch processing ---


def list_image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def folder_signature(files: list[Path]) -> tuple:
    return tuple((p.name, p.stat().st_size) for p in files)


def process_customer_batch(sv_code: str, paths: dict, area) -> None:
    input_dir, processed_dir, output_dir, base = paths["input"], paths["processed"], paths["output"], paths["base"]
    image_files = list_image_files(input_dir)

    area.info(f"📸 {sv_code}: detected {len(image_files)} photo(s). Working on them...")
    progress = area.progress(0.0, text="Starting...")

    all_items = []
    all_review_crops = {}
    all_debug_info = []
    failures = []
    total = len(image_files)
    done = 0
    item_memory = load_item_memory()
    item_catalog = load_item_catalog()
    learned_associations = load_learned_associations()
    with ThreadPoolExecutor(max_workers=min(5, total)) as executor:
        futures = {
            executor.submit(extract_from_image, p.name, p.read_bytes(), item_memory, item_catalog, learned_associations): p
            for p in image_files
        }
        for future in as_completed(futures):
            path = futures[future]
            done += 1
            try:
                items, photo_debug_info, review_crops = future.result()
            except Exception as e:
                failures.append(f"{path.name}: {e}")
                progress.progress(done / total, text=f"Failed {path.name} ({done}/{total})")
                continue
            all_items.extend(items)
            all_review_crops.update(review_crops)
            all_debug_info.append(photo_debug_info)
            dest = processed_dir / path.name
            if dest.exists():
                dest = processed_dir / f"{path.stem}_{int(time.time())}{path.suffix}"
            shutil.move(str(path), str(dest))
            progress.progress(done / total, text=f"Finished {done}/{total} ({path.name})")

    progress.progress(1.0, text="Done.")

    succeeded = total - len(failures)
    all_items = resolve_duplicate_item_codes(all_items)
    flagged_count = sum(1 for item in all_items if item.get("needs_review"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stats = {
        "num_photos": succeeded,
        "no_escalation": sum(d.get("no_escalation_needed", 0) for d in all_debug_info),
        "resolved_by_premium": sum(d.get("resolved_by_premium", 0) for d in all_debug_info),
    }

    if all_items and flagged_count:
        batch_id = f"{sv_code}_order_{timestamp}"
        save_pending_batch(base, batch_id, all_items, all_review_crops, stats)
        area.warning(
            f"🚨 {sv_code}: processed {succeeded} photo(s), found {len(all_items)} marked row(s), but "
            f"{flagged_count} need review before output can be produced. No output file was written yet."
        )
    elif all_items:
        df = pd.DataFrame(all_items).drop(columns=["review_id"], errors="ignore").rename(columns=EXPORT_COLUMN_RENAME)
        out_path = output_dir / f"{sv_code}_order_{timestamp}.csv"
        df.to_csv(out_path, index=False)
        log_batch_report(
            sv_code, f"{sv_code}_order_{timestamp}", stats["num_photos"],
            stats["no_escalation"], stats["resolved_by_premium"], 0, 0, 0,
        )
        area.success(
            f"✅ {sv_code}: processed {succeeded} photo(s), found {len(all_items)} marked row(s) - saved to {out_path.name}."
        )
    else:
        area.success(f"✅ {sv_code}: processed {succeeded} photo(s), no marked rows found.")
    if failures:
        area.error(f"{sv_code}: {len(failures)} photo(s) failed and were left in Input: " + "; ".join(failures))


def check_and_process_customer(parent: Path, sv_code: str, area) -> None:
    """Ensures this customer's folders exist, waits for Input to settle (unchanged across one
    full poll cycle) before processing, so photos dropped a few seconds apart land in one batch."""
    paths = sv_paths(parent, sv_code)
    for d in (paths["input"], paths["processed"], paths["output"]):
        d.mkdir(parents=True, exist_ok=True)

    files = list_image_files(paths["input"])
    sig_key = f"watch_last_signature_{sv_code}"
    if not files:
        st.session_state[sig_key] = None
        return

    signature = folder_signature(files)
    if signature == st.session_state.get(sig_key):
        st.session_state[sig_key] = None
        process_customer_batch(sv_code, paths, area)
    else:
        st.session_state[sig_key] = signature
        area.caption(f"📸 {sv_code}: detected {len(files)} photo(s), waiting for the folder to settle...")


SV_STATUS_STYLE = {
    "yellow": ("#fff3cd", "#664d03", "⏳"),
    "red": ("#f8d7da", "#842029", "⚠️"),
    "blue": ("#cfe2ff", "#084298", "🔵"),
    "green": ("#d4edda", "#0f5132", "✅"),
}


def get_sv_status(parent: Path, sv_code: str) -> dict:
    paths = sv_paths(parent, sv_code)
    input_files = list_image_files(paths["input"]) if paths["input"].exists() else []
    pending_flagged = count_pending_flagged(paths["base"]) if paths["base"].exists() else 0
    if input_files:
        return {"status": "yellow", "detail": f"{len(input_files)} photo(s) in progress", "pending_flagged": pending_flagged}
    if pending_flagged:
        return {"status": "red", "detail": f"{pending_flagged} row(s) need review", "pending_flagged": pending_flagged}
    if paths["output"].exists():
        today = date.today()
        # check actual last-modified date, not the timestamp in the filename - a batch flagged
        # yesterday but reviewed/committed today should count as processed TODAY, not yesterday
        for f in paths["output"].glob(f"{sv_code}_order_*.csv"):
            if date.fromtimestamp(f.stat().st_mtime) == today:
                return {"status": "blue", "detail": "already processed today", "pending_flagged": 0}
    return {"status": "green", "detail": "up to date", "pending_flagged": 0}


def render_sv_pane(parent: Path, sv_code: str, status: dict, area) -> None:
    bg, fg, icon = SV_STATUS_STYLE[status["status"]]
    area.markdown(
        f'<div style="background-color:{bg};color:{fg};padding:10px 16px;border-radius:6px;'
        f'font-weight:600;margin-top:10px;">{icon} {sv_code} — {status["detail"]}</div>',
        unsafe_allow_html=True,
    )

    if status["status"] != "red":
        return  # nothing actionable to show for yellow (still working), blue, or green

    paths = sv_paths(parent, sv_code)
    base, output_dir = paths["base"], paths["output"]
    with area.expander(f"Review {sv_code}", expanded=True):
        for batch in list_pending_batches(base):
            batch_id = batch["batch_id"]
            batch_items = batch["items"]
            batch_flagged = [item for item in batch_items if item.get("needs_review")]
            st.markdown(f"**Batch {batch_id}** — {len(batch_items)} row(s), {len(batch_flagged)} need review")
            for item in batch_flagged:
                review_id = item.get("review_id")
                crop_img = load_pending_crop(base, batch_id, review_id) if review_id else None
                full_img = load_pending_crop(base, batch_id, review_id, variant="full") if review_id else None
                render_review_row(item, review_id, crop_img, full_img)

            resolved_items = apply_review_overrides(batch_items)
            # a review-time item_no correction can create a NEW duplicate that didn't exist
            # at extraction time - re-check before export, not just once up front
            resolved_items = resolve_duplicate_item_codes(resolved_items)
            high_items = find_high_value_items(resolved_items)

            if high_items:
                ready = render_qty_confirmation_gate(high_items, key_prefix=f"{sv_code}_{batch_id}")
            else:
                ready = st.button("Commit review and produce output", key=f"commit_{sv_code}_{batch_id}")

            if ready:
                record_review_outcomes(batch_flagged)
                record_catalog_learning(resolved_items)
                agreed, disagreed = tally_human_agreement(batch_flagged)
                batch_stats = batch.get("stats") or {}
                output_dir.mkdir(parents=True, exist_ok=True)
                if resolved_items:
                    out_df = pd.DataFrame(resolved_items).drop(columns=["review_id"], errors="ignore").rename(columns=EXPORT_COLUMN_RENAME)
                    out_path = output_dir / f"{batch_id}.csv"
                    out_df.to_csv(out_path, index=False)
                    st.success(f"✅ Saved {out_path.name} with {len(resolved_items)} row(s).")
                else:
                    st.info("Every row in this batch was marked ignore - no output file was produced.")
                log_batch_report(
                    sv_code, batch_id, batch_stats.get("num_photos", 0),
                    batch_stats.get("no_escalation", 0), batch_stats.get("resolved_by_premium", 0),
                    len(batch_flagged), agreed, disagreed,
                )
                delete_pending_batch(base, batch_id)
                st.rerun()


@st.fragment(run_every=WATCH_POLL_SECONDS, parallel=True)
def customer_pane(parent: Path, sv_code: str):
    """Each customer gets its OWN fragment, independently scheduled and independently rerun.
    This is what lets you review/commit SV3's exceptions while SV9 is still being processed -
    a shared/sequential loop would block all interaction until the whole pass finished.

    Active scanning is gated by the watch toggle, but the pane itself never is - existing
    pending reviews must stay visible and actionable even when watching is paused/off."""
    area = st.container()
    if st.session_state.get("watch_enabled"):
        check_and_process_customer(parent, sv_code, area)
    render_sv_pane(parent, sv_code, get_sv_status(parent, sv_code), area)


def customer_batches_section():
    if "parent_folder_path" not in st.session_state:
        st.session_state["parent_folder_path"] = load_saved_parent_folder()

    st.text_input(
        "Parent folder (contains SV1 through SV13 subfolders)",
        key="parent_folder_path",
        placeholder=r"e.g. D:\Milagro\A G S\Proyecto AGS",
        help="Drop photos directly into each SVn folder - it's created automatically. Output is shared "
        "(files are named per-customer), and Processed keeps a per-customer subfolder. Remembered between sessions.",
        on_change=save_parent_folder,
    )
    watching = st.toggle("Watch all customer folders", key="watch_enabled")

    if not watching:
        st.caption("Not watching new photos - existing pending reviews below are still shown and actionable.")

    parent_path = st.session_state.get("parent_folder_path", "").strip()
    if not parent_path:
        st.warning("Enter a parent folder path above.")
        return

    parent = Path(parent_path)
    if not parent.exists():
        st.error(f"Folder does not exist: {parent_path}")
        return

    for sv_code in SV_CODES:
        customer_pane(parent, sv_code)


@st.fragment(run_every=WATCH_POLL_SECONDS)
def exception_banner():
    parent_path = st.session_state.get("parent_folder_path", "").strip()
    if not parent_path:
        return
    parent = Path(parent_path)
    if not parent.exists():
        return
    total_flagged = sum(
        count_pending_flagged(sv_paths(parent, sv_code)["base"])
        for sv_code in SV_CODES
        if sv_paths(parent, sv_code)["base"].exists()
    )
    if total_flagged:
        st.error(
            f"🚨 {total_flagged} row(s) across customer folders need your review before their output "
            "can be produced — see the \"Customer Batches\" tab."
        )


# ============================== page layout ==============================

if not CLOUD_MODE:
    if "parent_folder_path" not in st.session_state:
        st.session_state["parent_folder_path"] = load_saved_parent_folder()
    exception_banner()

if CLOUD_MODE:
    tab_upload, tab_reports, tab_settings = st.tabs(
        ["📤 Upload Photos", "📊 Reports", "⚙️ Settings"]
    )
    tab_batches = None
else:
    tab_upload, tab_batches, tab_reports, tab_settings = st.tabs(
        ["📤 Upload Photos", "🏢 Customer Batches", "📊 Reports", "⚙️ Settings"]
    )

def render_jobs_panel() -> None:
    watching = st.session_state.get("watching_job")
    if watching and "results" not in st.session_state:
        meta = read_job_meta(watching)
        if meta and job_display_status(meta) == "done":
            load_job_into_session(watching)
            st.session_state.pop("watching_job", None)
            st.rerun()

    jobs = list_jobs()
    if not jobs:
        return
    st.subheader("Extraction jobs")
    st.caption(
        "Photos are processed in the background on the server - you can close this tab or let your "
        "screen sleep, then come back and pick up here."
    )
    progress_registry = get_job_runtime()["progress"]
    for meta in jobs:
        job_id = meta["job_id"]
        status = job_display_status(meta)
        total = len(meta.get("photos", [])) or meta.get("total", 0)
        title = f"**{meta.get('customer', '?')}** · {total} photo(s) · {meta.get('created_label', '')}"
        with st.container(border=True):
            info_col, action_col = st.columns([4, 1.4])
            with info_col:
                st.markdown(title)
                if status == "processing":
                    prog = progress_registry.get(job_id, {"done": 0, "total": total})
                    st.progress(prog["done"] / max(prog["total"], 1), text=f"Processing... {prog['done']}/{prog['total']} photos done")
                elif status == "done":
                    st.caption("✅ Ready - open it to review and export.")
                elif status == "committed":
                    st.caption("✔️ Finished and exported.")
                elif status == "interrupted":
                    st.caption("⚠️ Interrupted (the server restarted mid-run) - your photos are saved, restart to continue.")
                else:
                    st.caption(f"❌ Failed: {meta.get('error', 'unknown error')}")
            with action_col:
                if status == "done" and st.button("Open results", key=f"open_{job_id}"):
                    load_job_into_session(job_id)
                    st.session_state.pop("watching_job", None)
                    st.rerun()
                elif status in ("interrupted", "failed") and (job_dir_for(job_id) / "photos").exists():
                    if st.button("Restart", key=f"restart_{job_id}"):
                        meta["status"] = "processing"
                        meta.pop("error", None)
                        write_job_meta(job_id, meta)
                        launch_job_thread(job_id)
                        st.rerun()


with tab_upload:
    nonce = st.session_state.get("uploader_nonce", 0)
    if st.session_state.get("job_started_notice"):
        st.success(st.session_state.pop("job_started_notice"))
    uploaded_files = st.file_uploader(
        "Drop photos here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{nonce}",
    )
    upload_customer = st.selectbox(
        "Customer",
        SV_CODES + ["Other Customer"],
        key=f"upload_customer_{nonce}",
        index=None,
        placeholder="Select a customer...",
        help="Required - names the output file after the customer."
        if CLOUD_MODE
        else "Required - names the output file after the customer and, if a parent folder "
        "is configured in Customer Batches, also saves a copy into that customer's Output folder.",
    )

    if uploaded_files and not upload_customer:
        st.warning("Select a customer above before extracting.")

    if uploaded_files and upload_customer and st.button(f"Extract marked rows from {len(uploaded_files)} photo(s)"):
        job_id = start_extraction_job([(f.name, f.getvalue()) for f in uploaded_files], upload_customer)
        st.session_state["watching_job"] = job_id
        st.session_state["uploader_nonce"] = nonce + 1
        st.session_state["job_started_notice"] = (
            f"Started processing {len(uploaded_files)} photo(s) for {upload_customer} in the background - "
            "you don't need to keep this page open. You can upload another batch right away."
        )
        st.rerun()

    st.fragment(run_every=3 if any(job_display_status(m) == "processing" for m in list_jobs()) else None)(render_jobs_panel)()

    if "debug_rows" in st.session_state and st.session_state["debug_rows"]:
        with st.expander("Per-photo counts (for checking accuracy)"):
            st.dataframe(pd.DataFrame(st.session_state["debug_rows"]), use_container_width=True)

    if st.session_state.get("job_errors"):
        st.warning("Some photos had problems:\n\n" + "\n".join(st.session_state["job_errors"]))

    if "results" in st.session_state:
        items = st.session_state["results"]
        st.subheader(f"Marked rows found: {len(items)}")
        st.caption(f"Customer: {st.session_state.get('review_customer', '?')}")

        flagged = [item for item in items if item.get("needs_review")]
        committed = st.session_state.get("review_committed", False) or not flagged

        if flagged and not committed:
            st.warning(f"⚠️ {len(flagged)} row(s) need a quick double-check before the results are shown.")
            st.caption("Fix the value if it's wrong, or check \"Ignore\" to drop a duplicate/bad row from the export.")
            review_crops = st.session_state.get("review_crops", {})
            for item in flagged:
                review_id = item.get("review_id")
                images = review_crops.get(review_id) or {}
                render_review_row(item, review_id, images.get("small"), images.get("full"))

            if st.button("Commit review and show results"):
                record_review_outcomes(flagged)
                st.session_state["review_committed"] = True
                committed = True

        if items and committed:
            resolved_items = apply_review_overrides(items)
            # a review-time item_no correction can create a NEW duplicate that didn't exist
            # at extraction time - re-check before export, not just once up front
            resolved_items = resolve_duplicate_item_codes(resolved_items)
            high_items = find_high_value_items(resolved_items)

            show_results = True
            if high_items:
                show_results = render_qty_confirmation_gate(high_items, key_prefix="manual_upload")

            if not show_results:
                st.info("Confirm the quantities above to see the results and export.")
            else:
                if not st.session_state.get("report_logged"):
                    agreed, disagreed = tally_human_agreement(flagged)
                    debug_rows_data = st.session_state.get("debug_rows", [])
                    chosen = st.session_state.get("review_customer")
                    log_batch_report(
                        chosen,
                        f"{chosen}_order_{time.strftime('%Y%m%d_%H%M%S')}",
                        st.session_state.get("num_photos", 0),
                        sum(d.get("no_escalation_needed", 0) for d in debug_rows_data),
                        sum(d.get("resolved_by_premium", 0) for d in debug_rows_data),
                        len(flagged), agreed, disagreed,
                    )
                    record_catalog_learning(resolved_items)
                    finish_job(st.session_state.get("loaded_job_id"))
                    st.session_state["report_logged"] = True

                df = pd.DataFrame(resolved_items).drop(columns=["review_id"], errors="ignore").rename(columns=EXPORT_COLUMN_RENAME)
                edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

                chosen_customer = st.session_state.get("review_customer")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_filename = f"{chosen_customer}_order_{timestamp}"

                csv_bytes = edited_df.to_csv(index=False).encode("utf-8")
                excel_buffer = io.BytesIO()
                edited_df.to_excel(excel_buffer, index=False, engine="openpyxl")

                upload_filename = f"{chosen_customer} For Upload {time.strftime('%Y-%m-%d')}.xlsx"
                upload_buffer = io.BytesIO()
                build_upload_dataframe(edited_df).to_excel(upload_buffer, index=False, engine="openpyxl")

                parent_path = st.session_state.get("parent_folder_path", "").strip()
                if parent_path and Path(parent_path).exists():
                    output_dir = sv_paths(Path(parent_path), chosen_customer)["output"]
                    output_dir.mkdir(parents=True, exist_ok=True)
                    saved_path = output_dir / f"{base_filename}.csv"
                    saved_path.write_bytes(csv_bytes)
                    st.caption(f"Also saved to {saved_path}")

                if not st.session_state.get("auto_downloaded"):
                    trigger_browser_download([
                        (excel_buffer.getvalue(), f"{base_filename}.xlsx", XLSX_MIME),
                        (upload_buffer.getvalue(), upload_filename, XLSX_MIME),
                    ])
                    st.session_state["auto_downloaded"] = True

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button("Download CSV", csv_bytes, f"{base_filename}.csv", "text/csv")
                with col2:
                    st.download_button("Download Excel (again)", excel_buffer.getvalue(), f"{base_filename}.xlsx", XLSX_MIME)
                with col3:
                    st.download_button("Download upload file (again)", upload_buffer.getvalue(), upload_filename, XLSX_MIME)

                st.success(
                    f"✅ Order finished — 2 files have been downloaded: {base_filename}.xlsx and {upload_filename}"
                )
        elif not items:
            st.info("No handwritten-marked rows were found in these photos.")

if tab_batches is not None:
    with tab_batches:
        st.subheader("Customer Batches")
        st.caption(
            "Watches each SVn subfolder under the parent folder for new photos. Clean batches produce "
            "output automatically; batches needing review show up here, color-coded, until committed."
        )
        customer_batches_section()

with tab_reports:
    st.subheader("Usage Reports")
    st.caption("Cumulative within a month; a new file starts each month.")
    available_months = list_available_reports()
    if not available_months:
        st.info("No batches committed yet this month or any prior month - nothing to report.")
    else:
        selected_month = st.selectbox("Month", available_months, format_func=format_month_label)
        report_file = report_path_for(selected_month)
        report_df = pd.read_csv(report_file)
        st.dataframe(report_df, use_container_width=True)
        st.download_button(
            f"Download {format_month_label(selected_month)} report (CSV)",
            report_file.read_bytes(),
            report_file.name,
            "text/csv",
        )

with tab_settings:
    st.subheader("Settings")
    st.caption("Change the OpenAI API key this app uses. Takes effect immediately, no restart needed.")

    current_key = os.getenv("OPENAI_API_KEY", "")
    masked = f"{'•' * max(len(current_key) - 4, 4)}{current_key[-4:]}" if current_key else "(none set)"
    st.text_input("Current key", value=masked, disabled=True)

    new_key = st.text_input("New API key", type="password", placeholder="sk-...")
    col_save, col_test = st.columns(2)
    with col_save:
        if st.button("Save and apply"):
            if new_key.strip():
                apply_new_api_key(new_key)
                st.success("Key saved and applied to this session.")
                st.rerun()
            else:
                st.warning("Paste a key before saving.")
    with col_test:
        if st.button("Test current key"):
            try:
                client.chat.completions.create(
                    model=CHEAP_MODEL,
                    max_completion_tokens=5,
                    messages=[{"role": "user", "content": "Reply with: OK"}],
                )
                st.success("Key works - test call succeeded.")
            except Exception as e:
                st.error(f"Test call failed: {e}")

    st.divider()
    st.subheader("Item Catalog")
    st.caption(
        "A master Item Number / Brand / Description reference (.xlsx). Used to catch misread "
        "item codes: a code that's never been seen with the description it's paired with gets "
        "auto-corrected when the fix is unambiguous, or flagged for review otherwise."
    )

    if ITEM_CATALOG_PATH.exists():
        try:
            row_count = len(pd.read_excel(ITEM_CATALOG_PATH))
            updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(ITEM_CATALOG_PATH.stat().st_mtime))
            st.success(f"Catalog loaded: {row_count} item(s), last updated {updated}.")
        except (ValueError, KeyError, OSError) as e:
            st.error(f"Catalog file exists but couldn't be read: {e}")
    else:
        st.info("No catalog uploaded yet - item-code cross-checking is off until one is.")

    catalog_upload = st.file_uploader("Upload catalog (.xlsx)", type=["xlsx"], key="catalog_upload")
    if catalog_upload and st.button("Save catalog"):
        ITEM_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ITEM_CATALOG_PATH.write_bytes(catalog_upload.getvalue())
        st.success("Catalog saved.")
        st.rerun()

    learned = load_learned_associations()
    pending_new = sum(1 for e in learned["new_items"].values() if e.get("confirmed_count", 0) < NEW_ITEM_CONFIRMATION_THRESHOLD)
    confirmed_new = len(learned["new_items"]) - pending_new
    st.caption(
        f"Learned: {confirmed_new} new item(s) confirmed and now trusted like a catalog entry, "
        f"{pending_new} still awaiting a {NEW_ITEM_CONFIRMATION_THRESHOLD}nd confirmation, "
        f"{len(learned['correction_log'])} auto-correction(s) made so far."
    )
    if learned["correction_log"]:
        with st.expander("Auto-correction history"):
            st.dataframe(pd.DataFrame(learned["correction_log"][::-1]), use_container_width=True)
