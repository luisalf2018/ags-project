import base64
import difflib
import io
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from html import escape as html_escape
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

# ============================== language (English / Spanish) ==============================
# Every user-visible string lives here as (English, Spanish). The chosen language is UI-only:
# exported files keep their stable English headers, since they feed an ordering system.

LANGS = {"en": "English", "es": "Español"}

_T = {
    "app_title": ("AI Handwritten Qty Extractor", "Extractor de Cantidades Manuscritas con IA"),
    "app_caption": (
        "Upload photos of order sheets. Rows with a handwritten number will be pulled out into a table you can export.",
        "Suba fotos de las hojas de pedido. Las filas con un número escrito a mano se extraerán a una tabla que puede exportar.",
    ),
    "api_key_missing": (
        "No API key found. Open the .env file in the project folder and paste your OpenAI API key "
        "after OPENAI_API_KEY=, or use the ⚙️ Settings tab once the app is running.",
        "No se encontró la clave de API. Abra el archivo .env en la carpeta del proyecto y pegue su clave de OpenAI "
        "después de OPENAI_API_KEY=, o use la pestaña ⚙️ Configuración cuando la app esté funcionando.",
    ),
    "tab_upload": ("📤 Upload Photos", "📤 Subir Fotos"),
    "tab_batches": ("🏢 Customer Batches", "🏢 Lotes por Cliente"),
    "tab_reports": ("📊 Reports", "📊 Reportes"),
    "tab_settings": ("⚙️ Settings", "⚙️ Configuración"),
    # --- upload tab ---
    "drop_photos": ("Drop photos here", "Suelte las fotos aquí"),
    "customer": ("Customer", "Cliente"),
    "select_customer": ("Select a customer...", "Seleccione un cliente..."),
    "other_customer": ("Other Customer", "Otro Cliente"),
    "customer_help_cloud": (
        "Required - names the output file after the customer.",
        "Obligatorio: el archivo de salida llevará el nombre del cliente.",
    ),
    "customer_help_local": (
        "Required - names the output file after the customer and, if a parent folder "
        "is configured in Customer Batches, also saves a copy into that customer's Output folder.",
        "Obligatorio: el archivo de salida llevará el nombre del cliente y, si hay una carpeta principal "
        "configurada en Lotes por Cliente, también guarda una copia en la carpeta Output de ese cliente.",
    ),
    "select_customer_warning": (
        "Select a customer above before extracting.",
        "Seleccione un cliente arriba antes de procesar.",
    ),
    "go_process": ("GO PROCESS", "¡A PROCESAR!"),
    "go_hint": (
        "Add photos and pick a customer to enable this button.",
        "Agregue fotos y elija un cliente para activar este botón.",
    ),
    "job_started": (
        "Started processing {n} photo(s) for {customer} in the background - you don't need to keep this page "
        "open. You can upload another batch right away.",
        "Se empezaron a procesar {n} foto(s) de {customer} en segundo plano: no necesita mantener esta página "
        "abierta. Puede subir otro lote de inmediato.",
    ),
    "jobs_title": ("Orders in progress", "Órdenes en proceso"),
    "jobs_caption": (
        "Photos are processed in the background on the server - you can close this tab or let your "
        "screen sleep, then come back and pick up here.",
        "Las fotos se procesan en segundo plano en el servidor: puede cerrar esta pestaña o dejar que la "
        "pantalla se suspenda, y luego volver y continuar aquí.",
    ),
    "job_photos": ("{n} photo(s)", "{n} foto(s)"),
    "status_processing": ("PROCESSING", "PROCESANDO"),
    "job_progress": ("{done} of {total} photos done", "{done} de {total} fotos listas"),
    "status_ready": ("READY TO REVIEW", "LISTO PARA REVISAR"),
    "open_review": ("OPEN TO REVIEW", "ABRIR PARA REVISAR"),
    "job_interrupted": (
        "⚠️ Interrupted (the server restarted mid-run) - your photos are saved, restart to continue.",
        "⚠️ Interrumpido (el servidor se reinició durante el proceso): sus fotos están guardadas, reinicie para continuar.",
    ),
    "job_failed": ("❌ Failed: {error}", "❌ Falló: {error}"),
    "restart": ("Restart", "Reiniciar"),
    "dismiss": ("Dismiss", "Descartar"),
    "dismiss_confirm": (
        "Delete this order and its photos? This can't be undone.",
        "¿Eliminar este pedido y sus fotos? No se puede deshacer.",
    ),
    "dismiss_yes": ("Yes, dismiss", "Sí, descartar"),
    "dismiss_no": ("Keep it", "Conservarlo"),
    "draft_restored": (
        "Your review edits are saved automatically - if the page reloads, they come back when you reopen this order.",
        "Sus cambios de revisión se guardan automáticamente: si la página se recarga, vuelven al reabrir este pedido.",
    ),
    "photo_counts": (
        "Per-photo counts (for checking accuracy)",
        "Conteo por foto (para verificar la precisión)",
    ),
    "photo_problems": ("Some photos had problems:", "Algunas fotos tuvieron problemas:"),
    "rows_found": ("Marked rows found: {n}", "Filas marcadas encontradas: {n}"),
    "customer_label": ("Customer: {name}", "Cliente: {name}"),
    "review_needed": (
        "⚠️ {n} row(s) need a quick double-check before the results are shown.",
        "⚠️ {n} fila(s) necesitan una revisión rápida antes de mostrar los resultados.",
    ),
    "review_hint": (
        "Fix the value if it's wrong, or check \"Ignore\" to drop a duplicate/bad row from the export.",
        "Corrija el valor si es incorrecto, o marque \"Ignorar\" para excluir una fila duplicada o errónea de la exportación.",
    ),
    "commit_review": ("Commit review and show results", "Confirmar revisión y mostrar resultados"),
    "no_preview": ("(no preview available)", "(vista previa no disponible)"),
    "cant_find": (
        "🔍 Can't find item {item}? Show the full section",
        "🔍 ¿No encuentra el artículo {item}? Mostrar la sección completa",
    ),
    "item_num": ("Item #", "Artículo #"),
    "catalog_item_for_upc": ("Catalog item # for this UPC: {code}", "Artículo # del catálogo para este UPC: {code}"),
    "hw_value": ("Handwritten value", "Valor manuscrito"),
    "ignore_row": ("Ignore this row", "Ignorar esta fila"),
    "qty_gate": (
        "⚠️ Are you sure these quantities are correct? {n} item(s) have a quantity of {threshold} or higher.",
        "⚠️ ¿Está seguro de que estas cantidades son correctas? {n} artículo(s) tienen una cantidad de {threshold} o más.",
    ),
    "qty_confirmed": (
        "✅ Quantities confirmed ({n} item(s) of {threshold} or higher).",
        "✅ Cantidades confirmadas ({n} artículo(s) de {threshold} o más).",
    ),
    "qty_edit": ("Edit the confirmed quantities", "Editar las cantidades confirmadas"),
    "quantity": ("Quantity", "Cantidad"),
    "confirm_qty": ("Confirm quantities and continue", "Confirmar cantidades y continuar"),
    "confirm_qty_first": (
        "Confirm the quantities above to see the results and export.",
        "Confirme las cantidades de arriba para ver los resultados y exportar.",
    ),
    "also_saved": ("Also saved to {path}", "También guardado en {path}"),
    "dl_csv": ("Download CSV", "Descargar CSV"),
    "dl_excel_again": ("Download Excel (again)", "Descargar Excel (otra vez)"),
    "dl_upload_again": ("Download upload file (again)", "Descargar archivo para cargar (otra vez)"),
    "order_finished": (
        "✅ Order finished — 2 files have been downloaded: {a} and {b}",
        "✅ Orden terminada — se descargaron 2 archivos: {a} y {b}",
    ),
    "no_rows": (
        "No handwritten-marked rows were found in these photos.",
        "No se encontraron filas con marcas manuscritas en estas fotos.",
    ),
    "close_order": ("Close this order", "Cerrar esta orden"),
    # --- customer batches tab (local only) ---
    "batches_title": ("Customer Batches", "Lotes por Cliente"),
    "batches_caption": (
        "Watches each SVn subfolder under the parent folder for new photos. Clean batches produce "
        "output automatically; batches needing review show up here, color-coded, until committed.",
        "Vigila cada subcarpeta SVn de la carpeta principal en busca de fotos nuevas. Los lotes limpios generan "
        "el archivo automáticamente; los lotes que necesitan revisión aparecen aquí, con colores, hasta que se confirmen.",
    ),
    "parent_folder": (
        "Parent folder (contains SV1 through SV13 subfolders)",
        "Carpeta principal (contiene las subcarpetas SV1 a SV13)",
    ),
    "parent_folder_help": (
        "Drop photos directly into each SVn folder - it's created automatically. Output is shared "
        "(files are named per-customer), and Processed keeps a per-customer subfolder. Remembered between sessions.",
        "Suelte las fotos directamente en cada carpeta SVn (se crea automáticamente). Output es compartida "
        "(los archivos llevan el nombre del cliente) y Processed guarda una subcarpeta por cliente. Se recuerda entre sesiones.",
    ),
    "watch_toggle": ("Watch all customer folders", "Vigilar todas las carpetas de clientes"),
    "not_watching": (
        "Not watching new photos - existing pending reviews below are still shown and actionable.",
        "No se están vigilando fotos nuevas: las revisiones pendientes de abajo se siguen mostrando y se pueden atender.",
    ),
    "enter_parent": ("Enter a parent folder path above.", "Escriba arriba la ruta de la carpeta principal."),
    "folder_missing": ("Folder does not exist: {path}", "La carpeta no existe: {path}"),
    "banner_flagged": (
        "🚨 {n} row(s) across customer folders need your review before their output "
        "can be produced — see the \"Customer Batches\" tab.",
        "🚨 {n} fila(s) en las carpetas de clientes necesitan su revisión antes de poder generar su archivo "
        "— vea la pestaña \"Lotes por Cliente\".",
    ),
    "sv_in_progress": ("{n} photo(s) in progress", "{n} foto(s) en proceso"),
    "sv_need_review": ("{n} row(s) need review", "{n} fila(s) necesitan revisión"),
    "sv_done_today": ("already processed today", "ya procesado hoy"),
    "sv_up_to_date": ("up to date", "al día"),
    "review_sv": ("Review {sv}", "Revisar {sv}"),
    "batch_line": (
        "**Batch {id}** — {n} row(s), {m} need review",
        "**Lote {id}** — {n} fila(s), {m} necesitan revisión",
    ),
    "commit_output": ("Commit review and produce output", "Confirmar revisión y generar archivo"),
    "saved_file": ("✅ Saved {name} with {n} row(s).", "✅ Se guardó {name} con {n} fila(s)."),
    "all_ignored": (
        "Every row in this batch was marked ignore - no output file was produced.",
        "Todas las filas de este lote se marcaron como ignoradas: no se generó ningún archivo.",
    ),
    "detected_working": (
        "📸 {sv}: detected {n} photo(s). Working on them...",
        "📸 {sv}: se detectaron {n} foto(s). Procesándolas...",
    ),
    "starting": ("Starting...", "Iniciando..."),
    "finished_photo": ("Finished {done}/{total} ({name})", "Terminada {done}/{total} ({name})"),
    "failed_photo": ("Failed {name} ({done}/{total})", "Falló {name} ({done}/{total})"),
    "done_text": ("Done.", "Listo."),
    "flagged_warning": (
        "🚨 {sv}: processed {done} photo(s), found {n} marked row(s), but {m} need review before output can be "
        "produced. No output file was written yet.",
        "🚨 {sv}: se procesaron {done} foto(s), se encontraron {n} fila(s) marcadas, pero {m} necesitan revisión "
        "antes de generar el archivo. Todavía no se escribió ningún archivo.",
    ),
    "clean_success": (
        "✅ {sv}: processed {done} photo(s), found {n} marked row(s) - saved to {name}.",
        "✅ {sv}: se procesaron {done} foto(s), se encontraron {n} fila(s) marcadas - guardado en {name}.",
    ),
    "none_found": (
        "✅ {sv}: processed {done} photo(s), no marked rows found.",
        "✅ {sv}: se procesaron {done} foto(s), no se encontraron filas marcadas.",
    ),
    "photos_failed": (
        "{sv}: {n} photo(s) failed and were left in Input: {list}",
        "{sv}: {n} foto(s) fallaron y se dejaron en Input: {list}",
    ),
    "settling": (
        "📸 {sv}: detected {n} photo(s), waiting for the folder to settle...",
        "📸 {sv}: se detectaron {n} foto(s), esperando a que la carpeta se estabilice...",
    ),
    # --- reports tab ---
    "reports_title": ("Usage Reports", "Reportes de Uso"),
    "reports_caption": (
        "Cumulative within a month; a new file starts each month.",
        "Acumulado dentro del mes; cada mes se inicia un archivo nuevo.",
    ),
    "no_reports": (
        "No batches committed yet this month or any prior month - nothing to report.",
        "Aún no se ha confirmado ningún lote este mes ni en meses anteriores: no hay nada que reportar.",
    ),
    "month": ("Month", "Mes"),
    "dl_report": ("Download {month} report (CSV)", "Descargar reporte de {month} (CSV)"),
    # --- settings tab ---
    "settings_title": ("Settings", "Configuración"),
    "settings_caption": (
        "Change the OpenAI API key this app uses. Takes effect immediately, no restart needed.",
        "Cambie la clave de API de OpenAI que usa esta app. Se aplica de inmediato, sin reiniciar.",
    ),
    "current_key": ("Current key", "Clave actual"),
    "none_set": ("(none set)", "(ninguna)"),
    "new_key": ("New API key", "Nueva clave de API"),
    "save_apply": ("Save and apply", "Guardar y aplicar"),
    "test_key": ("Test current key", "Probar clave actual"),
    "key_saved": ("Key saved and applied to this session.", "Clave guardada y aplicada a esta sesión."),
    "paste_key_first": ("Paste a key before saving.", "Pegue una clave antes de guardar."),
    "key_works": ("Key works - test call succeeded.", "La clave funciona: la llamada de prueba fue exitosa."),
    "test_failed": ("Test call failed: {e}", "La llamada de prueba falló: {e}"),
    "catalog_title": ("Item Catalog", "Catálogo de Artículos"),
    "catalog_caption": (
        "A master Item Number / Brand / Description reference (.xlsx). Used to catch misread "
        "item codes: a code that's never been seen with the description it's paired with gets "
        "auto-corrected when the fix is unambiguous, or flagged for review otherwise.",
        "Una referencia maestra de Número de Artículo / Marca / Descripción (.xlsx). Sirve para detectar códigos "
        "mal leídos: un código que nunca se ha visto con la descripción a la que está asociado se corrige "
        "automáticamente cuando la corrección es inequívoca, o se marca para revisión en caso contrario.",
    ),
    "catalog_loaded": (
        "Catalog loaded: {n} item(s), last updated {when}.",
        "Catálogo cargado: {n} artículo(s), última actualización {when}.",
    ),
    "catalog_unreadable": (
        "Catalog file exists but couldn't be read: {e}",
        "El archivo del catálogo existe pero no se pudo leer: {e}",
    ),
    "no_catalog": (
        "No catalog uploaded yet - item-code cross-checking is off until one is.",
        "Aún no se ha subido un catálogo: la verificación de códigos de artículo está desactivada hasta que se suba uno.",
    ),
    "upload_catalog": ("Upload catalog (.xlsx)", "Subir catálogo (.xlsx)"),
    "save_catalog": ("Save catalog", "Guardar catálogo"),
    "catalog_saved": ("Catalog saved.", "Catálogo guardado."),
    "learned_caption": (
        "Learned: {confirmed} new item(s) confirmed and now trusted like a catalog entry, "
        "{pending} still awaiting a {thr}nd confirmation, {aliases} alternate description wording(s) accepted, "
        "{corrections} auto-correction(s) made so far.",
        "Aprendido: {confirmed} artículo(s) nuevo(s) confirmado(s) y ya tratados como una entrada del catálogo, "
        "{pending} aún esperan la confirmación n.º {thr}, {aliases} redacción(es) alternativa(s) de descripción "
        "aceptada(s), {corrections} corrección(es) automática(s) hasta ahora.",
    ),
    "correction_history": ("Auto-correction history", "Historial de correcciones automáticas"),
}

MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
}

# display-only column labels (the CSV/Excel files always keep the stable English headers)
COLUMN_LABELS_ES = {
    "timestamp": "fecha y hora", "customer": "cliente", "batch_id": "id del lote", "num_photos": "fotos",
    "items_no_escalation": "artículos sin revisión premium", "items_resolved_by_premium": "resueltos por el modelo premium",
    "items_reached_human_review": "llegaron a revisión humana", "items_human_agreed": "humano de acuerdo",
    "items_human_disagreed": "humano corrigió", "from": "de", "to": "a", "description": "descripción",
    "file": "archivo", "deskew": "enderezado", "rotation_applied_degrees": "rotación aplicada (grados)",
    "rows_seen_across_crops": "filas vistas en los recortes", "escalated_to_premium": "escaladas al modelo premium",
    "dropped_as_false_detection": "descartadas como falsa detección",
    "auto_suppressed_known_artifact": "suprimidas (artefacto conocido)",
    "no_escalation_needed": "sin necesidad de escalar", "resolved_by_premium": "resueltas por el modelo premium",
    "items_before_dedupe": "artículos antes de depurar duplicados", "items_returned": "artículos devueltos",
    "flagged_for_review": "marcados para revisión",
    "item_no": "artículo #", "Qty": "Cant.", "brand": "marca", "pack": "paquete", "size": "tamaño",
    "old_item": "artículo anterior", "needs_review": "requiere revisión", "review_reason": "motivo de revisión",
    "source_image": "foto de origen", "confidence": "confianza", "mark_side": "lado de la marca",
    "appears_altered": "parece alterada", "escalated": "escalada",
}


def current_lang() -> str:
    try:
        code = st.session_state.get("lang")
    except Exception:
        code = None
    return code if code in LANGS else "en"


def t(key: str, **kwargs) -> str:
    en, es = _T[key]
    text = es if current_lang() == "es" else en
    return text.format(**kwargs) if kwargs else text


def column_label(name: str) -> str:
    return COLUMN_LABELS_ES.get(name, name) if current_lang() == "es" else name


def translate_columns(df):
    return df.rename(columns={c: column_label(c) for c in df.columns})


def month_name(month_index: int) -> str:
    return MONTHS[current_lang()][month_index - 1]


# the review-reason texts are written in English inside the extraction pipeline (which runs
# in a detached thread with no language), so they're translated at display time instead
_REASON_PATTERNS_ES = [
    (r"^model was not confident in this reading( \(even after premium re-check\))?$",
     lambda m: "el modelo no tuvo confianza en esta lectura" + (" (incluso después de la revisión con el modelo premium)" if m.group(1) else "")),
    (r"^mark appears crossed out / scratched / voided - please verify it wasn't cancelled$",
     lambda m: "la marca parece tachada / raspada / anulada: verifique que no haya sido cancelada"),
    (r"^same row read differently across overlapping crops - please verify$",
     lambda m: "la misma fila se leyó distinto en recortes superpuestos: verifique"),
    (r"^handwritten value is (\d+) or higher - please verify$",
     lambda m: f"el valor manuscrito es {m.group(1)} o mayor: verifique"),
    (r"^handwritten value read as 7 - easily confused with 1, please verify$",
     lambda m: "valor manuscrito leído como 7 (se confunde fácilmente con 1): verifique"),
    (r"^item code could not be read - please enter it from the sheet$",
     lambda m: "no se pudo leer el código de artículo: ingréselo desde la hoja"),
    (r"^item code not found in the catalog \(may be a new item\) - please verify$",
     lambda m: "código de artículo no encontrado en el catálogo (podría ser un artículo nuevo): verifique"),
    (r"^item code's description doesn't match the catalog - please verify$",
     lambda m: "la descripción del código de artículo no coincide con el catálogo: verifique"),
    (r"^found by only one of two independent readings - please verify it is really marked$",
     lambda m: "encontrada en solo una de dos lecturas independientes: verifique que realmente esté marcada"),
    (r"^two independent readings disagree on the handwritten value \((.*) vs (.*)\)$",
     lambda m: f"dos lecturas independientes no coinciden en el valor manuscrito ({m.group(1)} vs {m.group(2)})"),
    (r"^item code (.*) appears more than once with different handwritten values - please verify$",
     lambda m: f"el código de artículo {m.group(1)} aparece más de una vez con valores manuscritos distintos: verifique"),
]


def translate_reason(text: str) -> str:
    if current_lang() != "es" or not text:
        return text
    parts = []
    for part in text.split("; "):
        for pattern, render in _REASON_PATTERNS_ES:
            m = re.match(pattern, part)
            if m:
                part = render(m)
                break
        parts.append(part)
    return "; ".join(parts)


def display_customer(name: str | None) -> str:
    return t("other_customer") if name == "Other Customer" else (name or "?")


def format_job_time(epoch: float) -> str:
    lt = time.localtime(epoch)
    if current_lang() == "es":
        return f"{lt.tm_mday} {month_name(lt.tm_mon)[:3]} {time.strftime('%H:%M', lt)}"
    return time.strftime("%b %d %I:%M %p", lt)


# the language is decided BEFORE set_page_config so the browser-tab title follows it too;
# it's mirrored into the URL (?lang=es) so a refresh or a reconnect after the screen sleeps
# doesn't silently flip the app back to English
if "lang" not in st.session_state:
    _qp_lang = st.query_params.get("lang", "en")
    st.session_state["lang"] = _qp_lang if _qp_lang in LANGS else "en"

st.set_page_config(page_title=t("app_title"), layout="wide")

APP_CSS = """<style>
.st-key-lang { display:flex; justify-content:flex-end; }
.st-key-go_process button { background:#0B3A8F !important; border:2px solid #082B6B !important; color:#fff !important; min-height:3.4rem; padding:0.6rem 1.6rem; box-shadow:0 2px 6px rgba(11,58,143,.35); }
.st-key-go_process button p { color:#fff !important; font-size:1.2rem !important; font-weight:800 !important; letter-spacing:.08em; }
.st-key-go_process button:hover { background:#0A2F73 !important; }
.st-key-go_process button:disabled { background:#9AA3B2 !important; border-color:#8992A3 !important; box-shadow:none; cursor:not-allowed; }
.st-key-go_process button:disabled p { color:#F1F3F6 !important; }
[class*="st-key-open_"] button { background:#1B5E20 !important; border:2px solid #124116 !important; color:#fff !important; min-height:3.4rem; box-shadow:0 2px 6px rgba(27,94,32,.35); }
[class*="st-key-open_"] button p { color:#fff !important; font-size:1.05rem !important; font-weight:800 !important; letter-spacing:.06em; }
[class*="st-key-open_"] button:hover { background:#154A19 !important; }
[class*="st-key-dismiss_"] button { background:#C62828 !important; border:1px solid #8E1B1B !important; min-height:2rem; padding:0.1rem 0.9rem; }
[class*="st-key-dismiss_"] button p { color:#fff !important; font-size:.85rem !important; font-weight:700 !important; }
[class*="st-key-dismiss_"] button:hover { background:#A61E1E !important; }
.job-card { border-radius:10px; padding:16px 22px; color:#fff; margin:6px 0; }
.job-card.processing { background:#C75B00; animation:jobpulse 1.6s ease-in-out infinite; }
.job-card.ready { background:#1B5E20; }
.job-card .job-status { font-size:1.4rem; font-weight:800; letter-spacing:.06em; }
.job-card .job-title { font-size:1rem; margin-top:2px; }
.job-card .job-bar { height:10px; background:rgba(255,255,255,.3); border-radius:6px; margin-top:10px; overflow:hidden; }
.job-card .job-bar-fill { height:100%; background:#fff; border-radius:6px; }
.job-card .job-sub { font-size:.9rem; margin-top:6px; }
@keyframes jobpulse { 0%,100% { box-shadow:0 0 0 0 rgba(199,91,0,.55); } 50% { box-shadow:0 0 0 8px rgba(199,91,0,0); } }
</style>"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# The file-upload box's own text ("Upload", "200MB per file...") is built into Streamlit and
# can't be reached through the app's translation table, so in Spanish it's swapped via CSS.
# Scoped per uploader because the accepted file types differ. If a future Streamlit release
# changes that markup this quietly stops applying and the box just stays in English.
_DZ = '[data-testid="stFileUploaderDropzone"]'
_PHOTOS, _CATALOG = '[class*="st-key-uploader_"]', ".st-key-catalog_upload"
UPLOADER_CSS_ES = "<style>" + "".join(
    f'{scope} {_DZ} [data-testid="stMarkdownContainer"] p{{font-size:0 !important;}}'
    f'{scope} {_DZ} [data-testid="stMarkdownContainer"] p::after{{content:"{button}";font-size:1rem;}}'
    f'{scope} {_DZ} [data-testid="stFileUploaderDropzoneInstructions"] span{{font-size:0 !important;}}'
    f'{scope} {_DZ} [data-testid="stFileUploaderDropzoneInstructions"] span::after{{content:"{limit}";font-size:0.875rem;}}'
    for scope, button, limit in (
        (_PHOTOS, "Subir fotos", "200 MB por archivo • JPG, PNG"),
        (_CATALOG, "Subir archivo", "200 MB por archivo • XLSX"),
    )
) + "</style>"
if current_lang() == "es":
    st.markdown(UPLOADER_CSS_ES, unsafe_allow_html=True)

_title_col, _lang_col = st.columns([5, 1.4], vertical_alignment="center")
with _title_col:
    st.title(t("app_title"))
with _lang_col:
    st.radio("Language / Idioma", list(LANGS), format_func=LANGS.get, key="lang", horizontal=True, label_visibility="collapsed")
if st.query_params.get("lang") != st.session_state["lang"]:
    st.query_params["lang"] = st.session_state["lang"]
st.caption(t("app_caption"))

ENV_PATH = Path(__file__).parent.parent / ".env"

# Cloud deployments (no shared local drive to watch) set AGS_CLOUD_MODE=1 to hide the
# folder-monitoring tab, and AGS_DATA_DIR to point item memory / usage reports at a
# mounted persistent volume instead of the app folder.
CLOUD_MODE = os.getenv("AGS_CLOUD_MODE", "").strip().lower() in ("1", "true", "yes")
DATA_DIR = Path(os.environ["AGS_DATA_DIR"]) if os.getenv("AGS_DATA_DIR") else Path(__file__).parent

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key or api_key == "paste-your-key-here":
    st.error(t("api_key_missing"))
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


def to_export_df(items: list[dict]) -> pd.DataFrame:
    """Items -> the table that gets shown and exported. Internal working fields (review_id and
    anything starting with "_", e.g. the catalog-check flags) are for the app's own bookkeeping
    and must never leak into a file the user opens or uploads."""
    df = pd.DataFrame(items).drop(columns=["review_id"], errors="ignore")
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("_")])
    if "UPC" in df.columns and "item_no" in df.columns:  # UPC sits right next to the item number
        cols = [c for c in df.columns if c != "UPC"]
        cols.insert(cols.index("item_no") + 1, "UPC")
        df = df[cols]
        df["UPC"] = df["UPC"].fillna("")
    return df.rename(columns=EXPORT_COLUMN_RENAME)


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

The ITEM CODE is the most important field of every row - the row is useless without it. Never leave item_no
empty when ANY printed code is legible in that row. Some sheets print a short item number at the far left
(which the photo can cut off or blur) AND a long code (for example 12 digits) in a column to the right of the
description and the handwritten mark. Use the item-number column when it is fully legible; when it is cut off,
blurred or missing for that row, use the long printed code from that row's own line instead. Copy the code
digit by digit exactly as printed. If you can only make out part of a code, give your best reading and set
confidence to "low" rather than leaving it empty. The code must come from the same printed line as the
description, never from a neighboring row.

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


HIGH_QTY_THRESHOLD = 6  # a handwritten quantity at or above this always needs a human look (5 is fine)


def is_suspiciously_high(value, threshold: float = HIGH_QTY_THRESHOLD) -> bool:
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

    def fill_blank_codes(blank_side: list[dict], other_side: list[dict]) -> None:
        """One reading found the code, the other left it empty: they're the same row, so use the
        code that was read instead of leaving a codeless half-row (and its coded twin) behind."""
        coded_by_desc: dict[str, list[dict]] = {}
        for other in other_side:
            desc = str(other.get("description", "")).strip().lower()
            if desc and str(other.get("item_no", "")).strip():
                coded_by_desc.setdefault(desc, []).append(other)
        for item in blank_side:
            desc = str(item.get("description", "")).strip().lower()
            if desc and not str(item.get("item_no", "")).strip() and len(coded_by_desc.get(desc, [])) == 1:
                item["item_no"] = coded_by_desc[desc][0]["item_no"]

    fill_blank_codes(items_a, items_b)
    fill_blank_codes(items_b, items_a)

    def unify_descriptions_by_code() -> None:
        """Same printed code read by both runs but with a slightly different description ('STRW BAN'
        vs 'STRWB BAN'): that is one row, not two rows each 'found by only one reading'."""
        keys_a, keys_b = {row_key(i) for i in items_a}, {row_key(i) for i in items_b}
        lone_a: dict[str, list[dict]] = {}
        lone_b: dict[str, list[dict]] = {}
        for it in items_a:
            code = str(it.get("item_no", "")).strip().lower()
            if code and row_key(it) not in keys_b:
                lone_a.setdefault(code, []).append(it)
        for it in items_b:
            code = str(it.get("item_no", "")).strip().lower()
            if code and row_key(it) not in keys_a:
                lone_b.setdefault(code, []).append(it)
        for code, a_rows in lone_a.items():
            b_rows = lone_b.get(code, [])
            if len(a_rows) == 1 and len(b_rows) == 1:
                b_rows[0]["description"] = a_rows[0].get("description", "")

    unify_descriptions_by_code()

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
            if not str(item.get("item_no", "")).strip():
                # the code is the primary field - a row without one must never pass silently
                reasons.append("item code could not be read - please enter it from the sheet")
            if str(item.get("confidence", "high")).lower() != "high":
                suffix = " (even after premium re-check)" if item.get("escalated") else ""
                reasons.append(f"model was not confident in this reading{suffix}")
            if item.get("appears_altered"):
                reasons.append("mark appears crossed out / scratched / voided - please verify it wasn't cancelled")
            if row_key in conflicting:
                reasons.append("same row read differently across overlapping crops - please verify")
            if is_suspiciously_high(item.get("handwritten_number")):
                reasons.append(f"handwritten value is {HIGH_QTY_THRESHOLD} or higher - please verify")
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


def review_edit(review_id: str) -> tuple[bool, str | None, str | None]:
    """(ignored, handwritten value, item #) the user chose for a flagged row. Once the review is
    committed the values are frozen in session state: Streamlit drops a widget's state on the first
    run that doesn't draw it, so reading the widget keys later would silently revert every edit."""
    frozen = st.session_state.get("review_overrides")
    if frozen is not None:
        entry = frozen.get(review_id, {})
        return bool(entry.get("ignore")), entry.get("hw"), entry.get("item_no")
    return (
        bool(st.session_state.get(f"ignore_{review_id}")),
        st.session_state.get(f"hw_{review_id}"),
        st.session_state.get(f"itemno_{review_id}"),
    )


def collect_review_draft(flagged: list[dict]) -> dict:
    draft = {}
    for item in flagged:
        rid = item.get("review_id")
        if not rid:
            continue
        ignore, hw, item_no = review_edit(rid)
        entry = {"ignore": ignore}
        if hw is not None:
            entry["hw"] = hw
        if item_no is not None:
            entry["item_no"] = item_no
        draft[rid] = entry
    return draft


def review_draft_path(job_id: str) -> Path:
    return job_dir_for(job_id) / "review_draft.json"


def save_review_draft(job_id: str | None, draft: dict) -> None:
    """Review edits live in the browser session, which a server restart or a dropped connection
    wipes - so they're mirrored to the job's folder and restored when the order is reopened."""
    if not job_id or not job_dir_for(job_id).exists():
        return
    payload = json.dumps(draft, sort_keys=True)
    if st.session_state.get("_draft_saved") == (job_id, payload):
        return
    try:
        review_draft_path(job_id).write_text(payload)
        st.session_state["_draft_saved"] = (job_id, payload)
    except OSError:
        pass


def load_review_draft(job_id: str) -> dict:
    try:
        return json.loads(review_draft_path(job_id).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def shown_code(item: dict) -> str:
    """The code exactly as it is printed on the sheet, which is what a human reviewer can check
    against the photo: the UPC when the row was identified by one, else the item number. (Behind
    the scenes item_no already holds the catalog item number for a UPC row.)"""
    return str(item.get("UPC") or item.get("item_no", ""))


def recode_reviewed_item(edited: str) -> dict:
    """The human typed a different code in review: treat it as the code read from the sheet."""
    if upc_key(edited):
        hit = load_item_catalog().get("by_upc", {}).get(upc_key(edited))
        if hit:
            return {"item_no": hit["item_no"], "UPC": hit["upc"]}
        return {"item_no": edited, "UPC": edited}
    return {"item_no": edited, "UPC": ""}


def render_review_row(item: dict, review_id: str, crop_img, full_img=None) -> None:
    saved = st.session_state.get("review_draft", {}).get(review_id, {})
    if crop_img is not None:
        st.image(crop_img, use_container_width=True)
    else:
        st.caption(t("no_preview"))

    if full_img is not None:
        with st.expander(t("cant_find", item=shown_code(item))):
            st.image(full_img, use_container_width=True)

    cols = st.columns([1.6, 1, 1, 1])
    with cols[0]:
        st.markdown(f"**{item.get('source_image', '')}**  \n{item.get('description', '')}")
        st.caption(translate_reason(item.get("review_reason", "")))
        if item.get("UPC") and item.get("item_no") and item.get("UPC") != item.get("item_no"):
            st.caption(t("catalog_item_for_upc", code=item.get("item_no")))
    with cols[1]:
        st.text_input(
            t("item_num"),
            value=str(saved.get("item_no", shown_code(item))),
            key=f"itemno_{review_id}",
        )
    with cols[2]:
        st.text_input(
            t("hw_value"),
            value=str(saved.get("hw", item.get("handwritten_number", ""))),
            key=f"hw_{review_id}",
        )
    with cols[3]:
        st.checkbox(t("ignore_row"), value=bool(saved.get("ignore", False)), key=f"ignore_{review_id}")
    st.divider()


def apply_review_overrides(items: list[dict]) -> list[dict]:
    resolved = []
    for item in items:
        review_id = item.get("review_id")
        if item.get("needs_review") and review_id:
            ignored, edited_value, edited_item_no = review_edit(review_id)
            if ignored:
                continue
            overrides = {}
            if edited_value is not None:
                overrides["handwritten_number"] = edited_value
            if edited_item_no is not None and str(edited_item_no).strip() != shown_code(item).strip():
                overrides.update(recode_reviewed_item(str(edited_item_no).strip()))
            if overrides:
                item = {**item, **overrides}
                if not item.get("UPC"):
                    item.pop("UPC", None)
        resolved.append(item)
    return resolved


def find_high_value_items(items: list[dict]) -> list[dict]:
    return [item for item in items if is_suspiciously_high(item.get("handwritten_number"))]


def render_qty_confirmation_gate(high_items: list[dict], key_prefix: str, persist: bool = False) -> bool:
    """Final safety net before a commit actually writes output - a review-time edit (or a value
    that was never flagged for any other reason) could still be a suspiciously high quantity.
    Renders a red confirm-or-correct gate for those specific rows, mutating them in place so a
    correction here is reflected in what gets committed. Returns True once the user clicks through.

    persist=True remembers the confirmation across reruns. A bare st.button is only True during
    the single run in which it was clicked, so without this ANY later interaction (a download,
    closing the order, editing the table, switching language) would re-hide the results and put
    the gate back. The quantity inputs stay rendered (folded into an expander once confirmed) so
    a correction keeps applying on every rerun instead of silently reverting."""
    confirmed_key = f"{key_prefix}_qty_confirmed"
    confirmed = persist and st.session_state.get(confirmed_key, False)
    if confirmed:
        st.success(t("qty_confirmed", n=len(high_items), threshold=HIGH_QTY_THRESHOLD))
        holder = st.expander(t("qty_edit"))
    else:
        st.markdown(
            '<div style="background-color:#f8d7da;color:#842029;padding:10px 16px;border-radius:6px;'
            f'font-weight:600;margin-bottom:8px;">{t("qty_gate", n=len(high_items), threshold=HIGH_QTY_THRESHOLD)}</div>',
            unsafe_allow_html=True,
        )
        holder = st.container()
    with holder:
        for idx, item in enumerate(high_items):
            cols = st.columns([2, 1])
            with cols[0]:
                st.markdown(f"**{item.get('item_no', '')}** — {item.get('description', '')}")
            with cols[1]:
                item["handwritten_number"] = st.text_input(
                    t("quantity"),
                    value=str(item.get("handwritten_number", "")),
                    key=f"{key_prefix}_qty_{idx}",
                )
    if confirmed:
        return True
    clicked = st.button(t("confirm_qty"), key=f"{key_prefix}_confirm_qty")
    if clicked and persist:
        st.session_state[confirmed_key] = True
        st.rerun()  # redraw straight into the confirmed state instead of leaving the red prompt up
    return clicked


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
        if review_edit(review_id)[0]:
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
        ignored, edited_hw, edited_item_no = review_edit(review_id)
        if ignored:
            disagreed += 1
            continue
        changed_hw = edited_hw is not None and str(edited_hw) != str(item.get("handwritten_number", ""))
        changed_item_no = edited_item_no is not None and str(edited_item_no).strip() != shown_code(item).strip()
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
DESCRIPTION_ALIAS_CONFIRMATION_THRESHOLD = 2  # times a human must keep a wording difference before it stops being flagged
DESCRIPTION_SIMILARITY_THRESHOLD = 0.75  # sheet vs catalog description, for a row whose code is already pinned down
ALIAS_SIMILARITY_THRESHOLD = 0.85  # a new wording this close to one a human already approved is accepted too

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
    data.setdefault("accepted_descriptions", {})  # code -> {sheet wording -> times a human kept it}
    return data


def save_learned_associations(data: dict) -> None:
    try:
        LEARNED_ASSOCIATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEARNED_ASSOCIATIONS_PATH.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def descriptions_compatible(sheet: str, catalog: str) -> bool:
    """For a row already identified by its UPC: the same product text, allowing for the photo
    cutting off the start of the line (catalog '90 YOG STRWB BAN BLU 12CT' vs sheet 'YOG STRWB
    BAN BLU 12CT') or a catalog prefix the sheet doesn't print. Both inputs are normalized."""
    if not sheet or not catalog:
        return False
    if sheet == catalog:
        return True
    short, long_ = sorted((sheet, catalog), key=len)
    short_tokens, long_tokens = short.split(), long_.split()
    return len(short_tokens) >= 2 and long_.endswith(short) or (
        len(short_tokens) >= 2 and set(short_tokens) <= set(long_tokens)
    )


def description_similarity(sheet: str, catalog: str) -> float:
    """0-1 similarity of two normalized descriptions, tolerant of the photo cutting off the start
    of either line (tries dropping up to 4 leading characters from each side)."""
    if not sheet or not catalog:
        return 0.0
    best = 0.0
    for k in range(5):
        best = max(
            best,
            difflib.SequenceMatcher(None, sheet, catalog[k:]).ratio(),
            difflib.SequenceMatcher(None, sheet[k:], catalog).ratio(),
        )
    return best


def descriptions_close(sheet: str, catalog: str) -> bool:
    return descriptions_compatible(sheet, catalog) or description_similarity(sheet, catalog) >= DESCRIPTION_SIMILARITY_THRESHOLD


def alias_accepted(learned: dict, code: str, description: str) -> bool:
    """True once a human has kept this code with this wording (or one very close to it) enough times."""
    for wording, count in learned.get("accepted_descriptions", {}).get(code, {}).items():
        if count >= DESCRIPTION_ALIAS_CONFIRMATION_THRESHOLD and (
            wording == description or description_similarity(description, wording) >= ALIAS_SIMILARITY_THRESHOLD
        ):
            return True
    return False


def upc_key(value) -> str:
    """Digits only, leading zeros dropped, so '041383090714' and '41383090714' are the same UPC.
    Anything shorter than a real UPC is not treated as one."""
    digits = "".join(ch for ch in str(value if value is not None else "") if ch.isdigit()).lstrip("0")
    return digits if len(digits) >= 9 else ""


def load_item_catalog() -> dict:
    """Builds {"by_code": {code: {description, brand}}, "by_description": {desc: [candidates]}}
    from the uploaded Excel, merged with any confirmed-twice new items learned from review -
    those behave identically to a real catalog entry from then on."""
    by_code: dict[str, dict] = {}
    by_description: dict[str, list] = {}
    by_upc: dict[str, dict] = {}  # printed long codes on some sheets are the catalog's UPCs

    def add_entry(code: str, description: str, brand: str) -> None:
        if not code or not description:
            return
        by_code[code] = {"description": description, "brand": brand}
        by_description.setdefault(description, [])
        if not any(c["item_no"] == code for c in by_description[description]):
            by_description[description].append({"item_no": code, "brand": brand})

    if ITEM_CATALOG_PATH.exists():
        try:
            df = pd.read_excel(ITEM_CATALOG_PATH, dtype={"UPC": str, "Case UPC": str})
            for _, row in df.iterrows():
                code = normalize_catalog_text(row.get("Item Number"))
                description = normalize_catalog_text(row.get("Item Description"))
                add_entry(code, description, normalize_catalog_text(row.get("Brand")))
                for upc_column in ("UPC", "Case UPC"):
                    upc = upc_key(row.get(upc_column))
                    if upc and code and description:
                        by_upc.setdefault(upc, {
                            "item_no": code, "description": description,
                            "upc": str(row.get(upc_column)).strip(),  # as the catalog writes it, leading zero kept
                        })
        except (ValueError, KeyError, OSError):
            pass

    learned = load_learned_associations()
    for code, entry in learned.get("new_items", {}).items():
        # the master file always wins - a learned "new item" must never overwrite a real entry
        if entry.get("confirmed_count", 0) >= NEW_ITEM_CONFIRMATION_THRESHOLD and code not in by_code:
            add_entry(code, normalize_catalog_text(entry.get("description")), normalize_catalog_text(entry.get("brand")))

    return {"by_code": by_code, "by_description": by_description, "by_upc": by_upc}


def resolve_against_catalog(item: dict, catalog: dict, learned: dict) -> str:
    """Checks item_no against the catalog; auto-corrects it in place when a correction is
    unambiguous and high-confidence (see module note above), and always returns a review
    reason string ("" when there's nothing to flag)."""
    by_code = catalog.get("by_code", {})
    by_description = catalog.get("by_description", {})

    code = normalize_catalog_text(item.get("item_no"))
    description = normalize_catalog_text(item.get("description"))
    if upc_key(code):
        # the code identified on the photo is a UPC: it gets its own export column, whether or
        # not the catalog knows it (a catalog match below swaps item_no for the item number)
        item["UPC"] = code
    if not by_code or not code or not description:
        return ""

    def mismatch(for_code: str) -> str:
        item["_catalog_desc_mismatch"] = for_code
        return "item code's description doesn't match the catalog - please verify"

    def description_ok(for_code: str) -> bool:
        return descriptions_close(description, by_code[for_code]["description"]) or alias_accepted(learned, for_code, description)

    # This function runs twice per row (before and after the premium re-check). A row whose code
    # was pinned down on the first pass - by its UPC or by a partial code - must stay pinned down,
    # otherwise the second pass would judge the swapped-in item number by the strict rule below.
    if item.get("_catalog_resolved") == code and code in by_code:
        return "" if description_ok(code) else mismatch(code)

    known = by_code.get(code)
    if known and known["description"] == description:
        return ""  # exact match - nothing to do

    if not known:
        # a long printed code (12-digit) that is a UPC in the catalog identifies exactly one item, so
        # it is swapped for that item's number (the original is logged in the correction log at
        # commit). The code is what identifies the row; the description is only a similarity check.
        upc_hit = catalog.get("by_upc", {}).get(upc_key(code))
        if upc_hit and upc_hit["item_no"] in by_code:
            full = upc_hit["item_no"]
            item.update({"item_no": full, "UPC": upc_hit["upc"], "_catalog_corrected_from": code, "_catalog_resolved": full})
            return "" if description_ok(full) else mismatch(full)

        # a short code whose leading digits the photo cut off (e.g. '8581' for 278581): accept only if
        # exactly one catalog code ends with those digits AND its description is close to the sheet's
        if code.isdigit() and len(code) >= 3 and not upc_key(code):
            partial = [
                c for c in by_code
                if len(c) > len(code) and c.endswith(code) and descriptions_close(description, by_code[c]["description"])
            ]
            if len(partial) == 1:
                full = partial[0]
                item.update({"item_no": full, "_catalog_corrected_from": code, "_catalog_resolved": full})
                return ""
    if not known and upc_key(code):
        # a long code the catalog has never seen: the single-digit-misread correction below is
        # meant for short item numbers, so treat it as a possible new item - kept twice by a human,
        # the code + description are learned as a pair and it stops being flagged
        item["_catalog_new_item"] = code
        return "item code not found in the catalog (may be a new item) - please verify"

    candidates = [c for c in by_description.get(description, []) if c["item_no"] != code]
    if not candidates:
        # the flags record WHICH code they were about, so a later human correction of the code
        # can't be mistaken for a confirmation of the original one
        if not known:
            item["_catalog_new_item"] = code
            return "item code not found in the catalog (may be a new item) - please verify"
        # a real catalog code whose sheet wording matches no other product: usually the catalog
        # and the printed sheet just word the same product differently. Once a human has kept
        # this exact wording enough times, stop asking.
        if alias_accepted(learned, code, description):
            return ""
        return mismatch(code)

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
    seen_alias_codes: set[str] = set()
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
        flagged_new = item.get("_catalog_new_item")
        # counts only if the human kept the code as-is (True = a flag from before flags carried the code)
        if flagged_new and (flagged_new is True or flagged_new == code):
            entry = learned["new_items"].setdefault(code, {
                "description": normalize_catalog_text(item.get("description")),
                "brand": normalize_catalog_text(item.get("brand")),
                "confirmed_count": 0,
            })
            entry["confirmed_count"] = entry.get("confirmed_count", 0) + 1
            changed = True
        # a wording difference the human reviewed and kept (rows they ignored never reach here;
        # a retyped code doesn't count). At most one confirmation per code per commit.
        if item.get("_catalog_desc_mismatch") == code and code not in seen_alias_codes:
            seen_alias_codes.add(code)
            wording = normalize_catalog_text(item.get("description"))
            aliases = learned["accepted_descriptions"].setdefault(code, {})
            # a wording very close to one already recorded counts toward that one, so the small
            # per-photo reading differences ('STRW' vs 'STRWB') add up instead of starting over
            wording = next((w for w in aliases if w == wording or description_similarity(wording, w) >= ALIAS_SIMILARITY_THRESHOLD), wording)
            aliases[wording] = aliases.get(wording, 0) + 1
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
        year, month = int(month_str[:4]), int(month_str[4:6])
        return f"{month_name(month)} {year}"
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


def list_jobs(limit: int = 10, open_only: bool = False) -> list[dict]:
    """Newest first. open_only=True hides committed jobs - those are already in the usage
    report, so the main screen only needs orders still waiting on the user."""
    if not JOBS_DIR.exists():
        return []
    jobs = []
    for d in sorted((p for p in JOBS_DIR.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        meta = read_job_meta(d.name)
        if meta and not (open_only and meta.get("status") == "committed"):
            jobs.append(meta)
        if len(jobs) >= limit:
            break
    return jobs


def dismiss_job(job_id: str) -> None:
    shutil.rmtree(job_dir_for(job_id), ignore_errors=True)


def job_display_status(meta: dict) -> str:
    """'processing' only if a live worker actually exists - a job marked processing on disk
    with no in-memory progress entry was orphaned by a server restart."""
    status = meta.get("status", "failed")
    if status == "processing" and meta["job_id"] not in get_job_runtime()["progress"]:
        return "interrupted"
    return status


def reset_review_widgets() -> None:
    # review widgets are keyed by review_id, which repeats across jobs that share photo
    # filenames - clear leftovers so one job's review state can't leak into the next
    for key in [k for k in st.session_state.keys() if k.startswith(("hw_", "ignore_", "itemno_", "manual_upload"))]:
        del st.session_state[key]
    for key in ("review_overrides", "review_draft", "_draft_saved", "dismiss_pending"):
        st.session_state.pop(key, None)


def clear_loaded_order() -> None:
    reset_review_widgets()
    for key in ("results", "debug_rows", "num_photos", "job_errors", "review_crops", "review_customer",
                "loaded_job_id", "review_committed", "report_logged", "auto_downloaded"):
        st.session_state.pop(key, None)


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
    reset_review_widgets()
    st.session_state["review_draft"] = load_review_draft(job_id)  # edits made before a reload/restart
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

    area.info(t("detected_working", sv=sv_code, n=len(image_files)))
    progress = area.progress(0.0, text=t("starting"))

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
                progress.progress(done / total, text=t("failed_photo", name=path.name, done=done, total=total))
                continue
            all_items.extend(items)
            all_review_crops.update(review_crops)
            all_debug_info.append(photo_debug_info)
            dest = processed_dir / path.name
            if dest.exists():
                dest = processed_dir / f"{path.stem}_{int(time.time())}{path.suffix}"
            shutil.move(str(path), str(dest))
            progress.progress(done / total, text=t("finished_photo", done=done, total=total, name=path.name))

    progress.progress(1.0, text=t("done_text"))

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
        area.warning(t("flagged_warning", sv=sv_code, done=succeeded, n=len(all_items), m=flagged_count))
    elif all_items:
        df = to_export_df(all_items)
        out_path = output_dir / f"{sv_code}_order_{timestamp}.csv"
        df.to_csv(out_path, index=False)
        log_batch_report(
            sv_code, f"{sv_code}_order_{timestamp}", stats["num_photos"],
            stats["no_escalation"], stats["resolved_by_premium"], 0, 0, 0,
        )
        area.success(t("clean_success", sv=sv_code, done=succeeded, n=len(all_items), name=out_path.name))
    else:
        area.success(t("none_found", sv=sv_code, done=succeeded))
    if failures:
        area.error(t("photos_failed", sv=sv_code, n=len(failures), list="; ".join(failures)))


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
        area.caption(t("settling", sv=sv_code, n=len(files)))


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
        return {"status": "yellow", "detail": t("sv_in_progress", n=len(input_files)), "pending_flagged": pending_flagged}
    if pending_flagged:
        return {"status": "red", "detail": t("sv_need_review", n=pending_flagged), "pending_flagged": pending_flagged}
    if paths["output"].exists():
        today = date.today()
        # check actual last-modified date, not the timestamp in the filename - a batch flagged
        # yesterday but reviewed/committed today should count as processed TODAY, not yesterday
        for f in paths["output"].glob(f"{sv_code}_order_*.csv"):
            if date.fromtimestamp(f.stat().st_mtime) == today:
                return {"status": "blue", "detail": t("sv_done_today"), "pending_flagged": 0}
    return {"status": "green", "detail": t("sv_up_to_date"), "pending_flagged": 0}


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
    with area.expander(t("review_sv", sv=sv_code), expanded=True):
        for batch in list_pending_batches(base):
            batch_id = batch["batch_id"]
            batch_items = batch["items"]
            batch_flagged = [item for item in batch_items if item.get("needs_review")]
            st.markdown(t("batch_line", id=batch_id, n=len(batch_items), m=len(batch_flagged)))
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
                ready = st.button(t("commit_output"), key=f"commit_{sv_code}_{batch_id}")

            if ready:
                record_review_outcomes(batch_flagged)
                record_catalog_learning(resolved_items)
                agreed, disagreed = tally_human_agreement(batch_flagged)
                batch_stats = batch.get("stats") or {}
                output_dir.mkdir(parents=True, exist_ok=True)
                if resolved_items:
                    out_df = to_export_df(resolved_items)
                    out_path = output_dir / f"{batch_id}.csv"
                    out_df.to_csv(out_path, index=False)
                    st.success(t("saved_file", name=out_path.name, n=len(resolved_items)))
                else:
                    st.info(t("all_ignored"))
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
        t("parent_folder"),
        key="parent_folder_path",
        placeholder=r"D:\Milagro\A G S\Proyecto AGS",
        help=t("parent_folder_help"),
        on_change=save_parent_folder,
    )
    watching = st.toggle(t("watch_toggle"), key="watch_enabled")

    if not watching:
        st.caption(t("not_watching"))

    parent_path = st.session_state.get("parent_folder_path", "").strip()
    if not parent_path:
        st.warning(t("enter_parent"))
        return

    parent = Path(parent_path)
    if not parent.exists():
        st.error(t("folder_missing", path=parent_path))
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
        st.error(t("banner_flagged", n=total_flagged))


# ============================== page layout ==============================

if not CLOUD_MODE:
    if "parent_folder_path" not in st.session_state:
        st.session_state["parent_folder_path"] = load_saved_parent_folder()
    exception_banner()

if CLOUD_MODE:
    tab_upload, tab_reports, tab_settings = st.tabs([t("tab_upload"), t("tab_reports"), t("tab_settings")])
    tab_batches = None
else:
    tab_upload, tab_batches, tab_reports, tab_settings = st.tabs(
        [t("tab_upload"), t("tab_batches"), t("tab_reports"), t("tab_settings")]
    )


def job_card_html(kind: str, customer: str, n_photos: int, when: str, done: int = 0, total: int = 0) -> str:
    title = f"{html_escape(display_customer(customer))} · {t('job_photos', n=n_photos)} · {when}"
    if kind == "processing":
        pct = int(100 * done / max(total, 1))
        return (
            f'<div class="job-card processing"><div class="job-status">⏳ {t("status_processing")}</div>'
            f'<div class="job-title">{title}</div>'
            f'<div class="job-bar"><div class="job-bar-fill" style="width:{pct}%"></div></div>'
            f'<div class="job-sub">{t("job_progress", done=done, total=total)}</div></div>'
        )
    return (
        f'<div class="job-card ready"><div class="job-status">✅ {t("status_ready")}</div>'
        f'<div class="job-title">{title}</div></div>'
    )


def render_dismiss_control(job_id: str, scope: str) -> None:
    """Small red Dismiss button with a one-step confirmation - it deletes the order's photos and
    saved results for good, reviewed or not, so an accidental click must not be enough."""
    if st.session_state.get("dismiss_pending") != job_id:
        if st.button(t("dismiss"), key=f"dismiss_{scope}_{job_id}"):
            st.session_state["dismiss_pending"] = job_id
            st.rerun()
        return
    st.warning(t("dismiss_confirm"))
    if st.button(t("dismiss_yes"), key=f"dismiss_yes_{scope}_{job_id}"):
        dismiss_job(job_id)
        if st.session_state.get("loaded_job_id") == job_id:
            clear_loaded_order()
        st.session_state.pop("dismiss_pending", None)
        st.rerun()
    if st.button(t("dismiss_no"), key=f"keep_{scope}_{job_id}"):
        st.session_state.pop("dismiss_pending", None)
        st.rerun()


def render_jobs_panel() -> None:
    """Only orders still waiting on the user: processing (dark orange), ready to review (dark
    green), or needing attention. Committed orders are already in the usage report, and the
    order currently open for review is already on screen below, so neither is listed here."""
    loaded = st.session_state.get("loaded_job_id")
    jobs = [m for m in list_jobs(open_only=True) if m["job_id"] != loaded]
    if not jobs:
        return
    st.subheader(t("jobs_title"))
    st.caption(t("jobs_caption"))
    progress_registry = get_job_runtime()["progress"]
    for meta in jobs:
        job_id = meta["job_id"]
        status = job_display_status(meta)
        total = len(meta.get("photos", [])) or meta.get("total", 0)
        when = format_job_time(meta.get("created", time.time()))
        customer = meta.get("customer", "?")
        if status == "processing":
            prog = progress_registry.get(job_id, {"done": 0, "total": total})
            st.markdown(
                job_card_html("processing", customer, total, when, prog["done"], prog["total"]),
                unsafe_allow_html=True,
            )
        elif status == "done":
            card_col, button_col = st.columns([3.2, 1.4], vertical_alignment="center")
            with card_col:
                st.markdown(job_card_html("ready", customer, total, when), unsafe_allow_html=True)
            with button_col:
                if st.button(t("open_review"), key=f"open_{job_id}", use_container_width=True):
                    load_job_into_session(job_id)
                    st.rerun()
                render_dismiss_control(job_id, "card")
        else:  # interrupted or failed - needs the user's attention
            with st.container(border=True):
                st.markdown(f"**{display_customer(customer)}** · {t('job_photos', n=total)} · {when}")
                st.caption(
                    t("job_interrupted") if status == "interrupted"
                    else t("job_failed", error=meta.get("error", "unknown error"))
                )
                restart_col, dismiss_col = st.columns(2)
                with restart_col:
                    if (job_dir_for(job_id) / "photos").exists() and st.button(t("restart"), key=f"restart_{job_id}"):
                        meta["status"] = "processing"
                        meta.pop("error", None)
                        write_job_meta(job_id, meta)
                        launch_job_thread(job_id)
                        st.rerun()
                with dismiss_col:
                    if st.button(t("dismiss"), key=f"dismiss_{job_id}"):
                        dismiss_job(job_id)
                        st.rerun()


with tab_upload:
    nonce = st.session_state.get("uploader_nonce", 0)
    if st.session_state.get("job_started_notice"):
        st.success(st.session_state.pop("job_started_notice"))
    uploaded_files = st.file_uploader(
        t("drop_photos"),
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{nonce}",
    )
    upload_customer = st.selectbox(
        t("customer"),
        SV_CODES + ["Other Customer"],
        key=f"upload_customer_{nonce}",
        index=None,
        placeholder=t("select_customer"),
        format_func=display_customer,
        help=t("customer_help_cloud") if CLOUD_MODE else t("customer_help_local"),
    )

    if uploaded_files and not upload_customer:
        st.warning(t("select_customer_warning"))

    ready_to_go = bool(uploaded_files and upload_customer)
    go_clicked = st.button(t("go_process"), key="go_process", disabled=not ready_to_go, use_container_width=True)
    if not ready_to_go:
        st.caption(t("go_hint"))

    if go_clicked and ready_to_go:
        if st.session_state.get("report_logged"):
            clear_loaded_order()  # the previous order is already committed - don't leave it on screen
        start_extraction_job([(f.name, f.getvalue()) for f in uploaded_files], upload_customer)
        st.session_state["uploader_nonce"] = nonce + 1
        st.session_state["job_started_notice"] = t(
            "job_started", n=len(uploaded_files), customer=display_customer(upload_customer)
        )
        st.rerun()

    st.fragment(
        run_every=3 if any(job_display_status(m) == "processing" for m in list_jobs(open_only=True)) else None
    )(render_jobs_panel)()

    if "debug_rows" in st.session_state and st.session_state["debug_rows"]:
        with st.expander(t("photo_counts")):
            st.dataframe(translate_columns(pd.DataFrame(st.session_state["debug_rows"])), use_container_width=True)

    if st.session_state.get("job_errors"):
        st.warning(t("photo_problems") + "\n\n" + "\n".join(st.session_state["job_errors"]))

    if "results" in st.session_state:
        items = st.session_state["results"]
        st.subheader(t("rows_found", n=len(items)))
        st.caption(t("customer_label", name=display_customer(st.session_state.get("review_customer"))))

        flagged = [item for item in items if item.get("needs_review")]
        committed = st.session_state.get("review_committed", False) or not flagged

        if flagged and not committed:
            st.warning(t("review_needed", n=len(flagged)))
            st.caption(t("review_hint"))
            if st.session_state.get("loaded_job_id"):
                st.caption(t("draft_restored"))
            review_crops = st.session_state.get("review_crops", {})
            for item in flagged:
                review_id = item.get("review_id")
                images = review_crops.get(review_id) or {}
                render_review_row(item, review_id, images.get("small"), images.get("full"))

            loaded_job = st.session_state.get("loaded_job_id")
            save_review_draft(loaded_job, collect_review_draft(flagged))
            if st.button(t("commit_review")):
                st.session_state["review_overrides"] = collect_review_draft(flagged)
                record_review_outcomes(flagged)
                st.session_state["review_committed"] = True
                committed = True
            if loaded_job and not committed:
                render_dismiss_control(loaded_job, "review")

        if items and committed:
            resolved_items = apply_review_overrides(items)
            # a review-time item_no correction can create a NEW duplicate that didn't exist
            # at extraction time - re-check before export, not just once up front
            resolved_items = resolve_duplicate_item_codes(resolved_items)
            high_items = find_high_value_items(resolved_items)

            show_results = True
            if high_items:
                show_results = render_qty_confirmation_gate(high_items, key_prefix="manual_upload", persist=True)

            if not show_results:
                st.info(t("confirm_qty_first"))
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

                df = to_export_df(resolved_items)
                edited_df = st.data_editor(
                    df, num_rows="dynamic", use_container_width=True,
                    column_config={c: st.column_config.Column(column_label(c)) for c in df.columns},
                )

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
                    st.caption(t("also_saved", path=saved_path))

                if not st.session_state.get("auto_downloaded"):
                    trigger_browser_download([
                        (excel_buffer.getvalue(), f"{base_filename}.xlsx", XLSX_MIME),
                        (upload_buffer.getvalue(), upload_filename, XLSX_MIME),
                    ])
                    st.session_state["auto_downloaded"] = True

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(t("dl_csv"), csv_bytes, f"{base_filename}.csv", "text/csv")
                with col2:
                    st.download_button(t("dl_excel_again"), excel_buffer.getvalue(), f"{base_filename}.xlsx", XLSX_MIME)
                with col3:
                    st.download_button(t("dl_upload_again"), upload_buffer.getvalue(), upload_filename, XLSX_MIME)

                st.success(t("order_finished", a=f"{base_filename}.xlsx", b=upload_filename))
                if st.button(t("close_order"), key="close_order"):
                    clear_loaded_order()
                    st.rerun()
        elif not items:
            st.info(t("no_rows"))
            if st.button(t("close_order"), key="close_empty_order"):
                finish_job(st.session_state.get("loaded_job_id"))  # nothing to commit, so retire it here
                clear_loaded_order()
                st.rerun()

if tab_batches is not None:
    with tab_batches:
        st.subheader(t("batches_title"))
        st.caption(t("batches_caption"))
        customer_batches_section()

with tab_reports:
    st.subheader(t("reports_title"))
    st.caption(t("reports_caption"))
    available_months = list_available_reports()
    if not available_months:
        st.info(t("no_reports"))
    else:
        selected_month = st.selectbox(t("month"), available_months, format_func=format_month_label)
        report_file = report_path_for(selected_month)
        report_df = pd.read_csv(report_file)
        st.dataframe(translate_columns(report_df), use_container_width=True)
        st.download_button(
            t("dl_report", month=format_month_label(selected_month)),
            report_file.read_bytes(),
            report_file.name,
            "text/csv",
        )

with tab_settings:
    st.subheader(t("settings_title"))
    st.caption(t("settings_caption"))

    current_key = os.getenv("OPENAI_API_KEY", "")
    masked = f"{'•' * max(len(current_key) - 4, 4)}{current_key[-4:]}" if current_key else t("none_set")
    st.text_input(t("current_key"), value=masked, disabled=True)

    new_key = st.text_input(t("new_key"), type="password", placeholder="sk-...")
    col_save, col_test = st.columns(2)
    with col_save:
        if st.button(t("save_apply")):
            if new_key.strip():
                apply_new_api_key(new_key)
                st.success(t("key_saved"))
                st.rerun()
            else:
                st.warning(t("paste_key_first"))
    with col_test:
        if st.button(t("test_key")):
            try:
                client.chat.completions.create(
                    model=CHEAP_MODEL,
                    max_completion_tokens=5,
                    messages=[{"role": "user", "content": "Reply with: OK"}],
                )
                st.success(t("key_works"))
            except Exception as e:
                st.error(t("test_failed", e=e))

    st.divider()
    st.subheader(t("catalog_title"))
    st.caption(t("catalog_caption"))

    if ITEM_CATALOG_PATH.exists():
        try:
            row_count = len(pd.read_excel(ITEM_CATALOG_PATH))
            updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(ITEM_CATALOG_PATH.stat().st_mtime))
            st.success(t("catalog_loaded", n=row_count, when=updated))
        except (ValueError, KeyError, OSError) as e:
            st.error(t("catalog_unreadable", e=e))
    else:
        st.info(t("no_catalog"))

    catalog_upload = st.file_uploader(t("upload_catalog"), type=["xlsx"], key="catalog_upload")
    if catalog_upload and st.button(t("save_catalog")):
        ITEM_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ITEM_CATALOG_PATH.write_bytes(catalog_upload.getvalue())
        st.success(t("catalog_saved"))
        st.rerun()

    learned = load_learned_associations()
    pending_new = sum(1 for e in learned["new_items"].values() if e.get("confirmed_count", 0) < NEW_ITEM_CONFIRMATION_THRESHOLD)
    confirmed_new = len(learned["new_items"]) - pending_new
    accepted_wordings = sum(
        1 for wordings in learned["accepted_descriptions"].values()
        for count in wordings.values() if count >= DESCRIPTION_ALIAS_CONFIRMATION_THRESHOLD
    )
    st.caption(t(
        "learned_caption", confirmed=confirmed_new, pending=pending_new,
        thr=NEW_ITEM_CONFIRMATION_THRESHOLD, aliases=accepted_wordings,
        corrections=len(learned["correction_log"]),
    ))
    if learned["correction_log"]:
        with st.expander(t("correction_history")):
            st.dataframe(translate_columns(pd.DataFrame(learned["correction_log"][::-1])), use_container_width=True)
