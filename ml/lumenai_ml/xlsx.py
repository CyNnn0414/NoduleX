from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


XML_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_index(cell_ref: str) -> int:
    letters = "".join(character for character in cell_ref if character.isalpha()).upper()
    total = 0
    for character in letters:
        total = (total * 26) + (ord(character) - ord("A") + 1)
    return max(total - 1, 0)


def _read_shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for string_item in root.findall("main:si", XML_NS):
        parts = [node.text or "" for node in string_item.findall(".//main:t", XML_NS)]
        values.append("".join(parts))
    return values


def read_first_sheet_rows(path: Path) -> list[list[object | None]]:
    with ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook_root.find("main:sheets/main:sheet", XML_NS)
        if first_sheet is None:
            return []
        sheet_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "rId1")

        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels_root:
            if rel.attrib.get("Id") == sheet_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            return []
        sheet_path = f"xl/{target.lstrip('/')}"
        sheet_root = ET.fromstring(archive.read(sheet_path))

    rows: list[list[object | None]] = []
    for row in sheet_root.findall(".//main:sheetData/main:row", XML_NS):
        values: dict[int, object | None] = {}
        max_index = -1
        for cell in row.findall("main:c", XML_NS):
            ref = cell.attrib.get("r", "A1")
            index = _column_index(ref)
            max_index = max(max_index, index)
            cell_type = cell.attrib.get("t")
            value_node = cell.find("main:v", XML_NS)
            text_node = cell.find("main:is/main:t", XML_NS)
            if cell_type == "s" and value_node is not None and value_node.text is not None:
                values[index] = shared_strings[int(value_node.text)]
            elif cell_type == "inlineStr" and text_node is not None:
                values[index] = text_node.text
            elif value_node is not None and value_node.text is not None:
                raw = value_node.text
                if "." in raw:
                    try:
                        values[index] = float(raw)
                    except ValueError:
                        values[index] = raw
                else:
                    try:
                        values[index] = int(raw)
                    except ValueError:
                        values[index] = raw
            else:
                values[index] = None
        rows.append([values.get(index) for index in range(max_index + 1)])
    return rows
