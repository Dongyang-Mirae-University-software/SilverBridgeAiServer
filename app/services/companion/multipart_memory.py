from __future__ import annotations

from dataclasses import dataclass

from python_multipart.multipart import MultipartParser, parse_options_header


@dataclass
class UploadedPart:
    filename: str
    content_type: str
    data: bytes


def read_file_part(content_type_header: str, body: bytes, field_name: str) -> UploadedPart | None:
    """field_name 파일 하나를 꺼낸다. 없으면 None."""
    parts = read_file_parts(content_type_header, body, field_name)
    return parts[0] if parts else None


def read_file_parts(content_type_header: str, body: bytes, field_name: str) -> list[UploadedPart]:
    """multipart 본문에서 field_name 파일을 모두 메모리로만 꺼낸다 (디스크에 쓰지 않는다, medication_ocr 방식).

    multipart 가 아니거나 경계가 없으면 ValueError.
    """
    ctype, params = parse_options_header(content_type_header)
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise ValueError("multipart/form-data 가 아닙니다")

    state = {"field": b"", "value": b"", "filename": b"", "ctype": b"", "in_target": False}
    headers: dict[bytes, bytes] = {}
    out = bytearray()
    found: list[UploadedPart] = []

    def on_part_begin() -> None:
        headers.clear()
        out.clear()
        state.update(filename=b"", ctype=b"", in_target=False)

    def on_header_field(data: bytes, start: int, end: int) -> None:
        state["field"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        state["value"] += data[start:end]

    def on_header_end() -> None:
        headers[state["field"].lower()] = state["value"]
        state["field"], state["value"] = b"", b""

    def on_headers_finished() -> None:
        _, dparams = parse_options_header(headers.get(b"content-disposition", b""))
        name = dparams.get(b"name", b"").decode("utf-8", "replace")
        state["filename"] = dparams.get(b"filename", b"")
        state["ctype"] = headers.get(b"content-type", b"")
        state["in_target"] = name == field_name

    def on_part_data(data: bytes, start: int, end: int) -> None:
        if state["in_target"]:
            out.extend(data[start:end])

    def on_part_end() -> None:
        if state["in_target"]:
            found.append(UploadedPart(
                filename=state["filename"].decode("utf-8", "replace"),
                content_type=state["ctype"].decode("latin-1"),
                data=bytes(out),
            ))

    parser = MultipartParser(boundary, {
        "on_part_begin": on_part_begin,
        "on_header_field": on_header_field,
        "on_header_value": on_header_value,
        "on_header_end": on_header_end,
        "on_headers_finished": on_headers_finished,
        "on_part_data": on_part_data,
        "on_part_end": on_part_end,
    })
    parser.write(body)
    parser.finalize()
    return found
