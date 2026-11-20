import openpyxl
from openpyxl import Workbook
from openpyxl.worksheet._read_only import ReadOnlyWorksheet
import pytest

from localmind.parsers import DocumentParser
from localmind.preview_worker import preview_file


@pytest.fixture
def retained_readers(monkeypatch):
    books, streams, iterators = [], [], []
    load, source, iterate = openpyxl.load_workbook, ReadOnlyWorksheet._get_source, ReadOnlyWorksheet.iter_rows
    def track_book(*args, **kwargs):
        book = load(*args, **kwargs)
        books.append(book)
        return book
    def track_source(self):
        stream = source(self)
        streams.append(stream)
        return stream
    def track_iterator(self, *args, **kwargs):
        iterator = iterate(self, *args, **kwargs)
        iterators.append(iterator)
        return iterator
    monkeypatch.setattr(openpyxl, "load_workbook", track_book)
    monkeypatch.setattr(ReadOnlyWorksheet, "_get_source", track_source)
    monkeypatch.setattr(ReadOnlyWorksheet, "iter_rows", track_iterator)
    yield books, streams, iterators
    # Retain references so cleanup cannot accidentally pass through garbage collection.
    for iterator in iterators:
        iterator.close()
    for book in books:
        book.close()


def sample(path, rows):
    book = Workbook()
    book.active.title = "合成表"
    for i in range(1, rows + 1):
        book.active.append([f"记录{i:05d}", "合成业务说明和交接资料。"])
    book.save(path)
    book.close()
    return path


def assert_closed(readers, path):
    books, streams, iterators = readers
    assert books and all(book._archive.fp is None for book in books)
    assert streams and all(stream.closed for stream in streams)
    assert iterators and all(iterator.gi_frame is None for iterator in iterators)
    renamed = path.with_name("released.xlsx")
    path.rename(renamed)
    renamed.rename(path)


@pytest.mark.parametrize("count", [1, 2000, 2001, 5000])
def test_index_parser_closes_readers_on_completion_and_row_limit(tmp_path, retained_readers, count):
    path = sample(tmp_path / "source.xlsx", count)
    before = path.read_bytes()
    chunks, status, ocr = DocumentParser().parse(path)
    text = "\n".join(chunk.chunk_text for chunk in chunks)
    assert status == "parsed" and ocr == "not_required"
    assert f"记录{min(count, 2000):05d}" in text
    assert "记录02001" not in text
    assert ("已截断" in text) == (count > 2000)
    assert all(chunk.sheet_name == "合成表" for chunk in chunks)
    assert_closed(retained_readers, path)
    assert before == path.read_bytes()


def test_index_parser_closes_workbook_when_chunk_creation_fails(tmp_path, retained_readers, monkeypatch):
    path = sample(tmp_path / "source.xlsx", 2200)
    def fail(*args, **kwargs):
        raise ValueError("synthetic chunk failure")
    monkeypatch.setattr("localmind.parsers.chunk_text_block", fail)
    with pytest.raises(ValueError, match="synthetic chunk failure"):
        DocumentParser().parse(path)
    assert_closed(retained_readers, path)


@pytest.mark.parametrize("anchor", ["", "记录00150", "不存在的片段"])
def test_preview_closes_streams_after_bounded_reads(tmp_path, retained_readers, anchor):
    path = sample(tmp_path / "source.xlsx", 5000)
    before = path.read_bytes()
    result = preview_file(str(path), anchor=anchor)
    assert result["kind"] == "table" and len(result["rows"]) <= 60
    assert_closed(retained_readers, path)
    assert before == path.read_bytes()


def test_preview_closes_readers_when_rendering_fails(tmp_path, retained_readers, monkeypatch):
    path = sample(tmp_path / "source.xlsx", 5000)
    def fail(*args, **kwargs):
        raise ValueError("synthetic render failure")
    monkeypatch.setattr("localmind.preview_worker.table_window", fail)
    with pytest.raises(ValueError, match="synthetic render failure"):
        preview_file(str(path), anchor="记录00150")
    assert_closed(retained_readers, path)


@pytest.mark.parametrize("preview", [False, True])
def test_owned_input_stream_closes_even_when_workbook_loading_fails(tmp_path, monkeypatch, preview):
    path = sample(tmp_path / "source.xlsx", 2)
    streams = []
    def fail(source, **kwargs):
        streams.append(source)
        assert source.read(2) == b"PK"
        raise ValueError("synthetic workbook load failure")
    monkeypatch.setattr(openpyxl, "load_workbook", fail)
    with pytest.raises(ValueError, match="synthetic workbook load failure"):
        preview_file(str(path)) if preview else DocumentParser().parse(path)
    assert streams and all(stream.closed for stream in streams)
    path.rename(tmp_path / "released.xlsx")
