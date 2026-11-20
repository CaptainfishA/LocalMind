from __future__ import annotations

import csv
import io
import os
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .dataset import finish_dataset, new_directory


def _render(path: Path, title: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".txt", ".md"}:
        path.write_text(title + "\n" + text, encoding="utf-8")
    elif path.suffix == ".csv":
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["主题", "内容"])
            writer.writerow([title, text])
    elif path.suffix == ".docx":
        from docx import Document

        doc = Document()
        doc.add_heading(title, 0)
        doc.add_paragraph(text)
        doc.save(path)
    elif path.suffix == ".xlsx":
        from openpyxl import Workbook

        book = Workbook()
        sheet = book.active
        sheet.title = "项目资料"
        sheet.append(["主题", "内容"])
        sheet.append([title, text])
        sheet.column_dimensions["A"].width = 32
        sheet.column_dimensions["B"].width = 100
        book.save(path)
        book.close()
    elif path.suffix == ".pptx":
        from pptx import Presentation

        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = text
        deck.save(path)
    elif path.suffix == ".pdf":
        import fitz

        with fitz.open() as doc:
            page = doc.new_page()
            remaining = page.insert_textbox(fitz.Rect(48, 48, 547, 794), title + "\n\n" + text,
                                            fontname="china-s", fontsize=13)
            if remaining < 0:
                raise ValueError("Fixture PDF text overflow")
            doc.save(path, no_new_id=True)
    else:
        raise ValueError(path.suffix)
    if path.suffix in {".docx", ".xlsx", ".pptx"}:
        # Office ZIP metadata otherwise changes hashes on every regeneration.
        buffer = io.BytesIO()
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as dest:
            for name in sorted(source.namelist()):
                content = source.read(name)
                if name == "docProps/core.xml":
                    ET.register_namespace("cp", "http://schemas.openxmlformats.org/package/2006/metadata/core-properties")
                    ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
                    ET.register_namespace("dcterms", "http://purl.org/dc/terms/")
                    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
                    props = ET.fromstring(content)
                    for tag in ("created", "modified"):
                        element = props.find("{http://purl.org/dc/terms/}" + tag)
                        if element is not None:
                            element.text = "2025-02-01T00:00:00Z"
                    content = ET.tostring(props, encoding="utf-8", xml_declaration=True)
                info = zipfile.ZipInfo(name, date_time=(2025, 2, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                dest.writestr(info, content)
        path.write_bytes(buffer.getvalue())


def build_fixtures(output: str | Path, *, seed: int = 20261013, projects: int = 12) -> Path:
    if not 2 <= projects <= 1000:
        raise ValueError("projects must be between 2 and 1000")
    root = new_directory(output)
    rng = random.Random(seed)
    corpus, queries, qrels = [], [], []
    names = ["星河", "远航", "青禾", "云帆", "晨光", "启明", "北辰", "清源", "长风", "新桥", "明川", "松月"]
    formats = [".txt", ".md", ".csv", ".docx", ".xlsx", ".pptx", ".pdf"]
    dev_projects = set(rng.sample(range(projects), max(1, projects // 3)))
    for i in range(projects):
        project = names[i % len(names)] + (f"{i // len(names) + 1}期" if i >= len(names) else "")
        region = ["华东", "华南", "华北"][i % 3]
        days, amount = 10 + i, 100 + i * 7
        records = [
            ("采购合同", f"{project}项目采购合同，合同编号 HT-{i+1:04d}。验收后{days}天内付款，提前终止需支付三个月服务费。"),
            ("预算明细", f"{project}项目{region}区预算为{amount}万元，比上期增加23%。机房设备占总预算的40%。"),
            ("需求评审", f"{project}项目需求评审纪要。决定支持断网时查询已有资料，恢复联网后再同步变更。负责人为李工。"),
            ("旧版合同", f"{project}项目旧版合同，已作废。验收后90天付款，提前终止无需支付服务费。"),
            ("测试报告", f"{project}项目测试报告。高温测试持续72小时，异常恢复平均耗时8秒，数据无丢失。"),
            ("培训安排", f"{project}项目培训安排。周二上午讲解采购流程，周四下午介绍机房设备。"),
            ("运维手册", f"{project}项目运维手册。日志保留180天；出现磁盘告警时先清理临时缓存，禁止删除业务原件。"),
        ]
        for j, (title, text) in enumerate(records):
            ext = formats[(i + j) % len(formats)]
            relative = Path("files") / region / project / (title + ext)
            _render(root / relative, f"{project}{title}", text)
            timestamp = 1738368000 + i * 86400
            os.utime(root / relative, (timestamp, timestamp))
            corpus.append({"file_id": f"p{i}-d{j}", "path": relative.as_posix(), "group": project})
        samples = [
            (f"HT-{i+1:04d}", 0, "identifier", {}),
            (f"{project}项目提前结束合作需要赔偿几个月费用", 0, "semantic", {}),
            (f"{project}项目预算金额", 1, "table", {}),
            (f"{project}项目讨论没有网络也能查资料的会议", 2, "paraphrase", {}),
            (f"{project}项目测试报告", 4, "file_type", {"file_type": formats[(i+4) % 7].lstrip(".")}),
            (f"{project}项目日志保留时间", 6, "path", {"path_contains": f"/{region}/{project}/"}),
        ]
        for k, (text, relevant, intent, filters) in enumerate(samples):
            qid = f"p{i}-q{k}"
            if "path_contains" in filters:
                filters["path_contains"] = filters["path_contains"].replace("/", os.sep)
            queries.append({"query_id": qid, "text": text,
                            "split": "dev" if i in dev_projects else "test", "intent": [intent], "filters": filters})
            for doc in corpus:
                if doc["group"] == project:
                    qrels.append({"query_id": qid, "file_id": doc["file_id"],
                                  "relevance": 3 if doc["file_id"] == f"p{i}-d{relevant}" else 0})
    # Template semantics make every other project explicitly irrelevant.
    judged = {(r["query_id"], r["file_id"]) for r in qrels}
    qrels.extend({"query_id": q["query_id"], "file_id": d["file_id"], "relevance": 0}
                 for q in queries for d in corpus if (q["query_id"], d["file_id"]) not in judged)
    finish_dataset(root, corpus, queries, qrels, {
        "name": "localmind-office-regression-v1", "kind": "synthetic_regression", "seed": seed,
        "formats": formats, "split_policy": "Whole projects kept in one split",
        "comparability": "Template-generated functional regression; not a real-world accuracy estimate. No OCR images.",
    })
    return root


def build_answerability_fixtures(output: str | Path) -> Path:
    root = new_directory(output)
    records = [
        ("dev", "杭州运维组备用电源维护记录，编号 DY-482。每季度进行电池放电测试，停电后由备用电源维持机房运行。",
         ("DY-482", "杭州的同事怎样检查停电后接替供电的设备")),
        ("dev", "成都采购组供应商发票核对规程，编号 FP-735。发票含税金额需要和采购订单一致，差额超过五十元退回确认。",
         ("FP-735", "成都采购同事怎样核对开票金额与下单金额是否一致")),
        ("test", "青岛人事组休假办理规定，编号 XJ-126。休假申请至少提前三天提交，由部门负责人批准后登记剩余假期。",
         ("XJ-126", "青岛员工准备休息几天应提前向谁提交申请")),
        ("test", "武汉行政组访客登记办法，编号 FK-864。外来人员进入办公楼前登记姓名、单位和被访员工，离开时归还临时通行证。",
         ("FK-864", "武汉公司的外来人员进楼前要留下哪些资料")),
    ]
    corpus, queries, qrels = [], [], []
    for index, (split, body, descriptions) in enumerate(records):
        fid = f"d{index}"
        relative = Path("files") / f"document-{index}.txt"
        _render(root / relative, "模拟资料", body)
        os.utime(root / relative, (1738368000, 1738368000))
        corpus.append({"file_id": fid, "path": relative.as_posix()})
        for number, text in enumerate(descriptions):
            qid = f"{fid}-q{number}"
            queries.append({"query_id": qid, "text": text, "split": split,
                            "intent": ["identifier" if number == 0 else "paraphrase"]})
            qrels.append({"query_id": qid, "file_id": fid, "relevance": 3})
    for split, descriptions in (
        ("dev", (("维护记录 DY-999", "missing_identifier"),
                 ("成都采购组的备用电源维护记录", "missing_combination"),
                 ("火星探测器着陆降落伞怎样展开", "unrelated_topic"))),
        ("test", (("访客登记 FK-000", "missing_identifier"),
                  ("武汉行政组批准员工休假的办理规定", "missing_combination"),
                  ("近海潮汐发电设备的效率怎样计算", "unrelated_topic"))),
    ):
        for number, (text, intent) in enumerate(descriptions):
            queries.append({"query_id": f"{split}-none-{number}", "text": text, "split": split,
                            "answerability": "no_answer", "intent": [intent]})
    positive = {(row["query_id"], row["file_id"]) for row in qrels}
    qrels.extend({"query_id": q["query_id"], "file_id": d["file_id"], "relevance": 0}
                 for q in queries for d in corpus if (q["query_id"], d["file_id"]) not in positive)
    finish_dataset(root, corpus, queries, qrels, {
        "name": "localmind-answerability-regression-v1", "kind": "synthetic_regression",
        "split_policy": "Document topics grouped; no-answer cases refer only to their split's topics or absent topics",
        "label_policy": "Closed-world synthetic facts; every query/document pair explicitly judged. Partial combinations are not correct targets.",
        "comparability": "Generated file-body regression, not real account accuracy or a public benchmark. No chat adapter or OCR.",
    })
    return root
