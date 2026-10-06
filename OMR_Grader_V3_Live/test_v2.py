from __future__ import annotations

import cv2
import numpy as np

from omr import answers_from_detections, build_page_layout, grade_answers, scan_sheet
from sheet_template import create_sheet


def run_case(total: int) -> None:
    expected = {q: ("A", "B", "C", "D")[(q - 1) % 4] for q in range(1, total + 1)}
    detected_all = {}
    pages = 1 if total <= 50 else 2
    for page in range(1, pages + 1):
        sheet = create_sheet(total, page, expected)
        layout = build_page_layout(total, page)
        scan = scan_sheet(sheet, layout=layout, answer_key_mode=True)
        detected = answers_from_detections(scan.detections)
        detected_all.update(detected)
        assert all(detected[q] == expected[q] for q in range(layout.first_question, layout.last_question + 1))
    key = [expected[q] for q in range(1, total + 1)]
    grade = grade_answers(detected_all, key)
    assert grade["score"] == total


def perspective_case() -> None:
    total = 50
    expected = {q: ("A", "B", "C", "D")[(q - 1) % 4] for q in range(1, total + 1)}
    sheet = create_sheet(total, 1, expected)
    h, w = sheet.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[85, 55], [w - 120, 20], [w - 45, h - 80], [35, h - 25]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(sheet, matrix, (w, h), borderValue=(235, 235, 235))
    scan = scan_sheet(photo, layout=build_page_layout(total, 1), answer_key_mode=True)
    detected = answers_from_detections(scan.detections)
    correct = sum(detected[q] == expected[q] for q in range(1, total + 1))
    assert correct >= 48, correct


if __name__ == "__main__":
    for count in (10, 20, 25, 40, 50, 51, 75, 100):
        run_case(count)
        print("passed", count)
    perspective_case()
    print("passed perspective")
