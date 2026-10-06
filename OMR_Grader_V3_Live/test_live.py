from __future__ import annotations

import numpy as np

from live_scanner import LiveGradeController
from sheet_template import create_sheet


def feed_until_not_waiting(controller, frame, limit=12):
    for _ in range(limit):
        controller.process_frame(frame)
        snap = controller.snapshot()
        if snap.phase != "WAITING":
            return snap
    return controller.snapshot()


def remove_paper(controller, shape, n=6):
    blank = np.full(shape, 245, dtype=np.uint8)
    for _ in range(n):
        controller.process_frame(blank)
    return controller.snapshot()


def single_page_case():
    total = 20
    key = [("A", "B", "C", "D")[(q - 1) % 4] for q in range(1, total + 1)]
    marks = {q: key[q - 1] for q in range(1, total + 1)}
    sheet = create_sheet(total, 1, marks)
    c = LiveGradeController(total, key, stable_needed=3, absent_needed=3, process_every_n_frames=1)
    snap = feed_until_not_waiting(c, sheet)
    assert snap.phase == "WAIT_REMOVE", snap
    assert snap.latest_grade is not None
    assert snap.latest_grade["score"] == total
    # Same paper must not generate a second result.
    rid = snap.latest_record_id
    for _ in range(5):
        c.process_frame(sheet)
    assert c.snapshot().latest_record_id == rid
    snap = remove_paper(c, sheet.shape, n=4)
    assert snap.phase == "WAITING"
    assert snap.student_number == 2
    print("passed live single-page")


def two_page_case():
    total = 100
    key = [("A", "B", "C", "D")[(q - 1) % 4] for q in range(1, total + 1)]
    marks = {q: key[q - 1] for q in range(1, total + 1)}
    p1 = create_sheet(total, 1, marks)
    p2 = create_sheet(total, 2, marks)
    c = LiveGradeController(total, key, stable_needed=3, absent_needed=3, process_every_n_frames=1)
    snap = feed_until_not_waiting(c, p1)
    assert snap.phase == "WAIT_REMOVE"
    assert snap.expected_page == 1
    snap = remove_paper(c, p1.shape, n=4)
    assert snap.phase == "WAITING"
    assert snap.expected_page == 2
    snap = feed_until_not_waiting(c, p2)
    assert snap.phase == "WAIT_REMOVE", snap
    assert snap.latest_grade is not None
    assert snap.latest_grade["score"] == total
    assert len(c.history()) == 1
    print("passed live two-page")


if __name__ == "__main__":
    single_page_case()
    two_page_case()


def review_case():
    import cv2
    from omr import build_page_layout

    total = 20
    key = [("A", "B", "C", "D")[(q - 1) % 4] for q in range(1, total + 1)]
    marks = {q: key[q - 1] for q in range(1, total + 1)}
    sheet = create_sheet(total, 1, marks)
    layout = build_page_layout(total, 1)
    q5 = next(q for q in layout.questions if q.question == 5)
    # Q5 correct answer is A; deliberately fill B too to simulate a double mark.
    extra = next(b for b in q5.bubbles if b.choice == "B")
    cv2.circle(sheet, extra.center, extra.radius - 6, (0, 0, 0), -1)
    c = LiveGradeController(total, key, stable_needed=3, absent_needed=3, process_every_n_frames=1)
    snap = feed_until_not_waiting(c, sheet)
    assert snap.phase == "REVIEW", snap
    assert 5 in snap.unresolved
    grade = c.resolve_review({5: "A"})
    assert grade["score"] == total
    assert c.snapshot().phase == "WAIT_REMOVE"
    print("passed live unclear-review")
