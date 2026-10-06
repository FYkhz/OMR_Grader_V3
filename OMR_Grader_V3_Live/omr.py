from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import cv2
import numpy as np


PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200
MARKER_SIZE = 70
MARKER_MARGIN = 75
QUESTIONS_PER_PAGE = 50
CHOICES = ("A", "B", "C", "D")

MARKER_CENTERS = np.array(
    [
        [MARKER_MARGIN + MARKER_SIZE / 2, MARKER_MARGIN + MARKER_SIZE / 2],
        [PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE / 2, MARKER_MARGIN + MARKER_SIZE / 2],
        [PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE / 2, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE / 2],
        [MARKER_MARGIN + MARKER_SIZE / 2, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE / 2],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class BubbleLocation:
    question: int
    choice: str
    center: tuple[int, int]
    radius: int
    inner_radius: int


@dataclass(frozen=True)
class QuestionLayout:
    question: int
    number_center: tuple[int, int]
    bubbles: tuple[BubbleLocation, ...]
    row_bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class PageLayout:
    total_questions: int
    page_number: int
    page_count: int
    first_question: int
    last_question: int
    questions: tuple[QuestionLayout, ...]


@dataclass(frozen=True)
class QuestionDetection:
    question: int
    densities: tuple[float, float, float, float]
    selected: str | None
    status: str
    confidence: float
    row_bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class ScanResult:
    aligned_bgr: np.ndarray
    threshold: np.ndarray
    detections: tuple[QuestionDetection, ...]
    layout: PageLayout
    marker_detection_used: bool
    warning: str | None = None


class OMRProcessingError(RuntimeError):
    pass


def validate_total_questions(total_questions: int) -> int:
    total = int(total_questions)
    if not 10 <= total <= 100:
        raise ValueError("Question count must be between 10 and 100.")
    return total


def page_count_for(total_questions: int) -> int:
    return int(ceil(validate_total_questions(total_questions) / QUESTIONS_PER_PAGE))


def questions_on_page(total_questions: int, page_number: int) -> tuple[int, int]:
    total = validate_total_questions(total_questions)
    pages = page_count_for(total)
    if page_number < 1 or page_number > pages:
        raise ValueError(f"Page number must be from 1 to {pages}.")
    first = (page_number - 1) * QUESTIONS_PER_PAGE + 1
    last = min(total, page_number * QUESTIONS_PER_PAGE)
    return first, last


def build_page_layout(total_questions: int, page_number: int) -> PageLayout:
    total = validate_total_questions(total_questions)
    pages = page_count_for(total)
    first, last = questions_on_page(total, page_number)
    count = last - first + 1

    # One column for up to 25 questions, two columns for 26–50.
    columns = 1 if count <= 25 else 2
    left_count = count if columns == 1 else int(ceil(count / 2))
    right_count = 0 if columns == 1 else count - left_count
    rows_max = max(left_count, right_count)

    top_y = 500
    bottom_y = 1965
    row_gap = 0 if rows_max <= 1 else (bottom_y - top_y) / (rows_max - 1)
    bubble_radius = 24 if rows_max <= 20 else 21
    inner_radius = 15 if rows_max <= 20 else 13

    if columns == 1:
        number_xs = [420]
        bubble_xs = [[650, 860, 1070, 1280]]
        column_bounds = [(300, 1415)]
        counts = [count]
    else:
        number_xs = [175, 935]
        bubble_xs = [
            [310, 420, 530, 640],
            [1070, 1180, 1290, 1400],
        ]
        column_bounds = [(115, 715), (875, 1475)]
        counts = [left_count, right_count]

    question_layouts: list[QuestionLayout] = []
    q = first
    for col_idx, col_count in enumerate(counts):
        for row_idx in range(col_count):
            y = int(round(top_y + row_idx * row_gap)) if rows_max > 1 else top_y
            bubbles = tuple(
                BubbleLocation(
                    question=q,
                    choice=choice,
                    center=(int(x), y),
                    radius=bubble_radius,
                    inner_radius=inner_radius,
                )
                for choice, x in zip(CHOICES, bubble_xs[col_idx])
            )
            x1, x2 = column_bounds[col_idx]
            row_half = max(26, int(row_gap * 0.42)) if rows_max > 1 else 34
            bounds = (x1, max(330, y - row_half), x2, min(PAGE_HEIGHT - 150, y + row_half))
            question_layouts.append(
                QuestionLayout(
                    question=q,
                    number_center=(number_xs[col_idx], y),
                    bubbles=bubbles,
                    row_bounds=bounds,
                )
            )
            q += 1

    return PageLayout(
        total_questions=total,
        page_number=page_number,
        page_count=pages,
        first_question=first,
        last_question=last,
        questions=tuple(question_layouts),
    )


def decode_image_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise OMRProcessingError("The uploaded file could not be decoded as an image.")
    return image


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise OMRProcessingError("The image could not be encoded.")
    return bytes(buffer)


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def find_registration_markers(image_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    image_area = float(h * w)
    candidates: list[tuple[float, float, float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.00025 or area > image_area * 0.05:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw == 0 or ch == 0:
            continue
        aspect = cw / float(ch)
        extent = area / float(cw * ch)
        if not 0.70 <= aspect <= 1.30 or extent < 0.70:
            continue
        candidates.append((x + cw / 2.0, y + ch / 2.0, area))

    if len(candidates) < 4:
        return None

    targets = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    chosen: list[tuple[float, float]] = []
    used: set[int] = set()
    for target in targets:
        scored: list[tuple[float, int]] = []
        for idx, (cx, cy, area) in enumerate(candidates):
            if idx in used:
                continue
            dist = float(np.linalg.norm(np.array([cx, cy]) - target))
            scored.append((dist - np.sqrt(area) * 0.20, idx))
        if not scored:
            return None
        _, best = min(scored)
        used.add(best)
        cx, cy, _ = candidates[best]
        chosen.append((cx, cy))

    ordered = _order_points(np.array(chosen, dtype=np.float32))
    tl, tr, br, bl = ordered
    horizontal = min(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    vertical = min(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if horizontal < w * 0.45 or vertical < h * 0.45:
        return None
    return ordered


def _fallback_page_warp(image_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        if cv2.contourArea(approx) < image_bgr.shape[0] * image_bgr.shape[1] * 0.25:
            continue
        source = _order_points(approx.reshape(4, 2))
        destination = np.array(
            [[0, 0], [PAGE_WIDTH - 1, 0], [PAGE_WIDTH - 1, PAGE_HEIGHT - 1], [0, PAGE_HEIGHT - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        aligned = cv2.warpPerspective(image_bgr, matrix, (PAGE_WIDTH, PAGE_HEIGHT))
        return aligned, (
            "The corner squares were not found, so the page boundary was used. "
            "Keep all four black squares visible for better accuracy."
        )

    return cv2.resize(image_bgr, (PAGE_WIDTH, PAGE_HEIGHT), interpolation=cv2.INTER_AREA), (
        "The page could not be aligned reliably and was resized directly. Retake the photo from above with the full page visible."
    )


def align_sheet(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str | None]:
    markers = find_registration_markers(image_bgr)
    if markers is None:
        aligned, warning = _fallback_page_warp(image_bgr)
        return aligned, False, warning
    matrix = cv2.getPerspectiveTransform(markers, MARKER_CENTERS)
    aligned = cv2.warpPerspective(
        image_bgr,
        matrix,
        (PAGE_WIDTH, PAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned, True, None


def _prepare_threshold(aligned_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        12,
    )
    return cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _density_for_circle(binary: np.ndarray, center: tuple[int, int], radius: int) -> float:
    cx, cy = center
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(binary.shape[1], cx + radius + 1)
    y2 = min(binary.shape[0], cy + radius + 1)
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(mask, (cx - x1, cy - y1), radius, 255, -1)
    dark_pixels = cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=mask))
    total_pixels = cv2.countNonZero(mask)
    return float(dark_pixels / total_pixels) if total_pixels else 0.0


def classify_densities(densities: Sequence[float], *, answer_key_mode: bool) -> tuple[str | None, str, float]:
    values = np.asarray(densities, dtype=np.float32)
    order = np.argsort(values)[::-1]
    top_index = int(order[0])
    top = float(values[top_index])
    second = float(values[int(order[1])])
    margin = top - second
    confidence = float(np.clip((margin / 0.22) * 100.0, 0.0, 100.0))

    blank_threshold = 0.10 if answer_key_mode else 0.085
    clear_threshold = 0.20 if answer_key_mode else 0.17
    second_mark_threshold = 0.14 if answer_key_mode else 0.13
    minimum_margin = 0.085 if answer_key_mode else 0.070

    if top < blank_threshold:
        return None, "blank", 0.0
    if second >= second_mark_threshold and (margin < 0.13 or second / max(top, 1e-6) > 0.62):
        return None, "multiple", confidence
    if top < clear_threshold or margin < minimum_margin:
        return None, "unclear", confidence
    return CHOICES[top_index], "clear", confidence


def scan_sheet(image_bgr: np.ndarray, *, layout: PageLayout, answer_key_mode: bool) -> ScanResult:
    aligned, marker_used, warning = align_sheet(image_bgr)
    threshold = _prepare_threshold(aligned)
    detections: list[QuestionDetection] = []

    for q_layout in layout.questions:
        densities = tuple(
            _density_for_circle(threshold, bubble.center, bubble.inner_radius)
            for bubble in q_layout.bubbles
        )
        selected, status, confidence = classify_densities(densities, answer_key_mode=answer_key_mode)
        detections.append(
            QuestionDetection(
                question=q_layout.question,
                densities=densities,
                selected=selected,
                status=status,
                confidence=confidence,
                row_bounds=q_layout.row_bounds,
            )
        )

    return ScanResult(
        aligned_bgr=aligned,
        threshold=threshold,
        detections=tuple(detections),
        layout=layout,
        marker_detection_used=marker_used,
        warning=warning,
    )


def answers_from_detections(
    detections: Sequence[QuestionDetection], overrides: dict[int, str | None] | None = None
) -> dict[int, str | None]:
    overrides = overrides or {}
    answers: dict[int, str | None] = {}
    for detection in detections:
        value = overrides.get(detection.question, detection.selected)
        answers[detection.question] = None if value in (None, "Blank") else value
    return answers


def grade_answers(student_answers: dict[int, str | None], key_answers: Sequence[str]) -> dict:
    rows: list[dict] = []
    score = 0
    incorrect = 0
    blank = 0
    for question, correct in enumerate(key_answers, start=1):
        student = student_answers.get(question)
        if student is None:
            status = "Blank"
            blank += 1
        elif student == correct:
            status = "Correct"
            score += 1
        else:
            status = "Incorrect"
            incorrect += 1
        rows.append(
            {
                "Question": question,
                "Student": student or "Blank",
                "Correct answer": correct,
                "Result": status,
            }
        )
    total = len(key_answers)
    return {
        "score": score,
        "total": total,
        "percentage": (score / total * 100.0) if total else 0.0,
        "incorrect": incorrect,
        "blank": blank,
        "rows": rows,
    }


def draw_overlay(
    scan: ScanResult,
    *,
    key_answers: Sequence[str] | None = None,
    overrides: dict[int, str | None] | None = None,
) -> np.ndarray:
    image = scan.aligned_bgr.copy()
    overrides = overrides or {}
    layout_by_q = {q.question: q for q in scan.layout.questions}

    for detection in scan.detections:
        q_layout = layout_by_q[detection.question]
        selected = overrides.get(detection.question, detection.selected)
        if selected == "Blank":
            selected = None

        if detection.status != "clear" and detection.question not in overrides:
            color = (0, 165, 255)  # orange
        elif selected is None:
            color = (140, 140, 140)
        elif key_answers is None:
            color = (0, 170, 0)
        else:
            correct = key_answers[detection.question - 1]
            color = (0, 170, 0) if selected == correct else (0, 0, 220)

        x1, y1, x2, y2 = detection.row_bounds
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)

        for bubble in q_layout.bubbles:
            thickness = 5 if selected == bubble.choice else 2
            cv2.circle(image, bubble.center, bubble.radius + 5, color, thickness)

    return image


def crop_question(scan: ScanResult, question: int) -> np.ndarray:
    detection = next((d for d in scan.detections if d.question == question), None)
    if detection is None:
        raise ValueError(f"Question {question} is not on this page.")
    x1, y1, x2, y2 = detection.row_bounds
    pad_x, pad_y = 18, 18
    return scan.aligned_bgr[
        max(0, y1 - pad_y) : min(PAGE_HEIGHT, y2 + pad_y),
        max(0, x1 - pad_x) : min(PAGE_WIDTH, x2 + pad_x),
    ]


def detection_rows(detections: Sequence[QuestionDetection]) -> list[dict]:
    return [
        {
            "Question": d.question,
            "Detected": d.selected or "—",
            "Status": d.status.title(),
            "Confidence": f"{d.confidence:.0f}%",
            "A ink": f"{d.densities[0] * 100:.1f}%",
            "B ink": f"{d.densities[1] * 100:.1f}%",
            "C ink": f"{d.densities[2] * 100:.1f}%",
            "D ink": f"{d.densities[3] * 100:.1f}%",
        }
        for d in detections
    ]
