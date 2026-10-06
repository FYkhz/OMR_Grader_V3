from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from omr import (
    CHOICES,
    MARKER_MARGIN,
    MARKER_SIZE,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    PageLayout,
    build_page_layout,
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def create_sheet(
    total_questions: int,
    page_number: int,
    answer_marks: Mapping[int, str | None] | None = None,
    *,
    title: str = "OMR ANSWER SHEET",
) -> np.ndarray:
    layout: PageLayout = build_page_layout(total_questions, page_number)
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    marker_positions = [
        (MARKER_MARGIN, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
        (MARKER_MARGIN, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
    ]
    for x, y in marker_positions:
        draw.rectangle((x, y, x + MARKER_SIZE, y + MARKER_SIZE), fill="black")

    draw.text((PAGE_WIDTH // 2, 112), title, font=_font(52, bold=True), fill="black", anchor="ma")
    draw.text(
        (PAGE_WIDTH // 2, 172),
        f"Questions {layout.first_question}–{layout.last_question}  •  Page {layout.page_number} of {layout.page_count}",
        font=_font(31, bold=True),
        fill="black",
        anchor="ma",
    )
    draw.text(
        (PAGE_WIDTH // 2, 220),
        "Fill one bubble completely. Keep all four black corner squares visible when photographing.",
        font=_font(23),
        fill="black",
        anchor="ma",
    )
    draw.line((230, 265, 1470, 265), fill="black", width=3)
    draw.text((275, 305), "Name (optional): ______________________________", font=_font(27), fill="black")
    draw.text((1080, 305), "Date: ______________", font=_font(27), fill="black")

    # Column headers are positioned from the first question in each visual column.
    visual_columns: dict[int, list] = {}
    for q_layout in layout.questions:
        visual_columns.setdefault(q_layout.number_center[0], []).append(q_layout)

    for number_x, questions in visual_columns.items():
        first_q = questions[0]
        header_y = 415
        draw.text((number_x, header_y), "Q", font=_font(27, bold=True), fill="black", anchor="mm")
        for bubble in first_q.bubbles:
            draw.text((bubble.center[0], header_y), bubble.choice, font=_font(29, bold=True), fill="black", anchor="mm")

    for idx, q_layout in enumerate(layout.questions):
        if idx % 2 == 1:
            x1, y1, x2, y2 = q_layout.row_bounds
            draw.rectangle((x1, y1, x2, y2), fill=(246, 248, 251))

        draw.text(
            q_layout.number_center,
            str(q_layout.question),
            font=_font(25, bold=True),
            fill="black",
            anchor="mm",
        )
        selected = answer_marks.get(q_layout.question) if answer_marks else None
        for bubble in q_layout.bubbles:
            x, y = bubble.center
            r = bubble.radius
            draw.ellipse((x - r, y - r, x + r, y + r), outline="black", width=4)
            if selected == bubble.choice:
                inner = r - 6
                draw.ellipse((x - inner, y - inner, x + inner, y + inner), fill="black")

    draw.line((230, 2040, 1470, 2040), fill="black", width=3)
    draw.text(
        (PAGE_WIDTH // 2, 2082),
        f"Exam size: {layout.total_questions} questions • A–D choices • Page {layout.page_number}/{layout.page_count}",
        font=_font(23),
        fill="black",
        anchor="ma",
    )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def save_sheet(path: str | Path, total_questions: int, page_number: int) -> None:
    cv2.imwrite(str(path), create_sheet(total_questions, page_number))
