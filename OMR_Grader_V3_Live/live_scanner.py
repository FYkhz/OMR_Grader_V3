from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Sequence

import cv2
import numpy as np

from omr import (
    ScanResult,
    answers_from_detections,
    build_page_layout,
    crop_question,
    find_registration_markers,
    grade_answers,
    page_count_for,
    scan_sheet,
)


@dataclass(frozen=True)
class LiveSnapshot:
    phase: str
    message: str
    expected_page: int
    page_count: int
    stable_frames: int
    stable_needed: int
    latest_record_id: int
    latest_grade: dict | None
    unresolved: tuple[int, ...]
    student_number: int
    captured_pages: tuple[int, ...]


class LiveGradeController:
    """Thread-safe state machine for continuous OMR grading from a WebRTC camera feed."""

    def __init__(
        self,
        total_questions: int,
        key_answers: Sequence[str],
        *,
        stable_needed: int = 4,
        absent_needed: int = 4,
        process_every_n_frames: int = 3,
    ) -> None:
        self.total_questions = int(total_questions)
        self.key_answers = tuple(key_answers)
        self.page_count = page_count_for(self.total_questions)
        self.stable_needed = max(2, int(stable_needed))
        self.absent_needed = max(2, int(absent_needed))
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self._lock = RLock()
        self._frame_index = 0
        self._phase = "WAITING"
        self._message = "Show page 1"
        self._expected_page = 1
        self._stable_frames = 0
        self._absent_frames = 0
        self._last_markers_norm: np.ndarray | None = None
        self._current_pages: dict[int, ScanResult] = {}
        self._latest_grade: dict | None = None
        self._latest_record_id = 0
        self._student_number = 1
        self._unresolved: tuple[int, ...] = ()
        self._history: list[dict] = []
        self._student_complete = False

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            return LiveSnapshot(
                phase=self._phase,
                message=self._message,
                expected_page=self._expected_page,
                page_count=self.page_count,
                stable_frames=self._stable_frames,
                stable_needed=self.stable_needed,
                latest_record_id=self._latest_record_id,
                latest_grade=dict(self._latest_grade) if self._latest_grade else None,
                unresolved=tuple(self._unresolved),
                student_number=self._student_number,
                captured_pages=tuple(sorted(self._current_pages)),
            )

    def history(self) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._history]

    def review_items(self) -> list[dict]:
        with self._lock:
            items: list[dict] = []
            for page_number, scan in sorted(self._current_pages.items()):
                for detection in scan.detections:
                    if detection.question not in self._unresolved:
                        continue
                    items.append(
                        {
                            "page": page_number,
                            "question": detection.question,
                            "status": detection.status,
                            "confidence": detection.confidence,
                            "densities": tuple(detection.densities),
                            "crop": crop_question(scan, detection.question).copy(),
                        }
                    )
            return items

    def resolve_review(self, overrides: dict[int, str | None]) -> dict:
        with self._lock:
            if self._phase != "REVIEW":
                raise RuntimeError("There is no paper waiting for review.")
            if any(q not in overrides for q in self._unresolved):
                raise ValueError("Every unclear question must be reviewed before continuing.")
            combined: dict[int, str | None] = {}
            for scan in self._current_pages.values():
                combined.update(answers_from_detections(scan.detections, overrides))
            self._finalize_locked(combined)
            self._phase = "WAIT_REMOVE"
            self._message = f"{self._latest_grade['score']}/{self._latest_grade['total']} — remove paper"
            self._student_complete = True
            return dict(self._latest_grade)

    def reset_current_student(self) -> None:
        with self._lock:
            self._reset_for_next_student_locked(increment_student=False)
            self._message = "Current scan reset — show page 1"

    def process_frame(self, image_bgr: np.ndarray) -> np.ndarray:
        with self._lock:
            self._frame_index += 1
            frame_index = self._frame_index

        if frame_index % self.process_every_n_frames != 0:
            return self._annotate(image_bgr.copy(), None)

        markers = self._find_markers_fast(image_bgr)

        with self._lock:
            if self._phase == "REVIEW":
                return self._annotate(image_bgr.copy(), markers)

            if self._phase == "WAIT_REMOVE":
                if markers is None:
                    self._absent_frames += 1
                    if self._absent_frames >= self.absent_needed:
                        self._after_paper_removed_locked()
                else:
                    self._absent_frames = 0
                return self._annotate(image_bgr.copy(), markers)

            if markers is None:
                self._stable_frames = 0
                self._last_markers_norm = None
                self._message = f"Show page {self._expected_page}"
                return self._annotate(image_bgr.copy(), None)

            marker_norm = markers / np.array(
                [[image_bgr.shape[1], image_bgr.shape[0]]], dtype=np.float32
            )
            if self._last_markers_norm is None:
                stable = False
            else:
                movement = float(np.mean(np.linalg.norm(marker_norm - self._last_markers_norm, axis=1)))
                stable = movement < 0.009

            self._last_markers_norm = marker_norm
            self._stable_frames = self._stable_frames + 1 if stable else 1

            sharpness = self._sharpness(image_bgr, markers)
            if sharpness < 35.0:
                self._stable_frames = max(1, self._stable_frames - 1)
                self._message = "Image is blurry — hold steady"
            else:
                self._message = f"Hold steady {self._stable_frames}/{self.stable_needed}"

            if self._stable_frames < self.stable_needed:
                return self._annotate(image_bgr.copy(), markers)

            expected_page = self._expected_page

        layout = build_page_layout(self.total_questions, expected_page)
        scan = scan_sheet(image_bgr, layout=layout, answer_key_mode=False)

        with self._lock:
            self._current_pages[expected_page] = scan
            self._stable_frames = 0
            self._last_markers_norm = None
            self._absent_frames = 0

            if expected_page < self.page_count:
                self._phase = "WAIT_REMOVE"
                self._student_complete = False
                self._message = f"Page {expected_page} captured — remove it"
            else:
                self._complete_student_locked()

            return self._annotate(image_bgr.copy(), markers)

    def _complete_student_locked(self) -> None:
        unresolved: list[int] = []
        combined: dict[int, str | None] = {}
        for scan in self._current_pages.values():
            combined.update(answers_from_detections(scan.detections))
            unresolved.extend(d.question for d in scan.detections if d.status != "clear")

        self._unresolved = tuple(sorted(set(unresolved)))
        if self._unresolved:
            self._phase = "REVIEW"
            self._student_complete = False
            self._message = f"Review {len(self._unresolved)} unclear question(s) below"
            return

        self._finalize_locked(combined)
        self._phase = "WAIT_REMOVE"
        self._student_complete = True
        self._message = f"{self._latest_grade['score']}/{self._latest_grade['total']} — remove paper"

    def _finalize_locked(self, combined: dict[int, str | None]) -> None:
        grade = grade_answers(combined, self.key_answers)
        self._latest_record_id += 1
        grade = dict(grade)
        grade["record_id"] = self._latest_record_id
        grade["student_number"] = self._student_number
        grade["time"] = datetime.now().strftime("%I:%M:%S %p")
        self._latest_grade = grade
        self._history.append(
            {
                "Scan": self._student_number,
                "Time": grade["time"],
                "Score": grade["score"],
                "Total": grade["total"],
                "Percentage": round(grade["percentage"], 1),
                "Incorrect": grade["incorrect"],
                "Blank": grade["blank"],
            }
        )
        self._unresolved = ()

    def _after_paper_removed_locked(self) -> None:
        self._absent_frames = 0
        if self._student_complete:
            self._reset_for_next_student_locked(increment_student=True)
            return
        self._expected_page += 1
        self._phase = "WAITING"
        self._message = f"Show page {self._expected_page}"

    def _reset_for_next_student_locked(self, *, increment_student: bool) -> None:
        if increment_student:
            self._student_number += 1
        self._phase = "WAITING"
        self._message = "Show page 1"
        self._expected_page = 1
        self._stable_frames = 0
        self._absent_frames = 0
        self._last_markers_norm = None
        self._current_pages = {}
        self._unresolved = ()
        self._student_complete = False

    @staticmethod
    def _find_markers_fast(image_bgr: np.ndarray) -> np.ndarray | None:
        h, w = image_bgr.shape[:2]
        scale = min(1.0, 900.0 / max(w, 1))
        if scale < 1.0:
            small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = image_bgr
        markers = find_registration_markers(small)
        if markers is None:
            return None
        return markers / scale

    @staticmethod
    def _sharpness(image_bgr: np.ndarray, markers: np.ndarray) -> float:
        x1 = max(0, int(np.min(markers[:, 0])))
        x2 = min(image_bgr.shape[1], int(np.max(markers[:, 0])))
        y1 = max(0, int(np.min(markers[:, 1])))
        y2 = min(image_bgr.shape[0], int(np.max(markers[:, 1])))
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        if roi.shape[1] > 800:
            scale = 800 / roi.shape[1]
            roi = cv2.resize(roi, (800, max(1, int(roi.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _annotate(self, image: np.ndarray, markers: np.ndarray | None) -> np.ndarray:
        with self._lock:
            phase = self._phase
            message = self._message
            expected_page = self._expected_page
            page_count = self.page_count
            student_number = self._student_number
            latest_grade = dict(self._latest_grade) if self._latest_grade else None
            student_complete = self._student_complete

        if markers is not None:
            pts = markers.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 220, 0), 4)

        h, w = image.shape[:2]
        scale = max(0.65, min(1.35, w / 900.0))
        panel_h = int(110 * scale)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, min(h, panel_h)), (18, 28, 45), -1)
        cv2.addWeighted(overlay, 0.80, image, 0.20, 0, image)

        if phase == "REVIEW":
            color = (0, 165, 255)
        elif phase == "WAIT_REMOVE":
            color = (0, 220, 220)
        else:
            color = (80, 220, 80)

        cv2.putText(
            image,
            f"OMR LIVE  |  Student {student_number}  |  Page {expected_page}/{page_count}",
            (int(20 * scale), int(38 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70 * scale,
            (255, 255, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            message,
            (int(20 * scale), int(82 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78 * scale,
            color,
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )

        if latest_grade is not None and phase == "WAIT_REMOVE" and student_complete:
            score_text = f"{latest_grade['score']}/{latest_grade['total']}  ({latest_grade['percentage']:.1f}%)"
            size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, 2)[0]
            cv2.putText(
                image,
                score_text,
                (max(20, w - size[0] - int(24 * scale)), int(82 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.90 * scale,
                (80, 240, 80),
                max(1, int(2 * scale)),
                cv2.LINE_AA,
            )
        return image
