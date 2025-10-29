"""Gemini API integration and fallbacks for Mashaver Konkur."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv


class StudyAgent:
    """Wrapper around Gemini with safe fallbacks when API access is unavailable."""

    def __init__(self, model_name: str = "gemini-pro") -> None:
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.model: Optional[genai.GenerativeModel] = None

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception:
                # Keep model as None to enable graceful fallbacks.
                self.model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_daily_plan(
        self,
        grade: str,
        major: str,
        goals: str,
        study_hours: int,
    ) -> List[Dict[str, Any]]:
        prompt = (
            "تو یک مشاور تحصیلی حرفه‌ای هستی.\n"
            "بر اساس رشته و پایه تحصیلی و زمان مطالعه، یک برنامه روزانه برای دانش‌آموز کنکوری بنویس.\n"
            "خروجی را در قالب JSON بده با ساختار:\n"
            "[\n"
            "{\"subject\": \"ریاضی\", \"duration\": \"90 دقیقه\", \"goal\": \"تست حد و پیوستگی\"},\n"
            "{\"subject\": \"ادبیات\", \"duration\": \"60 دقیقه\", \"goal\": \"قرابت معنایی\"}\n"
            "]\n"
            f"پایه تحصیلی: {grade}\n"
            f"رشته: {major}\n"
            f"هدف‌ها: {goals}\n"
            f"زمان مطالعه روزانه: {study_hours} ساعت\n"
        )
        response_text = self._call_model(prompt)
        if response_text:
            plan = self._parse_json_list(response_text)
            if plan:
                return plan
        return self._fallback_daily_plan(goals)

    def answer_question(self, question: str) -> str:
        prompt = (
            "تو نقش یک مشاور کنکور حرفه‌ای را داری.\n"
            "سوال زیر را به فارسی ساده و کاربردی پاسخ بده:\n"
            f"{question}\n"
        )
        response_text = self._call_model(prompt)
        if response_text:
            return response_text.strip()
        return self._fallback_answer(question)

    def generate_progress_summary(
        self,
        tasks: List[Dict[str, Any]],
        conversations: List[Dict[str, Any]],
    ) -> str:
        progress_payload = json.dumps(
            {
                "tasks": tasks,
                "conversations": conversations,
            },
            ensure_ascii=False,
        )
        prompt = (
            "این لیست از تسک‌های دانش‌آموز و سوالات اخیر اوست:\n"
            f"{progress_payload}\n"
            "خلاصه عملکرد و پیشنهاد سه اقدام مهم برای هفته آینده را بده."
        )
        response_text = self._call_model(prompt)
        if response_text:
            return response_text.strip()
        return self._fallback_summary(tasks, conversations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _call_model(self, prompt: str) -> Optional[str]:
        if not self.model:
            return None
        try:
            response = self.model.generate_content(prompt)
            if response and hasattr(response, "text"):
                return response.text
        except Exception:
            return None
        return None

    @staticmethod
    def _parse_json_list(text: str) -> Optional[List[Dict[str, Any]]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                snippet = match.group(0)
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _fallback_daily_plan(goals: str) -> List[Dict[str, Any]]:
        return [
            {
                "subject": "ریاضی",
                "duration": "۹۰ دقیقه",
                "goal": "حل تست‌های مرتبط با مشتق و حد",
            },
            {
                "subject": "ادبیات فارسی",
                "duration": "۶۰ دقیقه",
                "goal": "مرور قرابت معنایی و لغت",
            },
            {
                "subject": "دروس تخصصی",
                "duration": "۷۵ دقیقه",
                "goal": goals or "مرور نکات کلیدی فصل جاری",
            },
            {
                "subject": "استراحت و جمع‌بندی",
                "duration": "۳۰ دقیقه",
                "goal": "مرور نکات مهم روز",
            },
        ]

    @staticmethod
    def _fallback_answer(question: str) -> str:
        return (
            "برای استفاده کامل از مشاور کنکور لازم است کلید Gemini را در فایل .env تنظیم کنید. "
            "تا آن زمان می‌توانید سوالات خود را در منابع معتبر یا با دبیران در میان بگذارید."
        )

    @staticmethod
    def _fallback_summary(
        tasks: List[Dict[str, Any]],
        conversations: List[Dict[str, Any]],
    ) -> str:
        completed = [task for task in tasks if task.get("status") == "انجام شده"]
        pending = [task for task in tasks if task.get("status") != "انجام شده"]
        summary_lines = [
            "گزارش هفتگی (نمونه بدون اتصال به Gemini):",
            f"- تعداد تسک‌های تکمیل‌شده: {len(completed)}",
            f"- تعداد تسک‌های در انتظار: {len(pending)}",
            "پیشنهادها:",
            "1. برنامه‌ریزی زمانی مشخص برای مرور مباحث باقی‌مانده داشته باشید.",
            "2. از سوالات مطرح‌شده در گفت‌وگوها برای جمع‌بندی یادداشت تهیه کنید.",
            "3. یک آزمون شبیه‌سازی در پایان هفته برگزار کنید تا نقاط ضعف مشخص شود.",
        ]
        if conversations:
            summary_lines.insert(1, f"- تعداد سوالات مطرح‌شده: {len(conversations)}")
        return "\n".join(summary_lines)


__all__ = ["StudyAgent"]
