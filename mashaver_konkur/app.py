"""Streamlit UI for the Mashaver Konkur MVP."""
from __future__ import annotations

from datetime import date

import streamlit as st

from db_manager import DatabaseManager
from study_agent import StudyAgent


st.set_page_config(page_title="مشاور کنکور", layout="wide")

st.markdown(
    """
    <style>
    body { direction: rtl; text-align: right; }
    .block-container { direction: rtl; }
    .stMarkdown, .stTextInput, .stTextArea, .stSelectbox, .stButton, .stDataFrame {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_database() -> DatabaseManager:
    return DatabaseManager()


@st.cache_resource
def get_agent() -> StudyAgent:
    return StudyAgent()


db = get_database()
agent = get_agent()


def render_daily_plan_tab() -> None:
    st.header("🧾 برنامه درسی روزانه")
    st.write("بر اساس اطلاعات شما یک برنامه روزانه شخصی‌سازی شده تولید می‌شود.")

    with st.form("daily_plan_form"):
        grade = st.selectbox(
            "پایه تحصیلی",
            options=["دهم", "یازدهم", "دوازدهم", "فارغ‌التحصیل"],
        )
        major = st.selectbox(
            "رشته تحصیلی",
            options=["ریاضی", "تجربی", "انسانی", "هنر", "زبان"],
        )
        goals = st.text_area("اهداف و مباحث مهم", help="به عنوان مثال: جمع‌بندی مثلثات، تست تعادل شیمی")
        study_hours = st.slider("ساعات مطالعه روزانه", min_value=1, max_value=12, value=5)
        submitted = st.form_submit_button("تولید برنامه")

    if submitted:
        with st.spinner("در حال تولید برنامه..."):
            plan = agent.generate_daily_plan(grade, major, goals, study_hours)
        st.success("برنامه پیشنهادی آماده است.")
        st.json(plan, expanded=False)


def render_chat_tab() -> None:
    st.header("💬 گفت‌وگو با مشاور")
    st.caption("پرسش خود را مطرح کنید تا پاسخ شخصی‌سازی شده دریافت کنید.")

    conversations = db.get_conversations(limit=20)
    if conversations:
        st.subheader("گفت‌وگوهای اخیر")
        for convo in conversations:
            with st.expander(convo["question"], expanded=False):
                st.write(convo["answer"])

    with st.form("chat_form"):
        question = st.text_area("سوال خود را بنویسید", height=120)
        submitted = st.form_submit_button("ارسال سوال")

    if submitted and question.strip():
        with st.spinner("در حال آماده‌سازی پاسخ..."):
            answer = agent.answer_question(question)
        db.add_conversation(question, answer)
        st.success("پاسخ مشاور آماده شد.")
        st.write(answer)
        st.experimental_rerun()
    elif submitted:
        st.warning("لطفاً ابتدا سوال خود را بنویسید.")


def render_tasks_tab() -> None:
    st.header("✅ وظایف / پیشرفت")
    st.caption("وظایف مطالعاتی خود را مدیریت کنید و وضعیت آن‌ها را به‌روزرسانی کنید.")

    with st.form("add_task_form"):
        title = st.text_input("عنوان تسک")
        due = st.date_input("تاریخ انجام", value=date.today())
        notes = st.text_area("یادداشت‌ها", height=80)
        submitted = st.form_submit_button("افزودن تسک")

    if submitted:
        if title.strip():
            db.add_task(
                title=title,
                due_date=due.isoformat() if isinstance(due, date) else str(due),
                notes=notes.strip() or None,
            )
            st.success("تسک جدید افزوده شد.")
            st.experimental_rerun()
        else:
            st.warning("عنوان تسک نمی‌تواند خالی باشد.")

    tasks = db.get_tasks()
    if tasks:
        st.subheader("لیست تسک‌ها")
        status_options = ["شروع نشده", "در حال انجام", "انجام شده"]
        for task in tasks:
            container = st.container()
            container.markdown(f"**{task['title']}**")
            meta_cols = container.columns([1, 1, 2])
            meta_cols[0].markdown(f"📅 تاریخ: {task.get('due_date') or 'نامشخص'}")
            meta_cols[1].markdown(f"📝 یادداشت: {task.get('notes') or '—'}")
            current_status = task.get("status", "شروع نشده")
            selected_status = meta_cols[2].selectbox(
                "وضعیت",
                status_options,
                index=status_options.index(current_status) if current_status in status_options else 0,
                key=f"status_{task['id']}",
            )
            action_cols = container.columns([1, 1])
            if action_cols[0].button("ذخیره", key=f"save_{task['id']}"):
                db.update_task_status(task["id"], selected_status)
                st.success("وضعیت تسک به‌روزرسانی شد.")
                st.experimental_rerun()
            if action_cols[1].button("حذف", key=f"delete_{task['id']}"):
                db.delete_task(task["id"])
                st.warning("تسک حذف شد.")
                st.experimental_rerun()
    else:
        st.info("هنوز هیچ تسکی ثبت نشده است.")


def render_report_tab() -> None:
    st.header("🧠 گزارش هفتگی هوشمند")
    st.caption("از روی تسک‌ها و گفت‌وگوها، گزارش هوشمند تولید کنید.")

    if st.button("تولید گزارش هفتگی"):
        tasks = db.get_tasks()
        conversations = db.get_conversations(limit=10)
        with st.spinner("در حال تحلیل داده‌ها..."):
            summary = agent.generate_progress_summary(tasks, conversations)
        st.markdown("### نتیجه")
        st.write(summary)


TAB_BUILDERS = [
    render_daily_plan_tab,
    render_chat_tab,
    render_tasks_tab,
    render_report_tab,
]


def main() -> None:
    tabs = st.tabs([
        "🧾 برنامه درسی روزانه",
        "💬 گفت‌وگو با مشاور",
        "✅ وظایف / پیشرفت",
        "🧠 گزارش هفتگی هوشمند",
    ])

    for tab, builder in zip(tabs, TAB_BUILDERS):
        with tab:
            builder()


if __name__ == "__main__":
    main()
