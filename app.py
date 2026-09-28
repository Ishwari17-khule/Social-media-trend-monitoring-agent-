"""Real-time AI-powered Social Media Trend Monitoring Dashboard."""

from __future__ import annotations

from datetime import date, datetime
from io import BytesIO

import streamlit as st
from config import ASSETS_DIR, DEFAULT_QUERY, FETCH_INTERVAL_SECONDS, setup_logging
from database.auth_db import AuthManager

logger = setup_logging()

st.set_page_config(
    page_title="Social Media Trend Monitoring Dashboard",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Load dashboard CSS."""
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def get_auth() -> AuthManager:
    """Return the shared auth manager."""
    return AuthManager()


# Lazy load heavy imports after authentication
def get_predictor():
    """Load the saved model once per Streamlit process."""
    from predict import SentimentPredictor
    if "predictor" not in st.session_state:
        st.session_state.predictor = SentimentPredictor()
    return st.session_state.predictor


def get_live_service():
    """Return the shared live data service."""
    from fetchers.live_data import LiveDataService
    if "live_service" not in st.session_state:
        st.session_state.live_service = LiveDataService()
    return st.session_state.live_service


def get_database():
    """Return the shared database manager."""
    from database.db import DatabaseManager
    if "database" not in st.session_state:
        st.session_state.database = DatabaseManager()
    return st.session_state.database


def require_login() -> None:
    """Use Case 1: User Login. Gate the dashboard behind username/password."""
    if st.session_state.get("authenticated"):
        return

    load_css()
    st.markdown('<div class="login-shell">', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.05, 1])
    with mid:
        st.markdown(
            """
            <div class="login-brand">
                <div class="login-logo">S</div>
                <h1>Social Trend Agent</h1>
                <p>AI-powered social media trend &amp; sentiment monitoring</p>
                <div class="login-badges">
                    <span class="login-badge">Live Data</span>
                    <span class="login-badge">Sentiment AI</span>
                    <span class="login-badge">Topic Clusters</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        # Lazy load auth only when needed for login
        auth = get_auth()
        tab_login, tab_register = st.tabs(["Login", "Create Account"])
        with tab_login:
            st.markdown(
                """
                <div class="login-card-title">Welcome back</div>
                <div class="login-card-subtitle">Sign in to open your dashboard.</div>
                """,
                unsafe_allow_html=True,
            )
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input(
                "Password", type="password", key="login_password", placeholder="Enter your password"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                remember_me = st.checkbox("Remember me", key="remember_me")
            with col2:
                st.markdown('<div style="text-align: right; padding-top: 0.5rem;"><a href="#" style="color: #2b8dde; text-decoration: none; font-size: 0.85rem;">Forgot password?</a></div>', unsafe_allow_html=True)
            
            login_btn = st.button("Sign In", width="stretch", type="primary")
            
            if login_btn:
                if not username or not password:
                    st.error("Please enter both a username and a password.")
                else:
                    with st.spinner("Signing in..."):
                        user = auth.verify_user(username, password)
                        if user:
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = user
                            if remember_me:
                                st.session_state["remember_me"] = True
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
            st.markdown(
                """
                <div class="login-hint">
                    Default admin account: <code>admin</code> / <code>admin123</code>
                    &mdash; change this after your first login.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tab_register:
            st.markdown(
                """
                <div class="login-card-title">Create an account</div>
                <div class="login-card-subtitle">Takes less than a minute.</div>
                """,
                unsafe_allow_html=True,
            )
            new_username = st.text_input("New Username", key="reg_username", placeholder="Choose a username")
            new_password = st.text_input(
                "New Password", type="password", key="reg_password", placeholder="At least 6 characters"
            )
            confirm_password = st.text_input(
                "Confirm Password", type="password", key="reg_confirm_password", placeholder="Confirm your password"
            )
            
            register_btn = st.button("Create Account", width="stretch", type="primary")
            
            if register_btn:
                if not new_username or not new_password:
                    st.error("Please fill in all fields.")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Creating account..."):
                        success, message = auth.register_user(new_username, new_password)
                        if success:
                            st.success(message)
                            st.info("You can now sign in with your new account.")
                        else:
                            st.error(message)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="login-footer">Secured session &bull; PostgreSQL-backed authentication</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


@st.cache_data(ttl=FETCH_INTERVAL_SECONDS, show_spinner=False)
def fetch_score_and_store(query: str) -> tuple[pd.DataFrame, str, int]:
    """Fetch live data, score sentiment, store in PostgreSQL, and return records."""
    logger.info("Dashboard refresh started for query: %s", query)
    live_df = get_live_service().fetch(query)
    scored_df = get_predictor().predict_dataframe(live_df)

    inserted = 0
    database = get_database()
    if database.is_available():
        inserted = database.upsert_articles(scored_df)
        stored = database.fetch_articles(limit=1000)
        if not stored.empty:
            scored_df = stored
    else:
        logger.info("Using in-memory records because PostgreSQL is unavailable.")

    import pandas as pd
    scored_df["published_at"] = pd.to_datetime(scored_df["published_at"], errors="coerce", utc=True)
    last_updated = datetime.now().strftime("%d %b %Y %I:%M:%S %p")
    return scored_df, last_updated, inserted


@st.cache_data(ttl=FETCH_INTERVAL_SECONDS, show_spinner=False)
def get_topic_clusters(clean_texts: pd.Series):
    """Run BERTopic clustering on the current filtered dataset."""
    from src.topic_modeling import extract_topics
    return extract_topics(clean_texts)


@st.cache_data(ttl=FETCH_INTERVAL_SECONDS, show_spinner=False)
def get_ai_briefing(keywords_df: pd.DataFrame, sentiment_counts: dict, topics_df: pd.DataFrame, article_count: int) -> str:
    """Generate the Gemini-powered AI trend briefing (Section 6.6)."""
    from src.ai_summary import generate_trend_briefing
    return generate_trend_briefing(keywords_df, sentiment_counts, topics_df, article_count)


def kpi_card(label: str, value: str, note: str, theme: str) -> None:
    """Render a dashboard KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card {theme}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str) -> None:
    """Open a styled panel."""
    st.markdown(f"<div class='panel'><div class='panel-title'>{title}</div>", unsafe_allow_html=True)


def end_panel() -> None:
    """Close a styled panel."""
    st.markdown("</div>", unsafe_allow_html=True)


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to the scored article data."""
    import pandas as pd
    filtered = df.copy()
    filtered["article_date"] = filtered["published_at"].dt.date
    valid_dates = filtered["article_date"].dropna()
    min_date = valid_dates.min() if not valid_dates.empty else date.today()
    max_date = valid_dates.max() if not valid_dates.empty else date.today()

    with st.sidebar:
        st.markdown("### Filters")
        view = st.radio("View", ["Dashboard Home", "Analytics View", "Database Status"])
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        keyword = st.text_input("Search Keyword", placeholder="Enter keyword...")
        source = st.selectbox("Source Filter", ["All Sources"] + sorted(filtered["source"].dropna().unique().tolist()))
        sentiment = st.selectbox("Sentiment Filter", ["All"] + sorted(filtered["sentiment"].dropna().unique().tolist()))
        top_n = st.slider("Top Keywords", 5, 30, 12)
        auto_refresh = st.toggle("Auto Refresh", value=True)
        if st.button("Refresh Now", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    st.session_state["auto_refresh"] = auto_refresh

    filtered = filtered[
        filtered["article_date"].isna()
        | ((filtered["article_date"] >= start_date) & (filtered["article_date"] <= end_date))
    ]
    if source != "All Sources":
        filtered = filtered[filtered["source"] == source]
    if sentiment != "All":
        filtered = filtered[filtered["sentiment"] == sentiment]
    if keyword:
        keyword_mask = (
            filtered["title"].fillna("").str.contains(keyword, case=False, na=False)
            | filtered["description"].fillna("").str.contains(keyword, case=False, na=False)
            | filtered["clean_text"].fillna("").str.contains(keyword, case=False, na=False)
        )
        filtered = filtered[keyword_mask]

    st.session_state["dashboard_view"] = view
    st.session_state["top_n"] = top_n
    return filtered


def table_view(df: pd.DataFrame, trend: bool = False) -> pd.DataFrame:
    """Return display-ready table data."""
    table = df.copy()
    table["Published Date"] = table["published_at"].dt.strftime("%d %b %Y %I:%M %p").fillna("Unknown")
    table["Source"] = table["source"].fillna("Unknown")
    table["Headline"] = table["title"].fillna("")
    table["Sentiment"] = table["sentiment"].fillna("Unknown")
    table["Confidence Score"] = table["confidence"].round(2)
    table["URL"] = table["url"].fillna("")
    columns = ["Published Date", "Source", "Headline", "Sentiment", "Confidence Score", "URL"]
    if trend:
        table["Trend Score"] = table["trend_score"].round(2)
        table = table.sort_values("Trend Score", ascending=False)
        columns = ["Published Date", "Source", "Headline", "Trend Score", "Sentiment", "Confidence Score", "URL"]
    return table[columns]


def to_excel(df: pd.DataFrame) -> BytesIO:
    """Create an Excel download in memory."""
    import pandas as pd
    from openpyxl import load_workbook
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="dashboard_data")
    output.seek(0)
    return output


def _wrap_text(text: str, width: int = 95) -> str:
    """Wrap plain text for a fixed-width PDF text block."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width)) or text


def to_pdf(df: pd.DataFrame, keywords_df: pd.DataFrame, briefing_text: str = "", topics_df: pd.DataFrame | None = None) -> BytesIO:
    """Create a PDF report including keywords, topics, and the AI trend briefing."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output = BytesIO()
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        text = [
            "Social Media Trend Monitoring Dashboard",
            f"Generated: {datetime.now().strftime('%d %b %Y %I:%M %p')}",
            f"Total Articles: {len(df):,}",
            "",
            "Top Keywords:",
        ]
        text.extend(f"- {row.keyword}: {row.frequency}" for row in keywords_df.head(10).itertuples())

        if briefing_text:
            text.extend(["", "AI Trend Briefing:", _wrap_text(briefing_text)])

        if topics_df is not None and not topics_df.empty:
            text.extend(["", "Topic Clusters (BERTopic):"])
            text.extend(f"- Topic {row.Topic}: {row.Keywords}" for row in topics_df.head(8).itertuples())

        ax.text(0.05, 0.95, "\n".join(text), va="top", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    output.seek(0)
    return output


require_login()
load_css()

# Show loading indicator while initializing heavy components
with st.spinner("Initializing dashboard components..."):
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `py -3 train_model.py` once. The dashboard will not retrain automatically.")
        st.stop()

with st.sidebar:
    st.title("Trend Agent")
    user = st.session_state.get("user", {})
    st.caption(f"Signed in as **{user.get('username', 'user')}** ({user.get('role', 'user')})")
    if st.button("Logout", width="stretch"):
        st.session_state["authenticated"] = False
        st.rerun()
    query = st.text_input("Live Query", value=DEFAULT_QUERY)


@st.fragment(run_every=f"{FETCH_INTERVAL_SECONDS}s")
def render_dashboard(query: str) -> None:
    """Render the auto-refreshing part of the dashboard.

    Runs inside a Streamlit fragment so periodic updates only re-execute
    this block in place (an in-page network call) instead of reloading the
    whole browser page. A full page reload would start a brand new
    Streamlit session and wipe st.session_state, which is what used to
    force the user back to the login screen every refresh cycle.
    """
    from utils.preprocess import detect_trending_keywords
    with st.spinner("Fetching live data, scoring sentiment, and updating dashboard..."):
        data, last_updated, inserted_rows = fetch_score_and_store(query)

    filtered_data = filter_data(data)
    if filtered_data.empty:
        st.warning("No records match the selected filters.")
        return

    top_n = int(st.session_state.get("top_n", 12))
    keywords = detect_trending_keywords(filtered_data["clean_text"], top_n=top_n)
    sentiment_counts = filtered_data["sentiment"].value_counts()
    positive = int(sentiment_counts.get("Positive", 0))
    negative = int(sentiment_counts.get("Negative", 0))
    neutral = int(sentiment_counts.get("Neutral", 0))

    st.markdown(
        f"""
        <div class="dashboard-header">
            <div>
                <h1 class="dashboard-title">Social Media Trend Monitoring Dashboard</h1>
                <p class="dashboard-subtitle">Real-time AI insights from NewsAPI, GNews, Reddit, RSS, Twitter/X, and local fallback data</p>
            </div>
            <div class="updated">Last Updated: {last_updated}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    view = st.session_state.get("dashboard_view", "Dashboard Home")

    if view == "Dashboard Home":
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            kpi_card("Total News", f"{len(filtered_data):,}", f"{inserted_rows:,} DB updates", "blue")
        with c2:
            kpi_card("Positive", f"{positive:,}", f"{positive / len(filtered_data) * 100:.1f}% of results", "green")
        with c3:
            kpi_card("Negative", f"{negative:,}", f"{negative / len(filtered_data) * 100:.1f}% of results", "red")
        with c4:
            kpi_card("Neutral", f"{neutral:,}", f"{neutral / len(filtered_data) * 100:.1f}% of results", "gray")
        with c5:
            kpi_card("Trending Topics", f"{len(keywords):,}", keywords.iloc[0]["keyword"] if not keywords.empty else "No keyword", "orange")
        with c6:
            kpi_card("Last Updated", last_updated.split(" ")[-2], f"Auto refresh every {FETCH_INTERVAL_SECONDS}s", "cyan")

        st.write("")
        # Lazy load chart functions
        from utils.charts import (
            hourly_activity,
            sentiment_distribution_bar,
            sentiment_line,
            sentiment_pie,
            top_keywords_bar,
            trend_chart,
        )
        from utils.wordcloud import build_wordcloud_image

        a, b, c = st.columns([0.95, 1.1, 1.1])
        with a:
            panel("Pie Chart")
            st.plotly_chart(sentiment_pie(filtered_data), width="stretch")
            end_panel()
        with b:
            panel("Top Keywords")
            st.plotly_chart(top_keywords_bar(keywords), width="stretch")
            end_panel()
        with c:
            panel("Line Chart")
            st.plotly_chart(sentiment_line(filtered_data), width="stretch")
            end_panel()

        d, e, f = st.columns(3)
        with d:
            panel("Trend Chart")
            st.plotly_chart(trend_chart(filtered_data), width="stretch")
            end_panel()
        with e:
            panel("Hourly Activity")
            st.plotly_chart(hourly_activity(filtered_data), width="stretch")
            end_panel()
        with f:
            panel("Sentiment Distribution")
            st.plotly_chart(sentiment_distribution_bar(filtered_data), width="stretch")
            end_panel()

        panel("Word Cloud")
        st.image(build_wordcloud_image(filtered_data["clean_text"].tolist()), width="stretch")
        end_panel()

        topic_result = get_topic_clusters(filtered_data["clean_text"])
        briefing_text = get_ai_briefing(
            keywords,
            {"Positive": positive, "Negative": negative, "Neutral": neutral},
            topic_result.topics_df,
            len(filtered_data),
        )
        st.session_state["latest_briefing"] = briefing_text
        st.session_state["latest_topics"] = topic_result

        g, h = st.columns([1.2, 1])
        with g:
            panel("AI Trend Briefing (Summary)")
            st.markdown(f"<div class='ai-briefing-text'>{briefing_text}</div>", unsafe_allow_html=True)
            end_panel()
        with h:
            panel("Topic Clusters (BERTopic)")
            if topic_result.available:
                st.caption(f"Approx. topic coherence score: {topic_result.coherence_score:.2f} (target >= 0.55)")
                st.dataframe(topic_result.topics_df, width="stretch", hide_index=True, height=220)
            else:
                st.info(topic_result.message or "Not enough data yet for topic modeling.")
            end_panel()

    else:
        if view == "Analytics View":
            from utils.charts import trend_chart, hourly_activity
            c1, c2 = st.columns([1.1, 1])
            with c1:
                panel("Trend Chart")
                st.plotly_chart(trend_chart(filtered_data), width="stretch")
                end_panel()
            with c2:
                panel("Hourly Activity")
                st.plotly_chart(hourly_activity(filtered_data), width="stretch")
                end_panel()
        else:
            db = get_database()
            st.subheader("Database Status")
            if st.button("Create / Verify Tables"):
                st.success("Database tables are ready.") if db.create_tables() else st.error("Database is unavailable.")
            st.json(db.counts())

    panel("Recent Articles")
    display_table = table_view(filtered_data.sort_values("published_at", ascending=False), trend=view == "Analytics View")
    st.dataframe(display_table.head(100), width="stretch", hide_index=True, height=360)
    end_panel()

    briefing_for_report = st.session_state.get("latest_briefing", "")
    topics_for_report = st.session_state.get("latest_topics")
    import pandas as pd
    topics_df_for_report = topics_for_report.topics_df if topics_for_report is not None else pd.DataFrame()

    st.subheader("Downloads")
    download_cols = st.columns(4)
    with download_cols[0]:
        st.download_button(
            "Download CSV",
            display_table.to_csv(index=False).encode("utf-8"),
            file_name="trend_dashboard.csv",
            mime="text/csv",
            width="stretch",
        )
    with download_cols[1]:
        st.download_button(
            "Download Excel",
            to_excel(display_table),
            file_name="trend_dashboard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with download_cols[2]:
        st.download_button(
            "Download PDF",
            to_pdf(filtered_data, keywords, briefing_for_report, topics_df_for_report),
            file_name="trend_dashboard_report.pdf",
            mime="application/pdf",
            width="stretch",
        )
    with download_cols[3]:
        st.download_button(
            "Download AI Briefing (PDF)",
            to_pdf(filtered_data, keywords, briefing_for_report, topics_df_for_report),
            file_name="ai_trend_briefing.pdf",
            mime="application/pdf",
            width="stretch",
            disabled=not briefing_for_report,
        )


render_dashboard(query)
