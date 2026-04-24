from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

    
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    load_metadata,
    load_feature_importance,
    load_drift_report,
    load_drift_summary,
    load_batch_monitoring,
    load_production,
    load_scored_holdout,
)
from src.predict import score_record

st.set_page_config(
    page_title="MLOps Control Tower",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #07111f;
                --panel: rgba(8, 15, 29, 0.82);
                --panel-2: rgba(14, 24, 43, 0.82);
                --line: rgba(148, 163, 184, 0.14);
                --text: #ecf3ff;
                --muted: #90a4c2;
                --cyan: #58e1ff;
                --blue: #7aa2ff;
                --violet: #a98fff;
                --green: #3ee7a6;
                --amber: #ffbe5c;
                --red: #ff6b7c;
            }
            .stApp {
                background:
                    radial-gradient(circle at 0% 0%, rgba(88,225,255,.12), transparent 28%),
                    radial-gradient(circle at 100% 0%, rgba(122,162,255,.12), transparent 34%),
                    linear-gradient(180deg, #050d18 0%, #07111f 60%, #081321 100%);
                color: var(--text);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(7,14,26,.96) 0%, rgba(8,17,31,.98) 100%);
                border-right: 1px solid rgba(255,255,255,0.06);
            }
            [data-testid="stSidebar"] * {
                color: #e9f1fb !important;
            }
            .block-container {
                padding-top: 1.15rem;
                padding-bottom: 2rem;
                max-width: 1440px;
            }
            .top-shell {
                position: relative;
                padding: 1.45rem 1.55rem 1.35rem 1.55rem;
                border: 1px solid rgba(148,163,184,.14);
                border-radius: 26px;
                overflow: hidden;
                background:
                    linear-gradient(135deg, rgba(14,24,43,.85), rgba(9,18,32,.88)),
                    linear-gradient(180deg, rgba(88,225,255,.14), rgba(122,162,255,.10));
                box-shadow: 0 18px 60px rgba(0,0,0,.32);
            }
            .top-shell:before {
                content: "";
                position: absolute;
                inset: 0;
                background:
                    radial-gradient(circle at 12% 15%, rgba(88,225,255,.16), transparent 24%),
                    radial-gradient(circle at 88% 18%, rgba(169,143,255,.13), transparent 24%);
                pointer-events: none;
            }
            .eyebrow {
                letter-spacing: .16em;
                text-transform: uppercase;
                font-size: .76rem;
                color: var(--cyan);
                font-weight: 700;
                margin-bottom: .45rem;
            }
            .hero-title {
                font-size: 2.35rem;
                line-height: 1.05;
                font-weight: 800;
                margin-bottom: .45rem;
                letter-spacing: -.03em;
            }
            .muted {
                color: var(--muted);
                font-size: .98rem;
            }
            .grid-note {
                margin-top: .7rem;
                display: flex;
                gap: .55rem;
                flex-wrap: wrap;
            }
            .status-pill {
                padding: .4rem .68rem;
                border-radius: 999px;
                border: 1px solid rgba(148,163,184,.12);
                font-size: .78rem;
                background: rgba(255,255,255,.04);
                color: #dce8ff;
                display: inline-block;
            }
            .glass-card {
                padding: 1rem 1.05rem;
                border-radius: 22px;
                border: 1px solid var(--line);
                background: linear-gradient(180deg, rgba(11,21,38,.88), rgba(8,16,30,.88));
                box-shadow: 0 12px 32px rgba(0,0,0,.22);
            }
            .metric-card {
                min-height: 148px;
                padding: 1rem 1.05rem;
                border-radius: 24px;
                background: linear-gradient(180deg, rgba(15,26,45,.92), rgba(8,15,29,.9));
                border: 1px solid rgba(148,163,184,.12);
                box-shadow: inset 0 1px 0 rgba(255,255,255,.03), 0 16px 36px rgba(0,0,0,.25);
            }
            .metric-label {
                font-size: .78rem;
                letter-spacing: .12em;
                text-transform: uppercase;
                color: var(--cyan);
                font-weight: 700;
                margin-bottom: .55rem;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 800;
                line-height: 1;
                margin-bottom: .3rem;
            }
            .metric-sub {
                color: var(--muted);
                font-size: .88rem;
            }
            .panel-title {
                font-size: 1.08rem;
                font-weight: 700;
                margin-bottom: .1rem;
            }
            .panel-kicker {
                font-size: .74rem;
                color: var(--cyan);
                text-transform: uppercase;
                letter-spacing: .12em;
                margin-bottom: .42rem;
                font-weight: 700;
            }
            .spot-card {
                padding: 1rem 1.05rem;
                border-radius: 22px;
                border: 1px solid rgba(148,163,184,.12);
                background: linear-gradient(135deg, rgba(8,17,31,.95), rgba(14,24,43,.9));
            }
            .list-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: .5rem 0;
                border-bottom: 1px dashed rgba(148,163,184,.12);
            }
            .list-row:last-child { border-bottom: none; }
            .tiny {
                color: var(--muted);
                font-size: .82rem;
            }
            .callout {
                padding: .9rem 1rem;
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(88,225,255,.09), rgba(122,162,255,.08));
                border: 1px solid rgba(88,225,255,.16);
            }
            .footer-note {
                padding: .85rem 1rem;
                border-radius: 16px;
                border: 1px solid rgba(148,163,184,.12);
                background: rgba(255,255,255,.03);
                color: #cfe0f9;
            }
            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(15,26,45,.92), rgba(8,15,29,.9));
                border: 1px solid rgba(148,163,184,.12);
                padding: .85rem 1rem;
                border-radius: 18px;
            }
            .stDataFrame, .stTable {
                border: 1px solid rgba(148,163,184,.10);
                border-radius: 16px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

meta = load_metadata()
feature_imp = load_feature_importance().head(18)
drift = load_drift_report()
drift_summary = load_drift_summary()
batches = load_batch_monitoring()
production = load_production()
holdout = load_scored_holdout()

batches["batch_date"] = pd.to_datetime(batches["batch_date"])
production["batch_date"] = pd.to_datetime(production["batch_date"])


def pill(text: str) -> str:
    return f'<span class="status-pill">{text}</span>'


def hero(title: str, subtitle: str, pills: list[str]) -> None:
    st.markdown(
        f"""
        <div class="top-shell">
            <div class="eyebrow">Production-grade ML engineering portfolio</div>
            <div class="hero-title">{title}</div>
            <div class="muted">{subtitle}</div>
            <div class="grid-note">{''.join(pill(p) for p in pills)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_open(kicker: str, title: str, body: str = "") -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="panel-kicker">{kicker}</div>
            <div class="panel-title">{title}</div>
            {f'<div class="tiny" style="margin-top:.25rem;">{body}</div>' if body else ''}
        """,
        unsafe_allow_html=True,
    )


def section_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def gauge(value: float, title: str, max_value: float = 1.0, color: str = "#58e1ff") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"size": 30}},
            title={"text": title},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.02)",
                "bordercolor": "rgba(255,255,255,0.08)",
                "steps": [
                    {"range": [0, max_value * 0.4], "color": "rgba(62,231,166,0.16)"},
                    {"range": [max_value * 0.4, max_value * 0.7], "color": "rgba(255,190,92,0.16)"},
                    {"range": [max_value * 0.7, max_value], "color": "rgba(255,107,124,0.16)"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


risk_rate = (production["priority_band"].isin(["High", "Critical"]).mean()) * 100
critical_rate = (production["priority_band"] == "Critical").mean() * 100
avg_sla = production["sla_hours_remaining"].mean()
queue_size = len(production)
latest_batch = batches.sort_values("batch_date").iloc[-1]
alert_mix = production["priority_band"].value_counts().rename_axis("priority_band").reset_index(name="alerts")
analyst_load = production["analyst_owner"].value_counts().reset_index()
analyst_load.columns = ["analyst_owner", "alerts"]
holdout_perf = holdout.copy()
holdout_perf["bucket"] = pd.qcut(holdout_perf["score"], q=10, duplicates="drop")
calibration = holdout_perf.groupby("bucket", observed=False).agg(avg_score=("score", "mean"), actual_rate=("actual_churn", "mean")).reset_index()

st.sidebar.markdown("## 🛰️ MLOps Control Tower")
page = st.sidebar.radio(
    "Navigate",
    ["Control Tower", "Inference Studio", "Observability Grid", "Release Ops", "Client Story"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Fleet status")
st.sidebar.markdown(
    f"""
    <div class="spot-card">
        <div class="list-row"><span>Model version</span><strong>{meta['version']}</strong></div>
        <div class="list-row"><span>Algorithm</span><strong>LogReg Pipeline</strong></div>
        <div class="list-row"><span>ROC-AUC</span><strong>{meta['metrics']['roc_auc']:.3f}</strong></div>
        <div class="list-row"><span>Drifted features</span><strong>{drift_summary['drifted_features']}</strong></div>
        <div class="list-row"><span>Latest batch</span><strong>{pd.to_datetime(drift_summary['evaluated_batch']).date()}</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(" ")
st.sidebar.caption("Docker • CI • API • Monitoring • Drift • Streamlit")

if page == "Control Tower":
    hero(
        "Production-Ready End-to-End MLOps Control Tower",
        "A premium command center that connects training, inference, deployment readiness, monitoring, and retraining signals in one client-facing interface.",
        [
            "Dockerized service",
            "CI-tested scoring API",
            "PSI drift monitoring",
            "Analyst queue visibility",
        ],
    )
    st.write("")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Model quality", f"{meta['metrics']['roc_auc']:.3f}", "Holdout ROC-AUC")
    with m2:
        metric_card("High-priority exposure", f"{risk_rate:.1f}%", "High + critical cases")
    with m3:
        metric_card("Critical escalation", f"{critical_rate:.1f}%", "Immediate analyst attention")
    with m4:
        metric_card("Queue pressure", f"{queue_size}", f"Avg SLA left {avg_sla:.1f}h")

    a, b, c = st.columns([1.15, 0.85, 0.8])
    with a:
        section_open("Live posture", "Weekly score pressure + alert escalation", "Combines average model risk with critical alert count across production windows.")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=batches["batch_date"],
                y=batches["avg_score"],
                mode="lines+markers",
                name="Avg risk score",
                line=dict(width=3, color="#58e1ff"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=batches["batch_date"],
                y=batches["critical_alerts"],
                name="Critical alerts",
                opacity=0.35,
                marker=dict(color="#a98fff"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=12, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Avg score"),
            yaxis2=dict(title="Critical alerts", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)
        section_close()

    with b:
        section_open("Priority distribution", "Current queue mix", "A quick health read on how much operational load sits in critical vs stable bands.")
        donut = px.pie(
            alert_mix,
            names="priority_band",
            values="alerts",
            hole=0.62,
            template="plotly_dark",
        )
        donut.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(donut, use_container_width=True)
        section_close()

    with c:
        section_open("Ops readiness", "Command status board")
        st.markdown(
            f"""
            <div class="callout">
                <div style="font-weight:700; font-size:1rem; margin-bottom:.35rem;">Deployment posture: ready</div>
                <div class="tiny">Model artifact, API, dashboard, Docker assets, and CI workflow are all present in the repo.</div>
            </div>
            <div style="height:.7rem"></div>
            <div class="spot-card">
                <div class="list-row"><span>Average precision</span><strong>{meta['metrics']['average_precision']:.3f}</strong></div>
                <div class="list-row"><span>Threshold</span><strong>{meta['threshold']}</strong></div>
                <div class="list-row"><span>Training rows</span><strong>{meta['training_rows']:,}</strong></div>
                <div class="list-row"><span>Test rows</span><strong>{meta['test_rows']:,}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        section_close()

    st.write("")
    left, right = st.columns([1.1, 0.9])
    with left:
        section_open("Decision support", "Top model risk drivers", "These coefficients make the business story clear when explaining why the model flags churn exposure.")
        topf = feature_imp.copy().sort_values("coefficient")
        feat_fig = px.bar(topf.tail(14), x="coefficient", y="feature", orientation="h", template="plotly_dark")
        feat_fig.update_layout(height=380, margin=dict(l=10, r=10, t=12, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(feat_fig, use_container_width=True)
        section_close()

    with right:
        section_open("Execution load", "Analyst queue allocation", "Shows whether case distribution is balanced or whether one analyst is overloaded.")
        analyst_fig = px.bar(analyst_load, x="analyst_owner", y="alerts", template="plotly_dark")
        analyst_fig.update_layout(height=380, margin=dict(l=10, r=10, t=12, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(analyst_fig, use_container_width=True)
        section_close()

    st.write("")
    section_open("Queue workbench", "Top active cases", "This table gives buyers an immediate feel for how predictions flow into an operations queue.")
    queue_cols = [
        "event_id",
        "batch_date",
        "score",
        "priority_band",
        "case_status",
        "analyst_owner",
        "sla_hours_remaining",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "tenure",
    ]
    queue = production.sort_values("score", ascending=False)[queue_cols].head(25).copy()
    queue["score"] = queue["score"].round(3)
    queue["batch_date"] = queue["batch_date"].dt.strftime("%Y-%m-%d")
    st.dataframe(queue, use_container_width=True, hide_index=True)
    section_close()

elif page == "Inference Studio":
    hero(
        "Inference Studio",
        "Interactive single-record scoring that mimics how an exposed API evaluates one customer in production and routes the outcome into an operational priority band.",
        ["Real scoring path", "Priority routing", "Buyer-friendly demo", "No notebook clutter"],
    )
    st.write("")
    sample = production.iloc[0]

    lcol, rcol = st.columns([1.15, 0.85])
    with lcol:
        section_open("Payload builder", "Create a customer record", "Use realistic telco attributes and generate a production-style score.")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if sample["gender"] == "Female" else 1)
            senior = st.selectbox("Senior citizen", [0, 1], index=int(sample["SeniorCitizen"]))
            partner = st.selectbox("Partner", ["Yes", "No"], index=0 if sample["Partner"] == "Yes" else 1)
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if sample["Dependents"] == "Yes" else 1)
            tenure = st.slider("Tenure (months)", 0, 72, int(sample["tenure"]))
            phone = st.selectbox("Phone service", ["Yes", "No"], index=0 if sample["PhoneService"] == "Yes" else 1)
        with c2:
            mult = st.selectbox("Multiple lines", sorted(production["MultipleLines"].dropna().unique().tolist()), index=0)
            internet = st.selectbox("Internet service", sorted(production["InternetService"].dropna().unique().tolist()), index=0)
            online_sec = st.selectbox("Online security", sorted(production["OnlineSecurity"].dropna().unique().tolist()), index=0)
            backup = st.selectbox("Online backup", sorted(production["OnlineBackup"].dropna().unique().tolist()), index=0)
            device = st.selectbox("Device protection", sorted(production["DeviceProtection"].dropna().unique().tolist()), index=0)
            tech = st.selectbox("Tech support", sorted(production["TechSupport"].dropna().unique().tolist()), index=0)
        with c3:
            tv = st.selectbox("Streaming TV", sorted(production["StreamingTV"].dropna().unique().tolist()), index=0)
            movies = st.selectbox("Streaming movies", sorted(production["StreamingMovies"].dropna().unique().tolist()), index=0)
            contract = st.selectbox("Contract", sorted(production["Contract"].dropna().unique().tolist()), index=0)
            paperless = st.selectbox("Paperless billing", ["Yes", "No"], index=0 if sample["PaperlessBilling"] == "Yes" else 1)
            payment = st.selectbox("Payment method", sorted(production["PaymentMethod"].dropna().unique().tolist()), index=0)
            monthly = st.number_input("Monthly charges", min_value=0.0, value=float(sample["MonthlyCharges"]))
            total = st.number_input("Total charges", min_value=0.0, value=float(sample["TotalCharges"]))

        payload = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": mult,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total),
        }
        score_click = st.button("Score live customer", type="primary", use_container_width=True)
        section_close()

    with rcol:
        section_open("Reference profile", "Scoring notes", "A buyer can immediately understand what the service returns and how it would be consumed downstream.")
        st.markdown(
            f"""
            <div class="spot-card">
                <div class="list-row"><span>Output fields</span><strong>score / label / priority</strong></div>
                <div class="list-row"><span>Threshold policy</span><strong>{meta['threshold']}</strong></div>
                <div class="list-row"><span>Intended consumer</span><strong>API + dashboard</strong></div>
                <div class="list-row"><span>Routing objective</span><strong>Queue prioritization</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge(meta["metrics"]["roc_auc"], "Model quality gauge"), use_container_width=True)
        section_close()

    if score_click:
        result = score_record(payload)
        st.write("")
        a1, a2, a3 = st.columns(3)
        with a1:
            metric_card("Predicted score", f"{result['score']:.3f}", "Estimated churn probability")
        with a2:
            metric_card("Priority band", str(result["priority_band"]), "Operational queue routing")
        with a3:
            metric_card("Predicted label", "Churn risk" if result["predicted_label"] == 1 else "Stable", "Thresholded classification")

        b1, b2 = st.columns([0.9, 1.1])
        with b1:
            section_open("Payload preview", "API-ready request body")
            st.json(payload, expanded=False)
            section_close()
        with b2:
            section_open("Routing interpretation", "Why this matters for clients")
            st.markdown(
                f"""
                <div class="footer-note">
                    A score of <strong>{result['score']:.3f}</strong> puts this record into the <strong>{result['priority_band']}</strong> band. In a real production setup, that can trigger analyst review, CRM retention action, or a downstream workflow engine.
                </div>
                """,
                unsafe_allow_html=True,
            )
            section_close()

elif page == "Observability Grid":
    hero(
        "Observability Grid",
        "A monitoring view that helps clients understand post-deployment model health, queue behavior, and when retraining should be triggered.",
        ["PSI drift", "Queue health", "Calibration view", "Retrain signal"],
    )
    st.write("")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        metric_card("Stable features", str(drift_summary["stable_features"]), "PSI < 0.10")
    with d2:
        metric_card("Watch features", str(drift_summary["watch_features"]), "0.10 ≤ PSI < 0.25")
    with d3:
        metric_card("Drifted features", str(drift_summary["drifted_features"]), "PSI ≥ 0.25")
    with d4:
        metric_card("Latest batch score", f"{latest_batch['avg_score']:.3f}", "Current window average")

    row1a, row1b = st.columns([1.05, 0.95])
    with row1a:
        section_open("Distribution shift", "Feature drift severity grid", "PSI makes it easy to explain where production data is no longer behaving like the training reference set.")
        drift_fig = px.bar(
            drift.sort_values("psi", ascending=True),
            x="psi",
            y="feature",
            color="severity",
            orientation="h",
            template="plotly_dark",
        )
        drift_fig.update_layout(height=420, margin=dict(l=10, r=10, t=12, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(drift_fig, use_container_width=True)
        section_close()
    with row1b:
        section_open("Queue health", "Priority volume over time", "Shows whether the operations queue is trending toward higher severity windows.")
        queue_health = production.groupby(["batch_date", "priority_band"]).size().reset_index(name="alerts")
        queue_fig = px.area(queue_health, x="batch_date", y="alerts", color="priority_band", template="plotly_dark")
        queue_fig.update_layout(height=420, margin=dict(l=10, r=10, t=12, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(queue_fig, use_container_width=True)
        section_close()

    row2a, row2b = st.columns([0.95, 1.05])
    with row2a:
        section_open("Calibration lens", "Predicted vs actual risk", "This view tells a more mature MLOps story than raw accuracy alone.")
        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(x=calibration["avg_score"], y=calibration["actual_rate"], mode="lines+markers", name="Observed"))
        cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(dash="dash")))
        cal_fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=12, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Predicted score", yaxis_title="Observed churn")
        st.plotly_chart(cal_fig, use_container_width=True)
        section_close()
    with row2b:
        section_open("Intervention signal", "Retraining recommendation board")
        retrain = "Recommended" if drift_summary["drifted_features"] >= 1 else "Monitor only"
        urgency = "High" if drift_summary["drifted_features"] >= 2 else "Moderate"
        st.markdown(
            f"""
            <div class="spot-card">
                <div class="list-row"><span>Retraining decision</span><strong>{retrain}</strong></div>
                <div class="list-row"><span>Urgency</span><strong>{urgency}</strong></div>
                <div class="list-row"><span>Watchlist features</span><strong>{drift_summary['watch_features']}</strong></div>
                <div class="list-row"><span>Drifted features</span><strong>{drift_summary['drifted_features']}</strong></div>
            </div>
            <div style="height:.9rem"></div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge(float(min(1.0, drift_summary["drifted_features"] / 3)), "Retrain pressure", max_value=1.0, color="#ffbe5c"), use_container_width=True)
        section_close()

    section_open("Drift ledger", "Detailed PSI table")
    drift_view = drift.copy()
    drift_view["psi"] = drift_view["psi"].round(3)
    st.dataframe(drift_view, use_container_width=True, hide_index=True)
    section_close()

elif page == "Release Ops":
    hero(
        "Release Ops",
        "A client-friendly deployment page that proves the model is packaged, reproducible, testable, and ready for a real engineering handoff.",
        ["Docker", "CI workflow", "FastAPI", "Reproducible training"],
    )
    st.write("")

    l1, l2 = st.columns([0.9, 1.1])
    with l1:
        section_open("Registry snapshot", "Model card")
        registry = pd.DataFrame(
            [
                {"Field": "Model", "Value": meta["model_name"]},
                {"Field": "Version", "Value": meta["version"]},
                {"Field": "Algorithm", "Value": meta["algorithm"]},
                {"Field": "Dataset", "Value": meta["dataset"]},
                {"Field": "Average precision", "Value": f"{meta['metrics']['average_precision']:.3f}"},
                {"Field": "F1", "Value": f"{meta['metrics']['f1']:.3f}"},
            ]
        )
        st.table(registry)
        section_close()
    with l2:
        section_open("Promotion pipeline", "From raw data to monitored service")
        st.markdown(
            """
            1. **Ingest** public churn dataset and validate schema  
            2. **Train** preprocessing + model pipeline  
            3. **Persist** artifact and metadata for registry usage  
            4. **Expose** prediction through FastAPI  
            5. **Containerize** dashboard + API with Docker assets  
            6. **Test** endpoints in CI before promotion  
            7. **Monitor** PSI drift and queue health after deployment  
            8. **Retrain** when drift or business KPIs cross thresholds
            """
        )
        section_close()

    x1, x2, x3 = st.columns(3)
    with x1:
        metric_card("API status", "Ready", "FastAPI service included")
    with x2:
        metric_card("Container assets", "Included", "Dockerfile + compose")
    with x3:
        metric_card("CI checks", "Included", "Pytest smoke workflow")

    section_open("Command deck", "Useful run commands")
    st.code(
        "docker compose up --build\n"
        "python -m uvicorn api.main:app --reload\n"
        "python -m streamlit run app/streamlit_app.py\n"
        "pytest -q\n"
        "python -m src.train\n"
        "python -m src.simulate_production",
        language="bash",
    )
    section_close()

elif page == "Client Story":
    hero(
        "Client Story",
        "This page converts the repo into a sales conversation: why the architecture matters, what problem it solves, and why a serious client should trust it.",
        ["ML engineer framing", "Buyer language", "Reusable pattern", "Portfolio-ready narrative"],
    )
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        section_open("Why clients care", "What this proves")
        st.markdown(
            """
            - The model is **not trapped in a notebook**.  
            - There is a clear path from **training to deployment**.  
            - The system includes **monitoring after launch**, not just pre-launch metrics.  
            - The architecture can be adapted to **fraud, demand forecasting, lead scoring, and ticket triage**.
            """
        )
        section_close()
    with c2:
        section_open("How to pitch it", "One-line value proposition")
        st.markdown(
            """
            <div class="footer-note">
                I built a production-style MLOps pipeline that trains a real-world churn model, serves predictions through an API, monitors post-deployment drift, and provides an executive dashboard for operational visibility.
            </div>
            """,
            unsafe_allow_html=True,
        )
        section_close()

    section_open("Buyer-facing file map", "What to show during a demo")
    st.markdown(
        """
        - `app/streamlit_app.py` → premium executive dashboard  
        - `api/main.py` → deployable prediction service  
        - `src/train.py` → reproducible training workflow  
        - `src/drift.py` + `reports/` → monitoring + drift evidence  
        - `.github/workflows/ci.yml` → CI confidence signal  
        - `Dockerfile` + `docker-compose.yml` → handoff-ready deployment assets
        """
    )
    section_close()
