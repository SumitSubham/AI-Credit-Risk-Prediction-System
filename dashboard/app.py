
import sys
import json
import csv
import html
from io import BytesIO, StringIO
from pathlib import Path
from datetime import datetime

import streamlit as st

# --------------------------------------------------
# 1. Project setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(PROJECT_ROOT / "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from predict import predict_credit_risk

st.set_page_config(
    page_title="AI Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
)

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 22px 0;">
            <h2 style="color: #1e3a8a; margin-bottom: 4px;">
                CreditRisk AI
            </h2>
            <p style="color: #64748b; font-size: 13px;">
                Intelligent Credit Assessment
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Credit Risk Prediction",
            "Model Performance",
            "About Project",
        ],
        index=0,
    )

    st.divider()

    st.caption("Model: XGBoost")
    st.caption("Decision threshold: 50%")

# --------------------------------------------------
# 2. Dashboard styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --navy: #102544;
        --blue: #2563eb;
        --ink: #18243a;
        --muted: #66758b;
        --line: #e3eaf3;
        --page: #f5f8fc;
    }
    .stApp { background: var(--page); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245,248,252,.92); }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }
    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(26,48,80,.05);
        min-height: 112px;
    }
    div[data-testid="stMetricLabel"] { color: #66758b; font-weight: 650; }
    div[data-testid="stMetricValue"] { color: #142640; font-weight: 780; }
    .hero {
        padding: 1.7rem 1.9rem;
        border-radius: 20px;
        color: white;
        background: linear-gradient(120deg, #102544 0%, #1d4e89 65%, #2563eb 100%);
        box-shadow: 0 12px 28px rgba(16,37,68,.13);
        margin-bottom: 1.1rem;
    }
    .hero-kicker {
        color: #bfdbfe; font-size: .78rem; letter-spacing: .14em;
        font-weight: 750; text-transform: uppercase; margin-bottom: .5rem;
    }
    .hero h1 { color: white; font-size: clamp(1.7rem, 3vw, 2.55rem);
        line-height: 1.15; margin: 0 0 .6rem 0; }
    .hero p { color: #e4edff; margin: 0; font-size: 1rem; }
    .section-kicker {
        color: #2563eb; font-size: .75rem; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .25rem;
    }
    .section-title { color: #142640; font-size: 1.45rem; font-weight: 760;
        margin: 0 0 .25rem 0; }
    .section-subtitle { color: #66758b; margin-bottom: 1rem; }
    .result-banner {
        border-radius: 14px; padding: 1rem 1.15rem; margin: .8rem 0 1rem 0;
        font-weight: 650; border: 1px solid;
    }
    .risk-low { color: #166534; background: #f0fdf4; border-color: #bbf7d0; }
    .risk-high { color: #9a3412; background: #fff7ed; border-color: #fed7aa; }
    .gauge-track {
        height: 13px; width: 100%; border-radius: 999px;
        background: linear-gradient(90deg, #22c55e 0%, #facc15 55%, #ef4444 100%);
        position: relative; margin: .7rem 0 .35rem 0;
    }
    .gauge-pointer {
        position: absolute; top: -5px; width: 4px; height: 23px;
        border-radius: 3px; background: #142640; border: 1px solid white;
        box-shadow: 0 1px 4px rgba(0,0,0,.28);
    }
    .gauge-labels { display: flex; justify-content: space-between;
        color: #758198; font-size: .78rem; gap: 1rem; }
    .small-note { color: #758198; font-size: .82rem; }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px; font-weight: 650; min-height: 2.7rem;
    }
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 10px; min-height: 3rem; font-weight: 750;
    }
    hr { border-color: var(--line); }
    @media (max-width: 760px) {
        .block-container { padding-top: 1rem; }
        .hero { padding: 1.2rem; border-radius: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. Helper: PDF report generation
# --------------------------------------------------

def create_pdf_report(customer, result):
    """Create a downloadable PDF assessment report."""

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=12,
    )

    story = [
        Paragraph("AI Credit Risk Assessment Report", title_style),
        Paragraph(
            "XGBoost-based machine-learning assessment",
            styles["Normal"],
        ),
        Spacer(1, 8 * mm),
        Paragraph("1. Assessment Summary", styles["Heading2"]),
        Paragraph(
            "Generated: "
            + datetime.now().astimezone().strftime("%d %b %Y, %I:%M %p %Z"),
            styles["Normal"],
        ),
        Spacer(1, 4 * mm),
    ]

    summary_rows = [
        ["Metric", "Result"],
        ["Default probability", f'{result["default_probability"]:.2f}%'],
        ["Classification", result["risk_class"]],
        ["Classification threshold", f'{result["threshold"]:.2f}'],
        ["Model", "XGBoost"],
    ]

    summary_table = Table(
        summary_rows,
        colWidths=[65 * mm, 95 * mm],
    )

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    story.extend([
        summary_table,
        Spacer(1, 8 * mm),
        Paragraph("2. Customer and Loan Inputs", styles["Heading2"]),
    ])

    customer_rows = [["Input", "Value"]]

    for field, value in customer.items():
        customer_rows.append([field, str(value)])

    customer_table = Table(
        customer_rows,
        colWidths=[65 * mm, 95 * mm],
        repeatRows=1,
    )

    customer_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    story.extend([
        customer_table,
        Spacer(1, 8 * mm),
        Paragraph("3. Interpretation and Limitations", styles["Heading2"]),
        Paragraph(
            "The default probability is the model's estimated probability "
            "for the supplied input profile. The classification is based on "
            "the saved model threshold. It is not a guarantee of repayment "
            "or default, and it is not a loan approval or rejection decision. "
            "This demonstration has not independently established suitability "
            "for real-world lending decisions.",
            styles["Normal"],
        ),
    ])

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# --------------------------------------------------
if page in ("Dashboard", "About Project"):
    # 4. Dashboard header
    # --------------------------------------------------

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">AI-powered decision support · Portfolio demo</div>
            <h1>🏦 Credit Risk Assessment</h1>
            <p>Estimate loan-default probability with a trained XGBoost classification model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Demonstration only. Predictions should not be used as the sole basis "
        "for approving or rejecting a loan."
    )

    with st.expander("About this model"):
        st.markdown(
            """
            - **Algorithm:** XGBoost
            - **Task:** Binary classification of loan default
            - **Threshold:** Loaded from the saved model artifact
            - **Output:** Estimated default probability and binary classification
            """
        )

    # --------------------------------------------------
if page == "About Project":
    st.markdown("## About this project")
    st.write(
        "This demonstration uses a trained XGBoost classification model to estimate "
        "loan-default probability from the applicant and loan details entered. It is "
        "a portfolio project and is not a production lending or approval system."
    )
    st.markdown("### Main features")
    st.markdown(
        "- Credit-risk probability and threshold-based classification\n"
        "- Downloadable PDF, CSV, and JSON assessment reports\n"
        "- Previously recorded test-set performance metrics"
    )
    st.warning("Do not use these predictions as the sole basis for real lending decisions.")

if page in ("Dashboard", "Credit Risk Prediction"):
    # 5. Customer input form
    # --------------------------------------------------

    st.markdown(
        '<div class="section-kicker">Applicant profile</div>'
        '<div class="section-title">Customer and loan information</div>'
        '<div class="section-subtitle">Enter the profile details, then run the assessment.</div>',
        unsafe_allow_html=True,
    )

    with st.form("credit_risk_form"):

        col1, col2 = st.columns(2)

        with col1:
            person_age = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
            )

            person_income = st.number_input(
                "Annual Income",
                min_value=1,
                value=60000,
                step=5000,
            )

            person_home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "OWN", "MORTGAGE", "OTHER"],
            )

            person_emp_length = st.number_input(
                "Employment Length (Years)",
                min_value=0,
                max_value=100,
                value=5,
                step=1,
            )

            cb_person_cred_hist_length = st.number_input(
                "Credit History Length (Years)",
                min_value=0,
                max_value=100,
                value=7,
                step=1,
            )

        with col2:
            loan_intent = st.selectbox(
                "Loan Purpose",
                [
                    "PERSONAL",
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "HOMEIMPROVEMENT",
                    "DEBTCONSOLIDATION",
                ],
            )

            loan_grade = st.selectbox(
                "Loan Grade",
                ["A", "B", "C", "D", "E", "F", "G"],
            )

            loan_amnt = st.number_input(
                "Loan Amount",
                min_value=1,
                value=10000,
                step=1000,
            )

            loan_int_rate = st.number_input(
                "Loan Interest Rate (%)",
                min_value=0.1,
                max_value=100.0,
                value=11.5,
                step=0.5,
            )

            cb_person_default_on_file = st.selectbox(
                "Historical Default on File",
                ["N", "Y"],
            )

        submitted = st.form_submit_button(
            "Predict Credit Risk",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------
    # 6. Generate and retain prediction
    # --------------------------------------------------

    if submitted:
        try:
            loan_percent_income = loan_amnt / person_income

            customer_data = {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": person_home_ownership,
                "person_emp_length": person_emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_default_on_file": cb_person_default_on_file,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
            }

            result = predict_credit_risk(customer_data)

            # Keep results available when download buttons are clicked.
            st.session_state["credit_risk_customer"] = customer_data
            st.session_state["credit_risk_result"] = result

        except (ValueError, TypeError, KeyError) as error:
            st.error(f"Unable to generate prediction: {error}")

    
        except Exception as error:
            st.error(
                "Prediction failed. Check the saved model, input schema, "
                "and terminal error."
            )
            st.exception(error)


    # --------------------------------------------------
    # 7. Display saved results and exports
    # --------------------------------------------------

    if "credit_risk_result" in st.session_state:

        customer = st.session_state["credit_risk_customer"]
        result = st.session_state["credit_risk_result"]

        probability = result["default_probability"]

        st.divider()
        st.markdown(
            '<div class="section-kicker">Assessment output</div>'
            '<div class="section-title">Prediction results</div>'
            '<div class="section-subtitle">A model estimate for the submitted profile.</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Default Probability", f"{probability:.2f}%")

        with col2:
            st.metric("Classification", result["risk_class"])

        with col3:
            st.metric("Model Threshold", f'{result["threshold"]:.2f}')

        pointer = min(max(float(probability), 0.0), 100.0)
        threshold = float(result["threshold"])
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3eaf3;border-radius:16px;
                        padding:1.15rem 1.25rem;margin:.8rem 0;box-shadow:0 3px 12px rgba(26,48,80,.035);">
                <div style="font-weight:750;color:#142640;">Default probability indicator</div>
                <div class="gauge-track"><div class="gauge-pointer" style="left:calc({pointer}% - 2px);"></div></div>
                <div class="gauge-labels"><span>0% · Lower estimated probability</span><span>100% · Higher estimated probability</span></div>
                <div class="small-note" style="margin-top:.65rem;">
                    Classification threshold: {threshold:.2f} ({threshold*100:.0f}%).
                    Probability is an estimate, not a certainty.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["prediction"] == 1:
            st.markdown(
                '<div class="result-banner risk-high">⚠️ The model flagged this profile for elevated default risk at the selected threshold.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-banner risk-low">✓ The model did not flag this profile at the selected threshold.</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown(
            '<div class="section-kicker">Export center</div>'
            '<div class="section-title">Download and share your report</div>'
            '<div class="section-subtitle">Export the current assessment for review. Share files only with authorized recipients.</div>',
            unsafe_allow_html=True,
        )

        # Prepare exports
        pdf_bytes = create_pdf_report(customer, result)

        report_json = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "model": "XGBoost",
            "customer_inputs": customer,
            "prediction": result,
            "disclaimer": (
                "Demonstration only. Not a lending decision."
            ),
        }

        json_bytes = json.dumps(
            report_json,
            indent=4,
            allow_nan=False,
        ).encode("utf-8")

        csv_buffer = StringIO()
        csv_writer = csv.writer(csv_buffer)

        csv_writer.writerow(["Section", "Field", "Value"])

        for field, value in customer.items():
            csv_writer.writerow(["Customer Input", field, value])

        for field, value in result.items():
            csv_writer.writerow(["Prediction", field, value])

        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        # Print-friendly HTML report
        printable_rows = "".join(
            "<tr><td>"
            + html.escape(str(field))
            + "</td><td>"
            + html.escape(str(value))
            + "</td></tr>"
            for field, value in customer.items()
        )

        printable_html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>Credit Risk Assessment Report</title>
    <style>
    body {{ font-family: Arial, sans-serif; max-width: 850px;
           margin: 35px auto; padding: 20px; color: #222; }}
    h1, h2 {{ color: #17365D; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 9px; text-align: left; }}
    th {{ background: #17365D; color: white; }}
    @media print {{
        body {{ margin: 0; padding: 10px; }}
        .no-print {{ display: none; }}
    }}
    </style>
    </head>
    <body>
    <button class="no-print" onclick="window.print()">Print / Save as PDF</button>
    <h1>AI Credit Risk Assessment Report</h1>
    <p>Generated: {html.escape(datetime.now().astimezone().strftime("%d %b %Y, %I:%M %p %Z"))}</p>
    <h2>Prediction Summary</h2>
    <table>
    <tr><th>Metric</th><th>Result</th></tr>
    <tr><td>Default Probability</td><td>{probability:.2f}%</td></tr>
    <tr><td>Classification</td><td>{html.escape(result["risk_class"])}</td></tr>
    <tr><td>Threshold</td><td>{result["threshold"]:.2f}</td></tr>
    <tr><td>Model</td><td>XGBoost</td></tr>
    </table>
    <h2>Customer and Loan Inputs</h2>
    <table>
    <tr><th>Input</th><th>Value</th></tr>
    {printable_rows}
    </table>
    <h2>Important Disclaimer</h2>
    <p>This machine-learning estimate is not a loan approval or rejection
    decision. Do not use it as the sole basis for lending decisions.</p>
    </body>
    </html>"""

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name="credit_risk_assessment.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.download_button(
                "📊 Download CSV",
                data=csv_bytes,
                file_name="credit_risk_assessment.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col2:
            st.download_button(
                "🧾 Download JSON",
                data=json_bytes,
                file_name="credit_risk_assessment.json",
                mime="application/json",
                use_container_width=True,
            )

            st.download_button(
                "🖨️ Download Printable Report",
                data=printable_html.encode("utf-8"),
                file_name="credit_risk_print_report.html",
                mime="text/html",
                use_container_width=True,
            )

        st.caption(
            "To print: download the printable HTML file, open it in your "
            "browser, click Print / Save as PDF, and choose your printer "
            "or Save as PDF. Share downloaded reports only with authorized "
            "recipients."
        )

        with st.expander("View submitted customer inputs"):
            st.json(customer)

        st.caption(
            "This demonstration does not establish real-world lending "
            "validity, fairness, or regulatory compliance."
        )

    # --------------------------------------------------
if page in ("Dashboard", "Model Performance"):
    # 8. Model evaluation metrics
    # --------------------------------------------------

    st.divider()
    st.markdown(
        '<div class="section-kicker">Evaluation overview</div>'
        '<div class="section-title">Model performance — test set</div>'
        '<div class="section-subtitle">Previously recorded evaluation metrics for the trained model.</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "These are the previously measured evaluation results for your "
        "XGBoost model. They describe test-set performance, not the "
        "individual customer's prediction."
    )


    metric_cols = st.columns(5)

    evaluation_metrics = [
        ("Accuracy", 0.9397),
        ("Precision", 0.9795),
        ("Recall", 0.7398),
        ("F1 Score", 0.8429),
        ("ROC-AUC", 0.9510),
    ]

    for column, (name, value) in zip(metric_cols, evaluation_metrics):
        column.metric(name, f"{value:.2%}")

    st.markdown("#### Performance comparison")

    import pandas as pd

    metrics_df = pd.DataFrame(
        evaluation_metrics,
        columns=["Metric", "Score"],
    )

    st.bar_chart(
        metrics_df,
        x="Metric",
        y="Score",
        horizontal=True,
    )

    st.caption(
        "These are previously recorded test-set metrics. "
        "Precision, recall, F1 score, and ROC-AUC measure different "
        "aspects of performance. Recalculate them if the model or "
        "test dataset changes."
    )
