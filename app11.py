"""
Centum Electronics — Working Capital Decision Support Dashboard
=================================================================
Methodology (per dissertation):
  - Historical KPIs (DSO, DIO, DPO, CCC) are CALCULATED from uploaded
    Revenue / COGS / Inventory / Receivables / Payables data.
  - Only REVENUE is a manual forecast input (entered by the user).
  - COGS, Receivables, Inventory and Payables are AUTO-FORECASTED using
    driver-based linear regressions fitted on historical data:
        COGS         = f(Revenue)
        Receivables  = f(Revenue)
        Inventory    = f(COGS)
        Payables     = f(COGS)
  - DSO, DIO, DPO, CCC are then derived automatically from these.
  - Everything is benchmarked against a weighted industry benchmark,
    and DPO is treated as a funding-risk indicator (not a strength) once
    it exceeds standard trade terms (default 90 days).
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
# =========================================================
# THEME / CONSTANTS
# =========================================================
NAVY = "#132A4C"
STEEL = "#3D5A80"
GOLD = "#D69A2D"
GREEN = "#2E7D4F"
RED = "#B5432B"
AMBER = "#B5842B"
ICE = "#E9EEF6"
GREY = "#5B6B82"

COVID_YEARS = {"FY2020-21", "FY2021-22"}

REQUIRED_COLS = ["Year", "Revenue", "COGS", "Inventory", "Receivables", "Payables"]

st.set_page_config(
    page_title="Centum WC Decision Support Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Global CSS for KPI cards
# ---------------------------------------------------------
st.markdown(f"""
<style>
.kpi-card {{
    border-radius: 10px;
    padding: 16px 18px;
    color: white;
    margin-bottom: 10px;
}}
.kpi-title {{ font-size: 13px; font-weight: 600; opacity: 0.9; }}
.kpi-value {{ font-size: 30px; font-weight: 800; margin: 4px 0; }}
.kpi-sub   {{ font-size: 12px; opacity: 0.85; }}
.status-pill {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 11.5px; font-weight: 700; margin-top: 6px;
}}
</style>
""", unsafe_allow_html=True)


def kpi_card(title, value, sub, bg):
    st.markdown(f"""
    <div class="kpi-card" style="background:{bg};">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def status_pill(label, color):
    st.markdown(
        f'<span class="status-pill" style="background:{color}22;color:{color};">{label}</span>',
        unsafe_allow_html=True,
    )


# =========================================================
# SAMPLE / DEMO DATA  (used only if no file is uploaded)
# NOTE: illustrative only — replace with your real CCC_DATA2.xlsx export.
# =========================================================
def sample_data():
    years = ["FY2015-16", "FY2016-17", "FY2017-18", "FY2018-19", "FY2019-20",
              "FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"]
    rng = np.random.default_rng(7)
    revenue = np.array([3350, 3860, 3580, 5060, 4830, 6340, 7480, 9740, 9720], dtype=float)
    cogs = revenue * 0.66 + rng.normal(0, 40, len(revenue))
    inventory = cogs * 0.62 + rng.normal(0, 60, len(revenue))
    receivables = revenue * 0.33 + rng.normal(0, 60, len(revenue))
    payables = cogs * 0.34 + rng.normal(0, 40, len(revenue))
    return pd.DataFrame({
        "Year": years, "Revenue": revenue.round(2), "COGS": cogs.round(2),
        "Inventory": inventory.round(2), "Receivables": receivables.round(2),
        "Payables": payables.round(2),
    })


def make_template_csv():
    df = sample_data()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(show_spinner=False)
def load_uploaded(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. "
                          f"Expected columns: {REQUIRED_COLS}")
    return df[REQUIRED_COLS].copy()


def compute_kpis(df):
    df = df.copy()
    df["DSO"] = df["Receivables"] / df["Revenue"] * 360
    df["DIO"] = df["Inventory"] / df["COGS"] * 360
    df["DPO"] = df["Payables"] / df["COGS"] * 360
    df["CCC"] = df["DSO"] + df["DIO"] - df["DPO"]
    return df


# =========================================================
# DRIVER-BASED REGRESSION MODELS (fit on historical data)
# =========================================================
@st.cache_data(show_spinner=False)
def fit_models(df):
    def fit(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"slope": slope, "intercept": intercept, "r2": r2}

    models = {
        "cogs_on_revenue": fit(df["Revenue"].values, df["COGS"].values),
        "receivables_on_revenue": fit(df["Revenue"].values, df["Receivables"].values),
        "inventory_on_cogs": fit(df["COGS"].values, df["Inventory"].values),
        "payables_on_cogs": fit(df["COGS"].values, df["Payables"].values),
    }
    return models


def predict(m, x):
    return m["slope"] * x + m["intercept"]


def forecast_from_revenue(revenue, models):
    cogs = predict(models["cogs_on_revenue"], revenue)
    receivables = predict(models["receivables_on_revenue"], revenue)
    inventory = predict(models["inventory_on_cogs"], cogs)
    payables = predict(models["payables_on_cogs"], cogs)
    dso = receivables / revenue * 360
    dio = inventory / cogs * 360
    dpo = payables / cogs * 360
    ccc = dso + dio - dpo
    return {
        "Revenue": revenue, "COGS": cogs, "Inventory": inventory,
        "Receivables": receivables, "Payables": payables,
        "DSO": dso, "DIO": dio, "DPO": dpo, "CCC": ccc,
    }


# =========================================================
# STATUS / COLOR LOGIC
# =========================================================
def lower_is_better_status(value, benchmark):
    """For DSO, DIO, CCC — lower than benchmark is favorable."""
    if value <= benchmark:
        return "Better than Benchmark", GREEN
    gap_pct = (value - benchmark) / benchmark * 100
    if gap_pct <= 10:
        return "Slightly Above Benchmark", AMBER
    return "Improvement Required", RED


def dpo_status(value, standard_term, benchmark):
    """DPO is a funding-risk indicator once it exceeds standard trade terms —
    NOT simply 'higher is better'."""
    if value <= standard_term:
        return "Within Standard Trade Terms", GREEN
    if value <= benchmark:
        return "Above Standard Terms — Within Peer Norm", AMBER
    return "Overdue Payables Risk", RED

# =========================================================
# REVENUE FORECASTING ENGINE
# =========================================================

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def revenue_forecasting_engine(revenue_df, forecast_periods=3):

    y = revenue_df["Revenue"].values
    n = len(y)

    results = []

    # ==================================
    # 1. LINEAR TREND
    # ==================================
    x = np.arange(1, n + 1)

    slope, intercept = np.polyfit(x, y, 1)

    fitted = slope * x + intercept
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    mae = mean_absolute_error(y, fitted)
    rmse = np.sqrt(mean_squared_error(y, fitted))
    mape = np.mean(np.abs((y - fitted) / y)) * 100

    future_x = np.arange(n + 1, n + forecast_periods + 1)
    forecast = slope * future_x + intercept

    results.append({
    "Model": "Linear Trend",
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "R2": r2,
    "Forecast": forecast
    })


     # ==================================
    # 2. MOVING AVERAGE
    # ==================================

    window = 3

    actual = []
    fitted = []

    for i in range(window, n):
        fitted.append(np.mean(y[i-window:i]))
        actual.append(y[i])

    actual = np.array(actual)
    fitted = np.array(fitted)
    ss_res = np.sum((actual - fitted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    mae = mean_absolute_error(actual, fitted)
    rmse = np.sqrt(mean_squared_error(actual, fitted))
    mape = np.mean(np.abs((actual - fitted) / actual)) * 100

    forecast = []
    history = list(y.copy())

    for _ in range(forecast_periods):
        next_value = np.mean(history[-window:])
        forecast.append(next_value)
        history.append(next_value)

    forecast = np.array(forecast)

    results.append({
        "Model": "Moving Average",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Forecast": forecast
    })

    # ==================================
    # 3. SES
    # ==================================
    ses_model = SimpleExpSmoothing(y, initialization_method="estimated").fit()

    fitted = ses_model.fittedvalues
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    mae = mean_absolute_error(y, fitted)
    rmse = np.sqrt(mean_squared_error(y, fitted))
    mape = np.mean(np.abs((y - fitted) / y)) * 100

    forecast = ses_model.forecast(forecast_periods)

    results.append({
        "Model": "SES",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Forecast": forecast
    })

    # ==================================
    # 4. DES (HOLT)
    # ==================================
    
    des_model = Holt(y, initialization_method="estimated").fit()
  
    fitted = des_model.fittedvalues
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    mae = mean_absolute_error(y, fitted)
    rmse = np.sqrt(mean_squared_error(y, fitted))
    mape = np.mean(np.abs((y - fitted) / y)) * 100

    forecast = des_model.forecast(forecast_periods)

    results.append({
        "Model": "DES",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Forecast": forecast
    })

    metrics_df = pd.DataFrame([
    {
        "Model": r["Model"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"],
        "MAPE": r["MAPE"],
        "R²": r["R2"]
    }
    for r in results
    ])

    best_model = metrics_df.sort_values(
        by="MAPE"
    ).iloc[0]["Model"]

    best_forecast = np.array(
        next(
            r["Forecast"]
            for r in results
            if r["Model"] == best_model
        )
    ).flatten()
    return metrics_df, best_model, best_forecast

# =========================================================
# SIDEBAR — DATA, BENCHMARK, FORECAST INPUT
# =========================================================
st.sidebar.markdown(f"""
<div style="background:{NAVY};color:white;padding:10px 12px;border-radius:8px;
            text-align:center;font-weight:800;font-size:14px;letter-spacing:0.5px;margin-bottom:8px;">
    CENTUM WC&nbsp;DSS
</div>
""", unsafe_allow_html=True)

st.sidebar.header("1 · Historical Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload historical financials (.xlsx / .csv)", type=["xlsx", "csv"]
)
st.sidebar.download_button(
    "⬇ Download data template (CSV)", data=make_template_csv(),
    file_name="centum_wc_data_template.csv", mime="text/csv",
)
exclude_covid = st.sidebar.checkbox(
    "Exclude FY2020-21 & FY2021-22 (COVID years)", value=True
)

if uploaded_file is not None:
    try:
        raw_df = load_uploaded(uploaded_file)
        using_demo = False
    except Exception as e:
        st.sidebar.error(str(e))
        raw_df = sample_data()
        using_demo = True
else:
    raw_df = sample_data()
    using_demo = True

if exclude_covid:
    raw_df = raw_df[~raw_df["Year"].isin(COVID_YEARS)].reset_index(drop=True)

hist_df = compute_kpis(raw_df)
models = fit_models(hist_df)



st.sidebar.header("2 · Weighted Industry Benchmark")
bm_dso = st.sidebar.number_input("Benchmark DSO (days)", value=119.91, step=0.1)
bm_dio = st.sidebar.number_input("Benchmark DIO (days)", value=210.75, step=0.1)
bm_dpo = st.sidebar.number_input("Benchmark DPO (days)", value=99.73, step=0.1)
bm_ccc = bm_dso + bm_dio - bm_dpo
standard_term = st.sidebar.number_input("Standard Trade Term (days)", value=90.0, step=1.0)
# =========================================================
# REVENUE FORECAST
# =========================================================

st.sidebar.header("3 · Revenue Forecast")

forecast_mode = st.sidebar.radio(
    "Select Forecast Method",
    [
        "Automatic (Best Model)",
        "Manual Input"
    ]
)

if forecast_mode == "Automatic (Best Model)":

    revenue_df = hist_df[["Year", "Revenue"]]

    metrics_df, best_model, best_forecast = revenue_forecasting_engine(
        revenue_df,
        forecast_periods=3
    )

    revenue_forecast_df = pd.DataFrame({
        "Year": [
            "FY2026-27",
            "FY2027-28",
            "FY2028-29"
        ],
        "Revenue": np.array(best_forecast).flatten()
    })

    st.sidebar.success(
        f"Selected Model: {best_model}"
    )

else:

    revenue_forecast_df = st.sidebar.data_editor(
        pd.DataFrame({
            "Year": [
                "FY2026-27",
                "FY2027-28",
                "FY2028-29"
            ],
            "Revenue": [
                8965.82,
                9665.19,
                10364.56
            ]
        }),
        num_rows="dynamic",
        hide_index=True
    )


# =========================================================
# COMPUTE FORECASTS
# =========================================================

forecast_rows = []

for _, row in revenue_forecast_df.iterrows():

    if pd.isna(row["Revenue"]):
        continue

    res = forecast_from_revenue(
        float(row["Revenue"]),
        models
    )

    res["Year"] = row["Year"]

    forecast_rows.append(res)

forecast_df = pd.DataFrame(forecast_rows)

if not forecast_df.empty:

    forecast_df["Net Working Capital"] = (
        forecast_df["Inventory"]
        + forecast_df["Receivables"]
        - forecast_df["Payables"]
    )

    forecast_df = forecast_df[
        [
            "Year",
            "Revenue",
            "COGS",
            "Inventory",
            "Receivables",
            "Payables",
            "Net Working Capital",
            "DSO",
            "DIO",
            "DPO",
            "CCC"
        ]
    ]

# =========================================================
# HEADER
# =========================================================
st.markdown(f"""
<div style="background:{NAVY};padding:22px 28px;border-radius:10px;margin-bottom:18px;">
    <div style="color:{GOLD};font-size:12px;font-weight:700;letter-spacing:2px;">
        WORKING CAPITAL OPTIMIZATION — DECISION SUPPORT SYSTEM
    </div>
    <div style="color:white;font-size:26px;font-weight:800;margin-top:4px;">
        Centum Electronics Ltd. — Cash Conversion Cycle Dashboard
    </div>
    <div style="color:#C9D6E8;font-size:13px;margin-top:4px;">
        Revenue is the only manual forecast input — COGS, Receivables, Inventory, Payables and all ratios are computed automatically via driver-based regression.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab_inv, tab3, tab4, tab5 = st.tabs([
    "📊 Current Data (Historical)", "🔮 Forecast & KPIs", "📦 Inventory Deep-Dive",
    "🎯 Benchmark & Gap", "💰 Cash Release Opportunity", "⚙️ Model Diagnostics"
])

# =========================================================
# TAB 1 — CURRENT DATA (HISTORICAL, AS PER DISSERTATION)
# =========================================================
with tab1:
    st.subheader("Historical Financials & Calculated KPIs")
    st.dataframe(
        hist_df.style.format({
            "Revenue": "{:,.2f}", "COGS": "{:,.2f}", "Inventory": "{:,.2f}",
            "Receivables": "{:,.2f}", "Payables": "{:,.2f}",
            "DSO": "{:.2f}", "DIO": "{:.2f}", "DPO": "{:.2f}", "CCC": "{:.2f}",
        }),
        use_container_width=True, height=340,
    )

    latest = hist_df.iloc[-1]
    st.markdown(f"#### Latest Year Snapshot — {latest['Year']}")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4], ["DSO", "DIO", "DPO", "CCC"],
        [latest["DSO"], latest["DIO"], latest["DPO"], latest["CCC"]]
    ):
        with col:
            kpi_card(f"{label} (Days)", f"{val:.1f}", "Latest historical value", STEEL)

    st.markdown("#### Historical Trend")
    fig = go.Figure()
    for col, color in zip(["DSO", "DIO", "DPO", "CCC"], [STEEL, GOLD, GREEN, NAVY]):
        fig.add_trace(go.Scatter(x=hist_df["Year"], y=hist_df[col], mode="lines+markers", name=col, line=dict(color=color, width=3)))
    fig.update_layout(height=420, plot_bgcolor="white", legend=dict(orientation="h", y=-0.2),
                       yaxis_title="Days", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 — FORECAST & KPIs (AUTO-COMPUTED FROM REVENUE INPUT)
# =========================================================
with tab2:
    if forecast_df.empty:
        st.info("Enter at least one forecasted Revenue value in the sidebar to see results.")
    else:
        st.subheader("Auto-Computed Forecast (from Manual Revenue Input)")
        st.dataframe(
            forecast_df.style.format({
                "Revenue": "{:,.2f}", "COGS": "{:,.2f}", "Inventory": "{:,.2f}",
                "Receivables": "{:,.2f}", "Payables": "{:,.2f}", "Net Working Capital": "{:,.2f}",
                "DSO": "{:.2f}", "DIO": "{:.2f}", "DPO": "{:.2f}", "CCC": "{:.2f}",
            }),
            use_container_width=True,
        )

        sel_year = st.selectbox("Select forecast year for KPI detail:", forecast_df["Year"].tolist(),
                                 index=len(forecast_df) - 1)
        row = forecast_df[forecast_df["Year"] == sel_year].iloc[0]
        # ---------------- Row 1: efficiency metrics ----------------
        st.markdown(f"#### KPI Status — {sel_year}")
        c1, c2, c3, c4 = st.columns(4)

        dso_label, dso_color = lower_is_better_status(row["DSO"], bm_dso)
        dio_label, dio_color = lower_is_better_status(row["DIO"], bm_dio)
        dpo_label, dpo_color = dpo_status(row["DPO"], standard_term, bm_dpo)
        ccc_label, ccc_color = lower_is_better_status(row["CCC"], bm_ccc)

        with c1:
            kpi_card("DSO (Days)", f"{row['DSO']:.1f}", f"Benchmark: {bm_dso:.1f}", dso_color)
        with c2:
            kpi_card("DIO (Days)", f"{row['DIO']:.1f}", f"Benchmark: {bm_dio:.1f}", dio_color)
        with c3:
            kpi_card("DPO (Days)", f"{row['DPO']:.1f}", f"Standard: {standard_term:.0f} · Benchmark: {bm_dpo:.1f}", dpo_color)
        with c4:
            kpi_card("CCC (Days)", f"{row['CCC']:.1f}", f"Benchmark: {bm_ccc:.1f}", ccc_color)

        # ---------------- Row 2: efficiency metrics ----------------
        st.markdown("#### Forecasted Working Capital")

       
        c1, c2, c3 = st.columns(3)

        with c1:
            kpi_card(
                "Receivables",
                f"₹{row['Receivables']:,.2f} Mn",
                "Forecast",
                STEEL
            )

        with c2:
            kpi_card(
                "Inventory",
                f"₹{row['Inventory']:,.2f} Mn",
                "Forecast",
                GOLD
            )

        with c3:
            kpi_card(
                "Net Working Capital",
                f"₹{row['Net Working Capital']:,.2f} Mn",
                "Inventory + Receivables - Payables",
                NAVY
            )


        # ---------------- Row 3: efficiency metrics ----------------
        c1, c2, c3, c4 = st.columns(4)
        with c1: status_pill(dso_label, dso_color)
        with c2: status_pill(dio_label, dio_color)
        with c3: status_pill(dpo_label, dpo_color)
        with c4: status_pill(ccc_label, ccc_color)

        st.markdown("####  ")
        st.markdown("#### Historical + Forecast Trend")
        fig2 = go.Figure()
        combined_years = list(hist_df["Year"]) + list(forecast_df["Year"])
        for col, color in zip(["DSO", "DIO", "DPO", "CCC"], [STEEL, GOLD, GREEN, NAVY]):
            y_hist = list(hist_df[col])
            y_fc = list(forecast_df[col])
            fig2.add_trace(go.Scatter(x=hist_df["Year"], y=y_hist, mode="lines+markers", name=f"{col} (Actual)", line=dict(color=color, width=3)))
            fig2.add_trace(go.Scatter(
                x=[hist_df["Year"].iloc[-1]] + list(forecast_df["Year"]),
                y=[y_hist[-1]] + y_fc, mode="lines+markers", name=f"{col} (Forecast)",
                line=dict(color=color, width=3, dash="dash"), showlegend=False,
            ))
        fig2.update_layout(height=440, plot_bgcolor="white", legend=dict(orientation="h", y=-0.2),
                            yaxis_title="Days", margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# TAB — INVENTORY DEEP-DIVE
# =========================================================
with tab_inv:
    if forecast_df.empty:
        st.info("Enter a forecasted Revenue value in the sidebar to see the inventory analysis.")
    else:
        sel_year_inv = st.selectbox("Forecast year:", forecast_df["Year"].tolist(),
                                     index=len(forecast_df) - 1, key="inv_year")
        row = forecast_df[forecast_df["Year"] == sel_year_inv].iloc[0]

        st.markdown("Inventory is the largest single driver of the working capital gap. "
                     "This page shows how much inventory *should* be held for the forecasted "
                     "business volume, versus how much the current trajectory implies.")

        # ---------------- Target DIO control ----------------
        c_target, c_info = st.columns([1, 2])
        with c_target:
            target_dio = st.slider(
                "Target DIO (days) — default is benchmark", min_value=float(standard_term),
                max_value=float(max(row["DIO"], bm_dio) + 20), value=float(bm_dio), step=0.5,
            )
        with c_info:
            st.caption(
                f"Default target = weighted industry benchmark DIO ({bm_dio:.1f} days). "
                f"Drag to test a more aggressive or more conservative inventory target — "
                f"every metric below recalculates instantly."
            )

        # ---------------- Core calculations ----------------
        daily_cogs = row["COGS"] / 360
        target_inventory = target_dio / 360 * row["COGS"]
        excess_inventory = row["Inventory"] - target_inventory
        excess_days = row["DIO"] - target_dio
        turnover_actual = row["COGS"] / row["Inventory"]
        turnover_target = 360 / target_dio
        inv_pct_revenue = row["Inventory"] / row["Revenue"] * 100
        inv_pct_cogs = row["Inventory"] / row["COGS"] * 100
        weeks_cover = row["DIO"] / 7
        months_cover = row["DIO"] / 30.44
        efficiency_index = target_dio / row["DIO"] * 100  # 100% = at target; <100% = holding excess
        cogs_pct_revenue = row["COGS"] / row["Revenue"] * 100

        # ---------------- Row 1: headline KPIs ----------------
        st.markdown(f"#### Inventory Position — {sel_year_inv}")
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Forecasted Inventory (BAU)", f"₹{row['Inventory']:,.2f} Mn",
                     "Regression-based, business-as-usual trajectory", NAVY)
        with c2:
            kpi_card("Target Inventory", f"₹{target_inventory:,.2f} Mn",
                     f"At {target_dio:.1f}-day DIO target", STEEL)
        with c3:
            excess_color = RED if excess_inventory > 0 else GREEN
            excess_label = "Excess vs. Target" if excess_inventory > 0 else "Already Below Target"
            kpi_card(excess_label, f"₹{abs(excess_inventory):,.2f} Mn",
                     f"{excess_days:+.1f} days vs. target", excess_color)

        # ---------------- Row 2: efficiency metrics ----------------
        st.markdown("#### Efficiency Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dio_label, dio_color = lower_is_better_status(row["DIO"], target_dio)
            kpi_card("DIO (Days)", f"{row['DIO']:.1f}", f"Target: {target_dio:.1f}", dio_color)
        with c2:
            turn_color = GREEN if turnover_actual >= turnover_target else RED
            kpi_card("Inventory Turnover", f"{turnover_actual:.2f}×",
                     f"Target: {turnover_target:.2f}× per year", turn_color)
        with c3:
            eff_color = GREEN if efficiency_index >= 95 else (AMBER if efficiency_index >= 80 else RED)
            kpi_card("Inventory Efficiency Index", f"{efficiency_index:.0f}%",
                     "100% = at target DIO", eff_color)
        with c4:
            kpi_card("Weeks / Months of Cover", f"{weeks_cover:.1f} wks",
                     f"≈ {months_cover:.1f} months", STEEL)

        st.markdown("#### Inventory as a Share of the Business")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Inventory % of Revenue", f"{inv_pct_revenue:.1f}%",
                     "Inventory ÷ Forecasted Revenue", NAVY)
        with c2:
            kpi_card("Inventory % of COGS", f"{inv_pct_cogs:.1f}%",
                     "Inventory ÷ Forecasted COGS", STEEL)
        with c3:
            kpi_card("Potential Cash Release", f"₹{max(excess_inventory,0):,.2f} Mn",
                     "If inventory is brought to target level", NAVY)
        with c4:
            kpi_card("COGS % of Revenue", f"{cogs_pct_revenue:.1f}%",
                     "Cost of goods sold as % of revenue", GOLD)

        # ---------------- Chart 1: BAU vs Target Inventory, all forecast years ----------------
        st.markdown("#### Forecasted Inventory vs. Target — All Forecast Years")
        inv_chart_df = forecast_df.copy()
        inv_chart_df["Target Inventory"] = target_dio / 360 * inv_chart_df["COGS"]
        inv_chart_df["Excess"] = inv_chart_df["Inventory"] - inv_chart_df["Target Inventory"]

        fig_inv1 = go.Figure()
        fig_inv1.add_trace(go.Bar(name="Forecasted Inventory (BAU)", x=inv_chart_df["Year"],
                                   y=inv_chart_df["Inventory"], marker_color=NAVY))
        fig_inv1.add_trace(go.Bar(name="Target Inventory", x=inv_chart_df["Year"],
                                   y=inv_chart_df["Target Inventory"], marker_color=GOLD))
        fig_inv1.update_layout(barmode="group", height=400, plot_bgcolor="white",
                                yaxis_title="₹ Mn", margin=dict(t=20),
                                legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_inv1, use_container_width=True)

        st.dataframe(
            inv_chart_df[["Year", "COGS", "Inventory", "Target Inventory", "Excess", "DIO"]]
            .rename(columns={"Inventory": "Forecasted Inventory (BAU)"})
            .style.format({
                "COGS": "{:,.2f}", "Forecasted Inventory (BAU)": "{:,.2f}",
                "Target Inventory": "{:,.2f}", "Excess": "{:+,.2f}", "DIO": "{:.2f}",
            }),
            use_container_width=True,
        )

        # ---------------- Chart 2: DIO trend with target line ----------------
        st.markdown("#### DIO Trend — Historical, Forecast & Target")
        fig_inv2 = go.Figure()
        fig_inv2.add_trace(go.Scatter(x=hist_df["Year"], y=hist_df["DIO"], mode="lines+markers",
                                       name="DIO (Actual)", line=dict(color=NAVY, width=3)))
        fig_inv2.add_trace(go.Scatter(
            x=[hist_df["Year"].iloc[-1]] + list(forecast_df["Year"]),
            y=[hist_df["DIO"].iloc[-1]] + list(forecast_df["DIO"]),
            mode="lines+markers", name="DIO (Forecast, BAU)",
            line=dict(color=NAVY, width=3, dash="dash"),
        ))
        fig_inv2.add_hline(y=bm_dio, line_dash="dot", line_color=GOLD,
                            annotation_text=f"Benchmark ({bm_dio:.1f}d)", annotation_position="top left")
        if abs(target_dio - bm_dio) > 0.01:
            fig_inv2.add_hline(y=target_dio, line_dash="dot", line_color=GREEN,
                                annotation_text=f"Target ({target_dio:.1f}d)", annotation_position="bottom left")
        fig_inv2.update_layout(height=400, plot_bgcolor="white", yaxis_title="Days", margin=dict(t=20),
                                legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_inv2, use_container_width=True)

        # ---------------- Phased glide path to target ----------------
        st.markdown("#### Phased Glide Path to Target")
        st.caption("Assumes a straight-line reduction from the current DIO to the target DIO, "
                    "spread evenly across the entered forecast years.")
        n_years = len(forecast_df)
        start_dio = hist_df["DIO"].iloc[-1]
        glide_rows = []
        for i, (_, fr) in enumerate(forecast_df.iterrows()):
            glide_dio = start_dio + (target_dio - start_dio) * (i + 1) / n_years
            glide_inventory = glide_dio / 360 * fr["COGS"]
            bau_inventory = fr["Inventory"]
            incremental_release = bau_inventory - glide_inventory
            glide_rows.append({
                "Year": fr["Year"], "BAU DIO": fr["DIO"], "Glide-Path Target DIO": glide_dio,
                "BAU Inventory": bau_inventory, "Glide-Path Inventory": glide_inventory,
                "Cash Release vs. BAU": incremental_release,
            })
        glide_df = pd.DataFrame(glide_rows)
        st.dataframe(
            glide_df.style.format({
                "BAU DIO": "{:.1f}", "Glide-Path Target DIO": "{:.1f}",
                "BAU Inventory": "{:,.2f}", "Glide-Path Inventory": "{:,.2f}",
                "Cash Release vs. BAU": "{:+,.2f}",
            }),
            use_container_width=True,
        )

        fig_glide = go.Figure()
        fig_glide.add_trace(go.Bar(name="BAU Inventory", x=glide_df["Year"], y=glide_df["BAU Inventory"], marker_color=STEEL))
        fig_glide.add_trace(go.Bar(name="Glide-Path Inventory", x=glide_df["Year"], y=glide_df["Glide-Path Inventory"], marker_color=GREEN))
        fig_glide.update_layout(barmode="group", height=380, plot_bgcolor="white", yaxis_title="₹ Mn",
                                 margin=dict(t=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_glide, use_container_width=True)

        st.markdown(f"""
        <div style="background:{GREEN}18;border-left:5px solid {GREEN};padding:14px 18px;border-radius:6px;margin-top:6px;">
        <b style="color:{GREEN};">Recommendation:</b> Reduce DIO from {start_dio:.1f} days toward
        {target_dio:.1f} days in equal steps over {n_years} year(s) via tighter procurement planning
        and demand-aligned inventory control — this phased path releases
        ₹{glide_df['Cash Release vs. BAU'].sum():,.2f} Mn cumulatively versus the business-as-usual trajectory,
        without a disruptive one-time inventory correction.
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3 — BENCHMARK & GAP ANALYSIS
# =========================================================
with tab3:
    if forecast_df.empty:
        st.info("Enter a forecasted Revenue value in the sidebar to see the benchmark comparison.")
    else:
        sel_year3 = st.selectbox("Forecast year:", forecast_df["Year"].tolist(),
                                  index=len(forecast_df) - 1, key="gap_year")
        row = forecast_df[forecast_df["Year"] == sel_year3].iloc[0]

        gap_df = pd.DataFrame({
            "Metric": ["DSO", "DIO", "DPO", "CCC"],
            "Forecast": [row["DSO"], row["DIO"], row["DPO"], row["CCC"]],
            "Benchmark": [bm_dso, bm_dio, bm_dpo, bm_ccc],
        })
        gap_df["Gap (Days)"] = gap_df["Forecast"] - gap_df["Benchmark"]
        gap_df["Gap %"] = gap_df["Gap (Days)"] / gap_df["Benchmark"] * 100

        def row_status(m, gap):
            if m == "DPO":
                return dpo_status(row["DPO"], standard_term, bm_dpo)[0]
            return "Better than Benchmark" if gap <= 0 else "Improvement Required"

        gap_df["Status"] = [row_status(m, g) for m, g in zip(gap_df["Metric"], gap_df["Gap (Days)"])]

        st.subheader(f"Forecast vs. Weighted Benchmark — {sel_year3}")
        st.dataframe(
            gap_df.style.format({"Forecast": "{:.2f}", "Benchmark": "{:.2f}", "Gap (Days)": "{:+.2f}", "Gap %": "{:+.1f}%"}),
            use_container_width=True,
        )

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Forecast", x=gap_df["Metric"], y=gap_df["Forecast"], marker_color=NAVY))
        fig3.add_trace(go.Bar(name="Benchmark", x=gap_df["Metric"], y=gap_df["Benchmark"], marker_color=GOLD))
        fig3.update_layout(barmode="group", height=420, plot_bgcolor="white", yaxis_title="Days", margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# TAB 4 — CASH RELEASE OPPORTUNITY (DURABLE vs SUPPLIER-FUNDED)
# =========================================================
with tab4:
    if forecast_df.empty:
        st.info("Enter a forecasted Revenue value in the sidebar to see the cash release analysis.")
    else:
        sel_year4 = st.selectbox("Forecast year:", forecast_df["Year"].tolist(),
                                  index=len(forecast_df) - 1, key="cash_year")
        row = forecast_df[forecast_df["Year"] == sel_year4].iloc[0]

        daily_cogs = row["COGS"] / 360
        daily_rev = row["Revenue"] / 360

        dio_gap = max(row["DIO"] - bm_dio, 0)
        dso_gap = max(row["DSO"] - bm_dso, 0)
        dpo_gap_bm = max(row["DPO"] - bm_dpo, 0)
        dpo_gap_std = max(row["DPO"] - standard_term, 0)

        cash_dio = dio_gap * daily_cogs
        cash_dso = dso_gap * daily_rev
        total_identified = cash_dio + cash_dso
        supplier_funded = dpo_gap_bm * daily_cogs
        durable_gain = total_identified - supplier_funded
        funding_risk_std = dpo_gap_std * daily_cogs

        st.subheader(f"Cash Release Opportunity — {sel_year4}")
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Inventory (DIO) Opportunity", f"₹{cash_dio:,.2f} Mn", f"{dio_gap:.2f} days vs. benchmark", NAVY)
        with c2:
            kpi_card("Receivables (DSO) Opportunity", f"₹{cash_dso:,.2f} Mn", f"{dso_gap:.2f} days vs. benchmark", NAVY)
        with c3:
            kpi_card("Total Identified Opportunity", f"₹{total_identified:,.2f} Mn", f"{dio_gap + dso_gap:.2f} days combined", GOLD)

        st.markdown("#### Durable Gain vs. Supplier-Funded Cushion")
        c1, c2 = st.columns(2)
        with c1:
            kpi_card("Durable Efficiency Gain", f"₹{durable_gain:,.2f} Mn",
                     "Genuine CCC-gap improvement vs. benchmark", GREEN)
        with c2:
            kpi_card("Currently Supplier-Funded", f"₹{supplier_funded:,.2f} Mn",
                     "Sustained only because DPO exceeds benchmark", RED)

        st.markdown(f"""
        <div style="background:{RED}18;border-left:5px solid {RED};padding:14px 18px;border-radius:6px;margin-top:6px;">
        <b style="color:{RED};">Risk statement:</b> If supplier terms tighten toward standard trade terms
        ({standard_term:.0f} days), an additional <b>₹{funding_risk_std:,.2f} Mn</b> of working capital funding
        would be required — with no change to DIO or DSO.
        </div>
        """, unsafe_allow_html=True)

        fig4 = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Durable Efficiency Gain", "Supplier-Funded Cushion", "Total Identified Opportunity"],
            y=[durable_gain, supplier_funded, 0],
            text=[f"₹{durable_gain:,.0f} Mn", f"₹{supplier_funded:,.0f} Mn", f"₹{total_identified:,.0f} Mn"],
            textposition="outside",
            connector={"line": {"color": GREY}},
            decreasing={"marker": {"color": RED}},
            increasing={"marker": {"color": GREEN}},
            totals={"marker": {"color": NAVY}},
        ))
        fig4.update_layout(height=420, plot_bgcolor="white", margin=dict(t=20), yaxis_title="₹ Mn")
        st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# TAB 5 — MODEL DIAGNOSTICS (TRANSPARENCY ON THE "AUTO" PART)
# =========================================================
with tab5:
    st.subheader("Driver-Based Regression Models (Fitted on Historical Data)")

    st.caption(
        "These equations are what automatically compute COGS, "
        "Receivables, Inventory and Payables once Revenue is entered."
    )

    diag_rows = [
        ("COGS = f(Revenue)", models["cogs_on_revenue"]),
        ("Receivables = f(Revenue)", models["receivables_on_revenue"]),
        ("Inventory = f(COGS)", models["inventory_on_cogs"]),
        ("Payables = f(COGS)", models["payables_on_cogs"]),
    ]

    diag_df = pd.DataFrame([
        {
            "Relationship": name,
            "Equation": f"y = {m['slope']:.4f}x + {m['intercept']:,.2f}",
            "R²": round(m["r2"], 4),
        }
        for name, m in diag_rows
    ])

    st.dataframe(diag_df, use_container_width=True)

    st.caption(
        f"Fitted on {len(hist_df)} historical annual observations"
        f"{' (COVID years excluded)' if exclude_covid else ''}."
    )

    st.markdown("---")

    st.subheader("Revenue Forecast Model Comparison")

    st.dataframe(
        metrics_df.style.format({
            "MAE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "MAPE": "{:.2f}"
        }),
        use_container_width=True
    )

    st.success(
        f"Selected Revenue Forecast Model: {best_model}"
    )
