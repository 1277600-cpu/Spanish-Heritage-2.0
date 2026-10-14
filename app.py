import io
import math
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import sympy as sp
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

st.set_page_config(page_title="Hispanic Heritage App 2.0 | Function Lab", page_icon="📈", layout="wide")

# ------------------------------- Context library -------------------------------
CONTEXTS = {
    "Business": dict(description="Revenue or profit over time", x="time", xu="months", y="profit", yu="thousands of dollars", kind="economic", visual="Business / economic indicator"),
    "Engineering": dict(description="Position of a moving object", x="time", xu="seconds", y="position", yu="meters", kind="motion", visual="Moving object / projectile"),
    "Medical": dict(description="Medication concentration over time", x="time", xu="hours", y="concentration", yu="mg/L", kind="concentration", visual="Tank / drug concentration"),
    "Sports science": dict(description="Position during a movement", x="time", xu="seconds", y="position", yu="meters", kind="motion", visual="Walking person with prosthetic leg"),
    "Biology": dict(description="Population growth over time", x="time", xu="days", y="population", yu="organisms", kind="population", visual="Population / cell colony"),
    "Bioengineering": dict(description="Cell-culture concentration", x="time", xu="hours", y="cell concentration", yu="million cells/mL", kind="population", visual="Population / cell colony"),
    "Economic forecasting": dict(description="An economic index over time", x="time", xu="years", y="index", yu="index points", kind="economic", visual="Business / economic indicator"),
    "Weather": dict(description="Temperature over time", x="time", xu="hours", y="temperature", yu="°C", kind="temperature", visual="Thermometer / weather indicator"),
    "Custom": dict(description="A quantity you choose to model", x="x", xu="units", y="f(x)", yu="units", kind="custom", visual="Moving object / projectile"),
}
SUGGESTIONS = {
    "Business": [("Quadratic profit", "x**2 - 8*x + 12"), ("Exponential growth", "100*exp(0.08*x)"), ("Seasonal demand", "50 + 12*sin(2*pi*x/12)")],
    "Engineering": [("Position", "x**3 - 6*x**2 + 9*x"), ("Projectile height", "-4.9*x**2 + 20*x + 2"), ("Oscillation", "3*sin(2*pi*x)")],
    "Medical": [("Concentration decay", "10*exp(-0.35*x)"), ("Dose response", "8*x/(2+x)"), ("Concentration pulse", "x*exp(-x)")],
    "Sports science": [("Stride motion", "sin(2*pi*x)"), ("Position", "x**3 - 3*x"), ("Projectile", "-4.9*x**2 + 15*x + 1")],
    "Biology": [("Population growth", "100*exp(0.12*x)"), ("Saturation", "1000/(1 + 9*exp(-0.4*x))"), ("Seasonal population", "500 + 80*sin(2*pi*x/12)")],
    "Bioengineering": [("Cell growth", "2*exp(0.2*x)"), ("Saturation", "10*x/(1+x)"), ("Oscillation", "5 + 0.5*sin(x)")],
    "Economic forecasting": [("Linear trend", "2*x + 100"), ("Quadratic trend", "x**2 - 4*x + 100"), ("Exponential index", "100*exp(0.03*x)")],
    "Weather": [("Daily temperature", "15 + 8*sin(2*pi*(x-6)/24)"), ("Cooling", "25 + 10*exp(-0.2*x)"), ("Linear trend", "0.2*x + 10")],
    "Custom": [("Quadratic", "x**2 - 4*x + 3"), ("Sine", "2*sin(x)"), ("Rational", "(x+1)/(x-2)")],
}
WB_INDICATORS = {
    "GDP per capita (current US$)": "NY.GDP.PCAP.CD",
    "Population, total": "SP.POP.TOTL",
    "Life expectancy at birth (years)": "SP.DYN.LE00.IN",
    "CO₂ emissions (metric tons per capita)": "EN.ATM.CO2E.PC",
    "Unemployment (% of labor force)": "SL.UEM.TOTL.ZS",
    "Inflation, consumer prices (annual %)": "FP.CPI.TOTL.ZG",
}

x = sp.Symbol("x", real=True)
SAFE_LOCALS = {
    "x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin,
    "acos": sp.acos, "atan": sp.atan, "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "E": sp.E, "sinh": sp.sinh,
    "cosh": sp.cosh, "tanh": sp.tanh,
}

def parse_expression(raw):
    raw = str(raw).strip().replace("^", "**").replace("π", "pi")
    if not raw:
        raise ValueError("Enter a function, for example x**2 - 4*x + 3.")
    if len(raw) > 300:
        raise ValueError("Function is too long (maximum 300 characters).")
    try:
        expr = sp.sympify(raw, locals=SAFE_LOCALS, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not parse the function. Use x, +, -, *, /, **, parentheses, and supported functions such as sin(x), exp(x), log(x).") from exc
    if expr.free_symbols - {x}:
        raise ValueError("Only the variable x is supported. Check for unknown variable names.")
    if expr.has(sp.Integral, sp.Derivative, sp.Sum):
        raise ValueError("Calculus operators are not accepted in the input function.")
    return expr

def safe_float(expr, value):
    try:
        result = complex(sp.N(expr.subs(x, float(value))))
        if abs(result.imag) > 1e-7 or not math.isfinite(result.real):
            return np.nan
        return float(result.real)
    except Exception:
        return np.nan

def eval_array(expr, values):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        try:
            yv = np.asarray(fn(np.asarray(values, dtype=float)), dtype=float)
            if yv.shape == ():
                yv = np.full(len(values), float(yv))
            yv = np.broadcast_to(yv, np.asarray(values).shape).astype(float)
            yv[~np.isfinite(yv)] = np.nan
            return yv
        except Exception:
            return np.array([safe_float(expr, v) for v in values], dtype=float)

def fmt(v, digits=6):
    if v is None:
        return "—"
    try:
        if not np.isfinite(float(v)):
            return "undefined"
        return f"{float(v):,.{digits}g}"
    except Exception:
        return str(v)

def expr_text(expr):
    try:
        return sp.sstr(sp.simplify(expr))
    except Exception:
        return str(expr)

def expression_domain_notes(expr):
    notes = []
    for den in sp.denom(sp.together(expr)).as_ordered_factors():
        try:
            roots = sp.solve(sp.Eq(den, 0), x)
            if roots:
                notes.append("denominator is zero at x = " + ", ".join(expr_text(r) for r in roots))
        except Exception:
            pass
    for log_arg in expr.atoms(sp.log):
        try:
            notes.append("logarithm requires " + expr_text(log_arg.args[0]) + " > 0")
        except Exception:
            pass
    for power in expr.atoms(sp.Pow):
        if power.exp.is_Rational and power.exp.q % 2 == 0:
            notes.append("even-root expressions require the radicand to be nonnegative")
    return list(dict.fromkeys(notes)) or ["No simple domain restriction detected; numerical checks still apply."]

def derivative_analysis(expr, lo, hi):
    f1 = sp.diff(expr, x)
    f2 = sp.diff(f1, x)
    rows = []
    try:
        roots = sp.solveset(sp.Eq(f1, 0), x, domain=sp.Interval(lo, hi))
        candidates = list(roots) if isinstance(roots, sp.FiniteSet) else []
    except Exception:
        candidates = []
    try:
        singular = list(sp.singularities(expr, x))
        candidates += [r for r in singular if r.is_real and lo <= float(r) <= hi]
    except Exception:
        singular = []
    candidates = sorted(set([float(sp.N(c)) for c in candidates if getattr(c, "is_real", False) is not False and np.isfinite(float(sp.N(c))) and lo <= float(sp.N(c)) <= hi]))
    eps = max((hi - lo) * 1e-5, 1e-5)
    for c in candidates:
        if not np.isfinite(safe_float(expr, c)):
            rows.append({"x": c, "f(x)": np.nan, "classification": "undefined / discontinuity", "reason": "Function is not defined here."})
            continue
        left_d = safe_float(f1, max(lo, c-eps))
        right_d = safe_float(f1, min(hi, c+eps))
        if abs(safe_float(f1, c)) < 1e-5:
            if left_d > 1e-6 and right_d < -1e-6:
                cl, reason = "local maximum", "f′ changes + to −"
            elif left_d < -1e-6 and right_d > 1e-6:
                cl, reason = "local minimum", "f′ changes − to +"
            elif left_d * right_d > 0:
                cl, reason = "stationary point, not an extremum", "f′ does not change sign"
            else:
                cl, reason = "critical point (numerical test inconclusive)", "sign test near point is inconclusive"
        else:
            cl, reason = "critical / non-differentiable point", "derivative is zero or undefined"
        rows.append({"x": c, "f(x)": safe_float(expr, c), "classification": cl, "reason": reason})
    # Numerical inflection check: evaluate concavity on both sides of candidate roots of f''
    inflections = []
    try:
        r2 = sp.solveset(sp.Eq(f2, 0), x, domain=sp.Interval(lo, hi))
        roots2 = list(r2) if isinstance(r2, sp.FiniteSet) else []
    except Exception:
        roots2 = []
    for r in roots2:
        try:
            c = float(sp.N(r))
            if not lo < c < hi:
                continue
            if not np.isfinite(safe_float(expr, c)):
                continue
            left = safe_float(f2, c-eps)
            right = safe_float(f2, c+eps)
            if np.isfinite(left) and np.isfinite(right) and left * right < 0:
                inflections.append({"x": c, "f(x)": safe_float(expr, c), "evidence": "f″ changes sign"})
        except Exception:
            pass
    # Endpoints are included for absolute extrema, where defined.
    sample_x = np.linspace(lo, hi, 1601)
    sample_y = eval_array(expr, sample_x)
    valid = np.isfinite(sample_y)
    abs_extrema = []
    if valid.any():
        valid_y = sample_y[valid]
        valid_x = sample_x[valid]
        imin, imax = int(np.argmin(valid_y)), int(np.argmax(valid_y))
        abs_extrema = [
            {"type": "Numerical minimum", "x": float(valid_x[imin]), "f(x)": float(valid_y[imin])},
            {"type": "Numerical maximum", "x": float(valid_x[imax]), "f(x)": float(valid_y[imax])},
        ]
    return f1, f2, rows, inflections, abs_extrema

def make_xlsx(summary, data_df, result_df):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    for row in summary:
        ws.append(row)
    ws["A1"].font = Font(bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="24476B")
    ws.freeze_panes = "A2"
    wd = wb.create_sheet("Working data")
    for row in [list(data_df.columns)] + data_df.astype(object).where(pd.notna(data_df), "").values.tolist():
        wd.append(row)
    wr = wb.create_sheet("Calculated years")
    for row in [list(result_df.columns)] + result_df.astype(object).where(pd.notna(result_df), "").values.tolist():
        wr.append(row)
    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        for col in sheet.columns:
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col)
            sheet.column_dimensions[get_column_letter(col[0].column)].width = min(max(max_len + 2, 12), 42)
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="24476B")
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()

def make_pdf(title, summary_lines, chart_bytes, data_df, result_df):
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=26, bottomMargin=26)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for line in summary_lines:
        story.append(Paragraph(str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
    story.append(Spacer(1, 8))
    if chart_bytes:
        story.append(Image(io.BytesIO(chart_bytes), width=650, height=300))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Data preview", styles["Heading2"]))
    preview = data_df.head(12).copy().fillna("")
    if len(preview.columns):
        tab = Table([list(preview.columns)] + preview.astype(str).values.tolist(), repeatRows=1)
        tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24476B")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
        story.append(tab)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Calculated years", styles["Heading2"]))
    if not result_df.empty:
        tab = Table([list(result_df.columns)] + result_df.astype(str).values.tolist(), repeatRows=1)
        tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24476B")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(tab)
    doc.build(story)
    return out.getvalue()

def get_world_bank(country, indicator, start=1960, end=2025):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    r = requests.get(url, params={"format":"json", "per_page":20000, "date":f"{start}:{end}"}, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError("The data provider returned no observations. Try another country/indicator or check the API status.")
    rows = [{"year": int(item["date"]), "value": item["value"]} for item in payload[1] if item.get("value") is not None]
    df = pd.DataFrame(rows).drop_duplicates("year").sort_values("year")
    if df.empty:
        raise ValueError("No non-missing observations were returned for this selection.")
    return df

# -------------------------------- Session state --------------------------------
for key, default in {
    "fit_results": None, "calc_results": None, "animation_playing": False,
    "animation_x": 0.0, "last_fit_signature": None, "last_calc_signature": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("📈 Hispanic Heritage App 2.0 — Function Lab")
st.caption("Explore data, build mathematical models, and connect derivatives to real-world contexts. Historical associations are not proof of causation.")
with st.sidebar:
    st.header("Classroom mode")
    course = st.selectbox("Course level", ["Honors Algebra 2", "Precalculus", "Calculus"], index=2)
    st.markdown("---")
    st.header("Model source")
    source = st.radio("Choose starting data", ["World Bank historical data", "Enter / edit data", "Use a sample dataset"], index=2)
    if source == "World Bank historical data":
        country = st.selectbox("Country", [("United States","USA"),("Mexico","MEX"),("Spain","ESP"),("Chile","CHL"),("Colombia","COL"),("Argentina","ARG"),("Brazil","BRA"),("Canada","CAN"),("Japan","JPN"),("India","IND")], format_func=lambda v:v[0])
        indicator_label = st.selectbox("Indicator", list(WB_INDICATORS))
        data_start = st.number_input("First year", min_value=1960, max_value=2025, value=1990, step=1)
        data_end = st.number_input("Last year", min_value=1960, max_value=2025, value=2024, step=1)
    st.markdown("---")
    st.header("Best-fit settings")
    degree = st.selectbox("Polynomial degree", [1,2,3,4,5,6], index=2)
    sample_step = st.slider("Sample increment (markers/table)", 1, 10, 3)
    holdout_pct = st.slider("Chronological holdout (%)", 0, 40, 20, 5)
    st.caption("Fit metrics are in-sample unless explicitly labeled holdout. Polynomial extrapolations can become unrealistic.")

# Data retrieval / sample initialization
data_error = None
if source == "World Bank historical data":
    if data_start > data_end:
        st.error("First year must be no later than last year.")
        raw_data = pd.DataFrame(columns=["year","value"])
    else:
        try:
            raw_data = get_world_bank(country[1], WB_INDICATORS[indicator_label], int(data_start), int(data_end))
        except Exception as exc:
            data_error = str(exc)
            raw_data = pd.DataFrame(columns=["year","value"])
elif source == "Use a sample dataset":
    years = np.arange(2000, 2025)
    values = 40 + 1.7*(years-2000) + 0.18*(years-2012)**2 + 4*np.sin((years-2000)*0.65)
    raw_data = pd.DataFrame({"year": years, "value": np.round(values, 3)})
else:
    if "manual_data" not in st.session_state:
        st.session_state.manual_data = pd.DataFrame({"year":[2000,2002,2004,2006,2008,2010], "value":[12,15,17,24,29,38]})
    raw_data = st.session_state.manual_data.copy()

if data_error:
    st.error(f"Data retrieval failed: {data_error}")
    st.info("Check the country code, internet connection, World Bank API response, and deployment logs. You can switch to sample or editable data.")
if source == "Enter / edit data":
    st.subheader("Editable working data")
    st.caption("Edit values directly. Years must be numeric; duplicate years are not used in a fit until resolved.")
    edited = st.data_editor(raw_data, num_rows="dynamic", use_container_width=True, key="working_data_editor")
    st.session_state.manual_data = edited.copy()
    raw_data = edited.copy()
if not raw_data.empty:
    raw_data = raw_data.rename(columns={"value":"observed"}).copy()
    raw_data["year"] = pd.to_numeric(raw_data["year"], errors="coerce")
    raw_data["observed"] = pd.to_numeric(raw_data["observed"], errors="coerce")
    raw_data = raw_data.dropna(subset=["year","observed"])
    raw_data["year"] = raw_data["year"].astype(int)
    duplicates = raw_data["year"].duplicated(keep=False)
    if duplicates.any():
        st.warning("Duplicate years detected. The fit uses the mean value for each year; review the editable table if duplicates were unintended.")
        raw_data = raw_data.groupby("year", as_index=False)["observed"].mean()
    raw_data = raw_data.sort_values("year").reset_index(drop=True)
else:
    raw_data = pd.DataFrame(columns=["year","observed"])

# Historical fit
fit_available = False
fit_error = None
fit_data = pd.DataFrame()
fit_summary = {}
fit_expr = None
if len(raw_data) >= degree + 1 and raw_data["year"].nunique() >= degree + 1:
    years_all = raw_data["year"].to_numpy(dtype=float)
    vals_all = raw_data["observed"].to_numpy(dtype=float)
    base_year = int(np.min(years_all))
    t_all = years_all - base_year
    holdout_n = int(round(len(raw_data) * holdout_pct / 100)) if holdout_pct else 0
    holdout_n = min(max(holdout_n, 0), max(0, len(raw_data) - degree - 1))
    if holdout_n >= 1 and len(raw_data) - holdout_n >= degree + 1:
        train_df, holdout_df = raw_data.iloc[:-holdout_n], raw_data.iloc[-holdout_n:]
    else:
        train_df, holdout_df = raw_data, raw_data.iloc[0:0]
    # Main model intentionally fits all selected observations. Holdout score is a separate chronological experiment.
    try:
        coef = np.polynomial.polynomial.polyfit(t_all, vals_all, degree)
        fit_values = np.polynomial.polynomial.polyval(t_all, coef)
        residuals = vals_all - fit_values
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((vals_all - np.mean(vals_all))**2))
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(residuals)))
        z = sp.Symbol("z", real=True)
        fit_expr = sp.Add(*[sp.Float(float(c), 12) * (x - base_year)**i for i,c in enumerate(coef)])
        fit_data = raw_data.copy()
        fit_data["fitted"] = fit_values
        fit_data["residual"] = residuals
        fit_summary = {"base_year":base_year, "degree":degree, "r2":r2, "mae":mae, "n":len(raw_data), "start":int(raw_data.year.min()), "end":int(raw_data.year.max()), "mean":float(np.mean(vals_all)), "coefficients":coef}
        if not holdout_df.empty:
            train_coef = np.polynomial.polynomial.polyfit(train_df.year.to_numpy(float)-base_year, train_df.observed.to_numpy(float), degree)
            holdout_pred = np.polynomial.polynomial.polyval(holdout_df.year.to_numpy(float)-base_year, train_coef)
            fit_summary["holdout_mae"] = float(np.mean(np.abs(holdout_df.observed.to_numpy(float)-holdout_pred)))
            fit_summary["holdout_n"] = len(holdout_df)
            fit_summary["holdout_start"] = int(holdout_df.year.min())
        fit_available = True
    except Exception as exc:
        fit_error = str(exc)
else:
    fit_error = f"Need at least degree + 1 = {degree+1} distinct valid years; currently have {raw_data.year.nunique() if not raw_data.empty else 0}."

if not fit_available:
    st.warning(f"Historical fit unavailable: {fit_error}")
else:
    st.subheader("Historical data and polynomial fit")
    st.caption("Centered-time model: t = year − base year. Predictions reuse this exact centered transformation once.")
    cols = st.columns(4)
    cols[0].metric("Observations", fit_summary["n"])
    cols[1].metric("R² (in-sample)", fmt(fit_summary["r2"], 5))
    cols[2].metric("MAE (in-sample)", fmt(fit_summary["mae"], 5))
    cols[3].metric("Observed mean", fmt(fit_summary["mean"], 7))
    equation = " + ".join([f"({c:.10g})·(year − {fit_summary['base_year']})^{i}" for i,c in enumerate(fit_summary["coefficients"])])
    st.code(f"f(year) = {equation}", language="text")
    st.write(f"**Base year:** {fit_summary['base_year']} · **Degree:** {degree} · **Historical range:** {fit_summary['start']}–{fit_summary['end']}")
    st.write(f"**Residuals** (observed − fitted): {', '.join(fmt(v,5) for v in fit_data['residual'].to_list())}")
    if "holdout_mae" in fit_summary:
        st.info(f"Chronological holdout MAE: {fit_summary['holdout_mae']:.6g} across {fit_summary['holdout_n']} later observations beginning in {fit_summary['holdout_start']}. This score is separate from the full-data in-sample metrics.")
    else:
        st.caption("Chronological holdout score not shown: there are not enough observations for a separate holdout while retaining degree + 1 training years.")
    st.caption("Uncertainty: the in-sample MAE summarizes typical absolute residual size, not a 95% prediction interval. The model assumes a polynomial trend and may miss structural changes, measurement error, and contextual factors.")

    # Predictions panel
    st.subheader("Calculate a year")
    c1,c2 = st.columns([1,1])
    with c1:
        single_year = st.number_input("Year (1950–2100)", min_value=1950, max_value=2100, value=max(1950,min(2100,fit_summary["end"])), step=1, key="single_year")
        calculate = st.button("Calculate Value", type="primary")
    with c2:
        batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
        if batch_mode == "Year range":
            range_start = st.number_input("Range start", min_value=1950, max_value=2100, value=max(1950,min(2100,fit_summary["start"])), step=1, key="range_start")
            range_end = st.number_input("Range end", min_value=1950, max_value=2100, value=max(1950,min(2100,fit_summary["end"])), step=1, key="range_end")
            range_step = st.slider("Range step", 1, 10, 1)
            custom_text = ""
        else:
            custom_text = st.text_input("Comma-separated years", "2000, 2010, 2030, 2050")
            range_start, range_end, range_step = 0, -1, 1
        batch_btn = st.button("Calculate Batch")
    signature = (source, degree, tuple(raw_data[["year","observed"]].itertuples(index=False, name=None)), fit_summary["base_year"] if fit_available else None)
    if st.session_state.last_fit_signature != signature:
        st.session_state.calc_results = None
        st.session_state.last_fit_signature = signature
    if calculate:
        yy = int(single_year)
        pred = float(np.polynomial.polynomial.polyval(yy-fit_summary["base_year"], fit_summary["coefficients"]))
        observed_row = raw_data.loc[raw_data.year == yy, "observed"]
        st.session_state.single_result = {"year":yy, "estimate":pred, "observed":float(observed_row.iloc[0]) if len(observed_row) else np.nan, "fitted_range":f"{fit_summary['start']}–{fit_summary['end']}", "classification":"within fitted historical range" if fit_summary["start"] <= yy <= fit_summary["end"] else "outside fitted historical range"}
    if "single_result" in st.session_state:
        sr = st.session_state.single_result
        if sr["year"] >= 1950 and sr["year"] <= 2100:
            st.markdown(f"**Year {sr['year']} estimate:** {sr['estimate']:,.6g} (data units)")
            st.write(f"Fitted range: {sr['fitted_range']} · Classification: **{sr['classification']}**")
            if np.isfinite(sr["observed"]):
                st.write(f"At an observed year: actual observation = {sr['observed']:,.6g}; fitted value = {sr['estimate']:,.6g}. These are different quantities.")
            else:
                st.write("No actual observation exists for this year; this is a model estimate, not an observed value.")
            if sr["classification"].startswith("outside"):
                st.warning("Polynomial extrapolation can become implausible rapidly. This estimate is not clipped; inspect the graph and compare against subject-matter limits.")
    if batch_btn:
        if batch_mode == "Year range":
            if range_start > range_end:
                st.error("Range start must be less than or equal to range end.")
                years_req = []
            else:
                years_req = list(range(int(range_start), int(range_end)+1, int(range_step)))
        else:
            years_req = []
            invalid = []
            for token in custom_text.split(","):
                token = token.strip()
                if not token:
                    continue
                if re.fullmatch(r"\d{4}", token) and 1950 <= int(token) <= 2100:
                    years_req.append(int(token))
                else:
                    invalid.append(token)
            if invalid:
                st.error("Ignored invalid year entries (expected whole years from 1950–2100): " + ", ".join(invalid))
        years_req = list(dict.fromkeys(years_req))
        if not years_req:
            st.error("No valid requested years. Provide at least one year in 1950–2100.")
        else:
            out_rows = []
            for yy in years_req:
                pred = float(np.polynomial.polynomial.polyval(yy-fit_summary["base_year"], fit_summary["coefficients"]))
                actual = raw_data.loc[raw_data.year == yy, "observed"]
                out_rows.append({"year":yy,"estimate":pred,"units":"same as indicator","actual_observation":float(actual.iloc[0]) if len(actual) else np.nan,"status":"within fitted range" if fit_summary["start"] <= yy <= fit_summary["end"] else "outside fitted range","observation_status":"observed year" if len(actual) else "no actual observation"})
            st.session_state.calc_results = pd.DataFrame(out_rows)
            st.session_state.last_calc_signature = signature
    if st.session_state.calc_results is not None and st.session_state.last_calc_signature == signature:
        st.markdown("**Batch results — valid requested years**")
        st.dataframe(st.session_state.calc_results, use_container_width=True, hide_index=True)
        st.caption("Batch estimates use the same centered-time coefficients as the single-year calculator. Actual observations are displayed separately.")

    # Visualization
    st.subheader("Visualization")
    plot_min = min(1950, int(fit_summary["start"]))
    plot_max = max(2100, int(fit_summary["end"]))
    curve_years = np.linspace(plot_min, plot_max, 800)
    curve_values = np.polynomial.polynomial.polyval(curve_years-fit_summary["base_year"], fit_summary["coefficients"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve_years, y=curve_values, mode="lines", name="Smooth fitted curve", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=raw_data.year, y=raw_data.observed, mode="markers", name="Observed points", marker=dict(size=8, symbol="circle")))
    sampled_fit = fit_data.iloc[::sample_step]
    fig.add_trace(go.Scatter(x=sampled_fit.year, y=sampled_fit.fitted, mode="markers", name=f"Fitted values (every {sample_step} observations)", marker=dict(size=7, symbol="x"), visible=True))
    fig.add_vline(x=fit_summary["start"], line_dash="dash", annotation_text="Historical start")
    fig.add_vline(x=fit_summary["end"], line_dash="dash", annotation_text="Historical end")
    fig.add_trace(go.Scatter(x=[fit_summary["start"],fit_summary["end"]], y=[np.nanmax(raw_data.observed),np.nanmax(raw_data.observed)], mode="markers", name="Historical boundary markers", marker=dict(size=0), showlegend=True))
    if st.session_state.calc_results is not None and st.session_state.last_calc_signature == signature:
        br = st.session_state.calc_results
        fig.add_trace(go.Scatter(x=br.year, y=br.estimate, mode="markers", name="Batch estimates", marker=dict(size=11, symbol="diamond")))
    if "single_result" in st.session_state:
        sr = st.session_state.single_result
        if sr["year"] >= 1950 and sr["year"] <= 2100:
            fig.add_trace(go.Scatter(x=[sr["year"]], y=[sr["estimate"]], mode="markers", name="Single-year estimate", marker=dict(size=13, symbol="star")))
    fig.update_layout(height=500, xaxis_title="Year", yaxis_title=f"Indicator value ({indicator_label if source == 'World Bank historical data' else 'data units'})", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True)
    if source == "World Bank historical data":
        st.caption(f"Source: World Bank API · {country[0]} · {indicator_label}. Coverage is limited to available non-missing observations; gaps may reflect reporting availability, not zero values.")
    if st.checkbox("Compare another country for the same indicator", value=False):
        if source != "World Bank historical data":
            st.info("Choose World Bank historical data in the sidebar to enable a country comparison.")
        else:
            other = st.selectbox("Comparison country", [("Mexico","MEX"),("United States","USA"),("Spain","ESP"),("Chile","CHL"),("Colombia","COL"),("Argentina","ARG"),("Brazil","BRA"),("Canada","CAN"),("Japan","JPN"),("India","IND")], format_func=lambda v:v[0], key="compare_country")
            if other[1] == country[1]:
                st.info("Choose a different country.")
            else:
                try:
                    other_df = get_world_bank(other[1], WB_INDICATORS[indicator_label], int(data_start), int(data_end))
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=raw_data.year,y=raw_data.observed,mode="lines+markers",name=country[0]))
                    fig2.add_trace(go.Scatter(x=other_df.year,y=other_df.value,mode="lines+markers",name=other[0]))
                    fig2.update_layout(height=380,xaxis_title="Year",yaxis_title=indicator_label)
                    st.plotly_chart(fig2,use_container_width=True)
                    st.caption("Country series can have different start/end years and missing-year gaps. Values are compared on the same indicator definition; gaps are not filled or treated as zero.")
                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")

    # Course-level mathematical analysis of fitted curve using centered time variable t, displayed as year-domain model.
    st.subheader(f"{course}: model interpretation")
    st.write(f"**Domain for the fitted historical model:** all real years mathematically for a polynomial; evidence-supported range is {fit_summary['start']}–{fit_summary['end']}.")
    st.write(f"**Observed range:** {fmt(raw_data.observed.min())} to {fmt(raw_data.observed.max())}. A polynomial's theoretical range may be unbounded and is not the same as the observed range.")
    st.write(f"**Graphical trend:** inspect the fitted curve for rises, falls, bends, and potential extrapolation blow-up. Mean observed value = {fmt(fit_summary['mean'])}.")
    if course in ["Precalculus", "Calculus"]:
        st.write(f"**Degree/coefficient effects:** degree {degree} controls the maximum possible number of turning points (at most {max(0,degree-1)}); the leading coefficient and parity determine end behavior. Small high-degree coefficients can still dominate far outside the data.")
        st.write("**Changing average rates:** compare neighboring fitted values; a rising average rate suggests the trend is accelerating, while a falling rate suggests it is slowing. This is descriptive, not a causal explanation.")
    if course == "Calculus":
        # Convert polynomial in centered t to year variable u to provide derivatives in year units.
        year_symbol = sp.Symbol("year", real=True)
        centered_expr = sp.Add(*[sp.Float(float(c),12)*(year_symbol-fit_summary["base_year"])**i for i,c in enumerate(fit_summary["coefficients"])])
        d1 = sp.diff(centered_expr, year_symbol)
        d2 = sp.diff(d1, year_symbol)
        st.markdown("**First derivative:**")
        st.latex(sp.latex(sp.simplify(d1)))
        st.markdown("**Second derivative:**")
        st.latex(sp.latex(sp.simplify(d2)))
        st.caption("Derivative units are indicator-units/year and indicator-units/year². Numerical critical-point and inflection analysis below applies to the editable function in the Function Lab, not automatically to this fitted year model.")

# ------------------------------- Function Lab ---------------------------------
st.divider()
st.header("Function Lab: change context, model, and simulation")
context = st.selectbox("Context", list(CONTEXTS), index=1, key="context_select")
default_ctx = CONTEXTS[context]
ctx_cols = st.columns(2)
with ctx_cols[0]:
    context_description = st.text_input("Context description", default_ctx["description"], key=f"desc_{context}")
    x_name = st.text_input("Independent variable name", default_ctx["x"], key=f"xname_{context}")
    x_unit = st.text_input("Independent variable units", default_ctx["xu"], key=f"xunit_{context}")
with ctx_cols[1]:
    y_name = st.text_input("Dependent variable name", default_ctx["y"], key=f"yname_{context}")
    y_unit = st.text_input("Dependent variable units", default_ctx["yu"], key=f"yunit_{context}")
    visual = st.selectbox("Contextual visual", ["Walking person with prosthetic leg","Moving object / projectile","Tank / drug concentration","Population / cell colony","Thermometer / weather indicator","Business / economic indicator","Simple moving point"], index=["Walking person with prosthetic leg","Moving object / projectile","Tank / drug concentration","Population / cell colony","Thermometer / weather indicator","Business / economic indicator","Simple moving point"].index(default_ctx["visual"]) if default_ctx["visual"] in ["Walking person with prosthetic leg","Moving object / projectile","Tank / drug concentration","Population / cell colony","Thermometer / weather indicator","Business / economic indicator","Simple moving point"] else 6)
method = st.radio("Function source", ["Suggested function","Build with parameters","Enter my own function","Use historical curve of best fit"], horizontal=True)
suggestions = SUGGESTIONS[context]
if method == "Suggested function":
    chosen = st.selectbox("Suggested model", suggestions, format_func=lambda a:a[0])
    expression_text = chosen[1]
elif method == "Build with parameters":
    model_type = st.selectbox("Model family", ["Polynomial (quadratic)","Rational","Exponential","Logarithmic","Trigonometric"])
    if model_type == "Polynomial (quadratic)":
        a = st.number_input("a (x² coefficient)", value=1.0, key="pa")
        b = st.number_input("b (x coefficient)", value=-4.0, key="pb")
        c = st.number_input("c (constant)", value=3.0, key="pc")
        expression_text = f"({a})*x**2 + ({b})*x + ({c})"
    elif model_type == "Rational":
        a = st.number_input("Numerator scale", value=1.0, key="ra")
        b = st.number_input("Numerator offset", value=1.0, key="rb")
        c = st.number_input("Denominator offset (denominator = x + c)", value=-2.0, key="rc")
        expression_text = f"(({a})*x + ({b}))/(x + ({c}))"
    elif model_type == "Exponential":
        a = st.number_input("Scale a", value=10.0, key="ea")
        k = st.number_input("Exponent rate k", value=-0.35, key="ek")
        expression_text = f"({a})*exp(({k})*x)"
    elif model_type == "Logarithmic":
        a = st.number_input("Scale a", value=2.0, key="la")
        b = st.number_input("Horizontal offset b (x-b)", value=0.0, key="lb")
        c = st.number_input("Vertical offset c", value=1.0, key="lc")
        expression_text = f"({a})*log(x-({b})) + ({c})"
    else:
        a = st.number_input("Amplitude a", value=2.0, key="ta")
        b = st.number_input("Angular frequency b", value=1.0, key="tb")
        c = st.number_input("Vertical offset c", value=0.0, key="tc")
        expression_text = f"({a})*sin(({b})*x) + ({c})"
    st.code(f"f(x) = {expression_text}")
elif method == "Enter my own function":
    expression_text = st.text_input("Function f(x)", "x**3 - 3*x", help="Use ^ or ** for powers. Supported: + - * /, parentheses, sin, cos, tan, exp, log/ln, sqrt, abs, pi.")
else:
    if fit_available:
        expression_text = str(sp.sstr(fit_expr).replace("year", "x"))
        # Fit expression was built with x as the symbol and (x - base year), so this is the year-domain fit in x.
        st.info(f"Using fitted historical curve with base year {fit_summary['base_year']}. Here x is interpreted as calendar year.")
        st.code(f"f(x) = {expression_text}")
    else:
        expression_text = "x**2"
        st.warning("A historical fit is unavailable; using x² as a temporary fallback. Fix the data/degree to enable the fitted curve.")

interval_cols = st.columns(3)
with interval_cols[0]:
    x_min = st.number_input("Analysis interval start", value=0.0, key="analysis_lo")
with interval_cols[1]:
    x_max = st.number_input("Analysis interval end", value=10.0, key="analysis_hi")
with interval_cols[2]:
    sample_count = st.slider("Graph sample points", 100, 1200, 500, 50)
if x_min >= x_max:
    st.error("Analysis interval start must be smaller than the end.")
    function_ok = False
    expr = None
else:
    try:
        expr = parse_expression(expression_text)
        function_ok = True
    except ValueError as exc:
        st.error(f"Function input error: {exc}")
        function_ok = False

if function_ok:
    f1 = sp.diff(expr, x)
    f2 = sp.diff(f1, x)
    st.markdown("#### Function and derivatives")
    dcols = st.columns(3)
    with dcols[0]:
        st.markdown("**Function**")
        st.latex("f(x) = " + sp.latex(expr))
    with dcols[1]:
        show_d1 = st.checkbox("Show first derivative", value=True)
        st.markdown("**First derivative**")
        if show_d1:
            st.latex("f'(x) = " + sp.latex(sp.simplify(f1)))
    with dcols[2]:
        show_d2 = st.checkbox("Show second derivative", value=True)
        st.markdown("**Second derivative**")
        if show_d2:
            st.latex("f''(x) = " + sp.latex(sp.simplify(f2)))
    domain_notes = expression_domain_notes(expr)
    st.markdown("**Domain restrictions / discontinuities to check**")
    for note in domain_notes:
        st.write("• " + note)
    st.caption("Graphing uses NaN breaks where a function is undefined, so lines do not connect across detected invalid samples.")

    # Interactive synchronized point and secant endpoints.
    default_point = min(max((x_min+x_max)/2, x_min), x_max)
    point_x = st.slider(f"Current {x_name} ({x_unit})", float(x_min), float(x_max), float(default_point), key="point_x")
    a_col, b_col = st.columns(2)
    with a_col:
        sec_a = st.slider("Average-rate interval start a", float(x_min), float(x_max), float(x_min), key="sec_a")
    with b_col:
        sec_b = st.slider("Average-rate interval end b", float(x_min), float(x_max), float(x_max), key="sec_b")
    speed = st.select_slider("Playback speed", options=["Slow","Normal","Fast"], value="Normal")
    pcols = st.columns(4)
    with pcols[0]:
        if st.button("▶ Play"):
            st.session_state.animation_playing = True
            st.session_state.animation_x = float(point_x)
    with pcols[1]:
        if st.button("⏸ Pause"):
            st.session_state.animation_playing = False
    with pcols[2]:
        if st.button("↺ Reset"):
            st.session_state.animation_playing = False
            st.session_state.animation_x = float(x_min)
            st.rerun()
    with pcols[3]:
        if st.session_state.animation_playing:
            st.caption("Playback active")
        else:
            st.caption("Playback paused")
    if st.session_state.animation_playing and st_autorefresh is not None:
        interval_ms = {"Slow":900,"Normal":450,"Fast":180}[speed]
        st_autorefresh(interval=interval_ms, key="function_playback_refresh")
        span = x_max - x_min
        step_size = span / 50 * {"Slow":0.5,"Normal":1,"Fast":2}[speed]
        st.session_state.animation_x = x_min if st.session_state.animation_x >= x_max else st.session_state.animation_x + step_size
        if st.session_state.animation_x >= x_max:
            st.session_state.animation_x = x_min
        point_x = float(st.session_state.animation_x)
    y_point = safe_float(expr, point_x)
    slope_point = safe_float(f1, point_x)
    accel_point = safe_float(f2, point_x)
    ya, yb = safe_float(expr, sec_a), safe_float(expr, sec_b)
    avg_rate = (yb-ya)/(sec_b-sec_a) if sec_b != sec_a and np.isfinite(ya) and np.isfinite(yb) else np.nan
    metric_cols = st.columns(4)
    metric_cols[0].metric(f"f({point_x:.4g})", fmt(y_point), y_unit)
    metric_cols[1].metric("Instantaneous rate f′(x)", fmt(slope_point), f"{y_unit}/{x_unit}")
    metric_cols[2].metric("Second derivative f″(x)", fmt(accel_point), f"{y_unit}/{x_unit}²")
    metric_cols[3].metric("Average rate of change", fmt(avg_rate), f"{y_unit}/{x_unit}")
    if sec_a == sec_b:
        st.info("Choose distinct endpoints a and b to define a secant slope.")
    if default_ctx["kind"] == "motion":
        interpretation = f"At x = {point_x:.4g} {x_unit}, the modeled position is {fmt(y_point)} {y_unit}; f′ gives instantaneous velocity ({y_unit}/{x_unit}), and f″ gives acceleration ({y_unit}/{x_unit}²)."
    elif default_ctx["kind"] == "concentration":
        interpretation = f"At x = {point_x:.4g} {x_unit}, concentration is {fmt(y_point)} {y_unit}; f′ describes the concentration's rate of increase/decrease ({y_unit}/{x_unit}), and f″ describes whether that rate is changing ({y_unit}/{x_unit}²)."
    elif default_ctx["kind"] == "population":
        interpretation = f"At x = {point_x:.4g} {x_unit}, the modeled amount is {fmt(y_point)} {y_unit}; f′ is the growth rate and f″ indicates whether growth is speeding up or slowing down."
    elif default_ctx["kind"] == "economic":
        trend = "increasing" if slope_point > 0 else ("decreasing" if slope_point < 0 else "locally flat")
        interpretation = f"At x = {point_x:.4g} {x_unit}, {y_name} is {fmt(y_point)} {y_unit}; the modeled value is {trend}. f′ measures change per {x_unit}; f″ indicates whether that rate is rising or falling."
    elif default_ctx["kind"] == "temperature":
        interpretation = f"At x = {point_x:.4g} {x_unit}, temperature is {fmt(y_point)} {y_unit}; f′ is the warming/cooling rate and f″ shows how that rate changes."
    else:
        interpretation = f"At x = {point_x:.4g} {x_unit}, f(x) = {fmt(y_point)} {y_unit}; f′ gives the instantaneous change per {x_unit}, and f″ gives the change in that rate."
    st.info(interpretation)
    st.caption("Visual simulation scaling: the selected function value is mapped to a display position or indicator level; the graphic is illustrative, not a physically validated simulator.")
    # Graph and tangent/secant.
    gx = np.linspace(float(x_min), float(x_max), int(sample_count))
    gy = eval_array(expr, gx)
    figf = go.Figure()
    figf.add_trace(go.Scatter(x=gx, y=gy, mode="lines", name="f(x)", line=dict(width=3)))
    if show_d1:
        figf.add_trace(go.Scatter(x=gx, y=eval_array(f1,gx), mode="lines", name="f′(x)", line=dict(dash="dash")))
    if show_d2:
        figf.add_trace(go.Scatter(x=gx, y=eval_array(f2,gx), mode="lines", name="f″(x)", line=dict(dash="dot")))
    if np.isfinite(y_point):
        figf.add_trace(go.Scatter(x=[point_x], y=[y_point], mode="markers", name="Current point", marker=dict(size=12, symbol="circle")))
    if np.isfinite(slope_point) and np.isfinite(y_point):
        tx = np.array([max(x_min,point_x-(x_max-x_min)*0.12), min(x_max,point_x+(x_max-x_min)*0.12)])
        ty = y_point + slope_point*(tx-point_x)
        figf.add_trace(go.Scatter(x=tx,y=ty,mode="lines",name="Tangent line",line=dict(dash="dashdot",width=2)))
    if sec_a != sec_b and np.isfinite(ya) and np.isfinite(yb):
        sx = np.array([sec_a,sec_b])
        sy = np.array([ya,yb])
        figf.add_trace(go.Scatter(x=sx,y=sy,mode="lines+markers",name="Secant line",line=dict(width=2,dash="longdash")))
    figf.update_layout(height=500,xaxis_title=f"{x_name} ({x_unit})",yaxis_title=f"{y_name} ({y_unit})",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0))
    st.plotly_chart(figf,use_container_width=True)

    # Visual simulation responding to function value (simple, safe schematic)
    vcol1, vcol2 = st.columns([1,2])
    with vcol1:
        st.markdown("#### Context simulation")
        normalized = 0.5
        finite_y = gy[np.isfinite(gy)]
        if len(finite_y) and np.isfinite(y_point) and np.nanmax(finite_y) != np.nanmin(finite_y):
            normalized = float(np.clip((y_point-np.nanmin(finite_y))/(np.nanmax(finite_y)-np.nanmin(finite_y)),0,1))
        if visual == "Thermometer / weather indicator":
            st.write(f"🌡️ **Indicator:** {normalized*100:.0f}% of displayed model span")
            st.progress(normalized)
        elif visual == "Tank / drug concentration":
            st.write(f"🧪 **Tank level:** {normalized*100:.0f}% of displayed model span")
            st.progress(normalized)
        elif visual == "Population / cell colony":
            count = int(5 + normalized*35)
            st.write("🔬 " + " ".join(["●"]*count))
            st.caption(f"Schematic colony size: {count} dots")
        elif visual == "Business / economic indicator":
            st.write(f"📊 **Index indicator:** {normalized*100:.0f}% of displayed model span")
            st.progress(normalized)
        elif visual == "Walking person with prosthetic leg":
            angle = -25 + normalized*50
            st.write("🚶")
            st.write(f"Illustrative gait phase: {angle:.0f}° (scaled from f(x))")
            st.progress(normalized)
        else:
            st.write("🛰️")
            st.write(f"Moving-object display coordinate: {normalized:.3f} of the plotted vertical span")
            st.progress(normalized)
    with vcol2:
        st.markdown("#### Mathematical analysis")
        if course == "Honors Algebra 2":
            st.write(f"**Domain:** the selected interval is [{x_min:g}, {x_max:g}], subject to restrictions above. **Range on interval:** estimated from valid graph samples as {fmt(np.nanmin(gy))} to {fmt(np.nanmax(gy))}.")
            st.write("**End behavior / trend:** use the graph and function family. Polynomial end behavior depends on degree parity and leading coefficient; rational functions can have asymptotes; exponential and logarithmic models have restricted behavior.")
            st.write(f"**Average rate of change:** {fmt(avg_rate)} {y_unit}/{x_unit} over [{sec_a:g}, {sec_b:g}].")
        elif course == "Precalculus":
            st.write(f"**Domain/range on selected interval:** [{x_min:g}, {x_max:g}] subject to restrictions; sampled range ≈ [{fmt(np.nanmin(gy))}, {fmt(np.nanmax(gy))}] {y_unit}.")
            st.write(f"**Average rate:** {fmt(avg_rate)} {y_unit}/{x_unit}. Compare secant slopes on adjacent intervals: increasing slopes suggest concave-up behavior; decreasing slopes suggest concave-down behavior.")
            st.write("**Degree and coefficients:** polynomial degree bounds turning points by degree − 1; coefficient signs and sizes change vertical scale, orientation, and roots. This model's turning points are approximated below.")
        else:
            st.write("Calculus mode includes numerical sign tests; candidates are not automatically labeled as maxima, minima, or inflection points.")
            st.write(f"**Secant slope:** ({fmt(yb)} − {fmt(ya)}) / ({sec_b:g} − {sec_a:g}) = {fmt(avg_rate)} {y_unit}/{x_unit}.")
            st.write(f"**Tangent slope:** f′({point_x:.4g}) ≈ {fmt(slope_point)} {y_unit}/{x_unit}.")
    if course in ["Precalculus","Calculus"]:
        f1a, f2a, critical_rows, inflection_rows, abs_rows = derivative_analysis(expr, float(x_min), float(x_max))
        if critical_rows:
            st.markdown("**Critical-point classification**")
            st.dataframe(pd.DataFrame(critical_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No finite critical candidates were found symbolically in the selected interval; numerical methods may miss complicated roots.")
        if inflection_rows:
            st.markdown("**Verified inflection candidates**")
            st.dataframe(pd.DataFrame(inflection_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No verified inflection points found in the selected interval. A zero of f″ alone is not enough; concavity must change.")
        if abs_rows:
            st.markdown("**Absolute extrema on the selected closed interval (numerical grid estimate)**")
            st.dataframe(pd.DataFrame(abs_rows), use_container_width=True, hide_index=True)
            st.caption("Numerical grid estimates are approximate and can miss narrow extrema or discontinuities. A rigorous absolute-extrema claim requires continuity on the closed interval and checking endpoints and all critical points.")
        # Trend intervals from derivative sign sampling
        mids = np.linspace(x_min, x_max, 21)
        deriv = eval_array(f1, mids)
        signs = ["increasing" if np.isfinite(v) and v>1e-7 else ("decreasing" if np.isfinite(v) and v < -1e-7 else "flat/undefined") for v in deriv]
        st.write("**Sampled increasing/decreasing behavior:** " + ", ".join(f"{a:.3g}: {s}" for a,s in zip(mids,signs)))
    # Export function lab data
    export_df = pd.DataFrame({"x":gx,"f(x)":gy})
    if show_d1: export_df["f'(x)"] = eval_array(f1,gx)
    if show_d2: export_df["f''(x)"] = eval_array(f2,gx)
    st.download_button("Download function samples (CSV)", export_df.to_csv(index=False).encode("utf-8"), file_name="function_samples.csv", mime="text/csv")

# -------------------------------- Exports -------------------------------------
if fit_available:
    st.divider()
    st.header("Export current historical model")
    result_df = st.session_state.calc_results if st.session_state.calc_results is not None and st.session_state.last_calc_signature == st.session_state.last_fit_signature else pd.DataFrame(columns=["year","estimate","units","actual_observation","status","observation_status"])
    summary_rows = [
        ["Field","Value"],
        ["Generated at", datetime.now().isoformat(timespec="seconds")],
        ["Source", source],
        ["Country", country[0] if source == "World Bank historical data" else "n/a"],
        ["Indicator", indicator_label if source == "World Bank historical data" else "editable/sample data"],
        ["Base year", fit_summary["base_year"]],
        ["Degree", degree],
        ["Equation", "f(year) = " + equation],
        ["Observation count", fit_summary["n"]],
        ["Historical range", f"{fit_summary['start']}–{fit_summary['end']}"],
        ["Observed mean", fit_summary["mean"]],
        ["R-squared (in-sample)", fit_summary["r2"]],
        ["MAE (in-sample)", fit_summary["mae"]],
        ["Holdout MAE (chronological; separate)", fit_summary.get("holdout_mae","not available")],
        ["Data status", "Current selection; editable data reflected where applicable"],
        ["Limitations", "MAE is not a prediction interval. Polynomial extrapolation can be implausible."],
    ]
    xlsx_bytes = make_xlsx(summary_rows, fit_data, result_df)
    st.download_button("Download Excel workbook (.xlsx)", xlsx_bytes, file_name="hispanic_heritage_model.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download working data + results (CSV)", fit_data.to_csv(index=False).encode("utf-8"), file_name="working_data.csv", mime="text/csv")
    chart_bytes = None
    try:
        chart_bytes = fig.to_image(format="png", width=1300, height=700, scale=1)
    except Exception:
        pass
    summary_lines = [
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source: {source}; country: {country[0] if source == 'World Bank historical data' else 'n/a'}; indicator: {indicator_label if source == 'World Bank historical data' else 'sample/editable data'}",
        f"Base year: {fit_summary['base_year']}; degree: {degree}; observations: {fit_summary['n']}; historical range: {fit_summary['start']}–{fit_summary['end']}",
        f"Equation: f(year) = {equation}",
        f"R-squared (in-sample): {fit_summary['r2']}; MAE (in-sample): {fit_summary['mae']}; observed mean: {fit_summary['mean']}",
        f"Residuals (observed − fitted): {', '.join(fmt(v,5) for v in fit_data['residual'].to_list())}",
        f"Limitations: MAE is not a validated 95% prediction interval. Polynomial extrapolation may be implausible. No historical cause is inferred from correlation or curve shape.",
    ]
    pdf_bytes = make_pdf("Hispanic Heritage App 2.0 — Historical Model Report", summary_lines, chart_bytes, fit_data, result_df)
    st.download_button("Download PDF report", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
    if chart_bytes is None:
        st.caption("PDF chart export may be omitted if Plotly image rendering is unavailable on this environment; the interactive chart remains visible in the app.")

with st.expander("Printer-friendly report panel"):
    st.markdown("### Function Lab report")
    st.write(f"Course: {course} · Context: {context} · Function: {expression_text if 'expression_text' in locals() else 'not available'}")
    st.write(f"Analysis interval: {x_min if 'x_min' in locals() else '—'} to {x_max if 'x_max' in locals() else '—'}")
    if function_ok if 'function_ok' in locals() else False:
        st.write(f"At selected x = {point_x:g}: f(x) = {fmt(y_point)} {y_unit}; f′(x) = {fmt(slope_point)} {y_unit}/{x_unit}; f″(x) = {fmt(accel_point)} {y_unit}/{x_unit}².")
    if fit_available:
        st.write(f"Historical fit: degree {degree}, base year {fit_summary['base_year']}, R² (in-sample) {fmt(fit_summary['r2'])}, MAE (in-sample) {fmt(fit_summary['mae'])}.")
    st.write("Limitations: models summarize the selected data and assumptions; extrapolations are uncertain; no causal claims are made.")
    st.caption("Use your browser's Print command to print this simplified panel.")
