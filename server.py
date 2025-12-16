from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from io import StringIO

app = Flask(__name__)

# ------------------------------------------------------------
# Core KPI calculator (DataFrame in -> DataFrame out)
# Output:
#   - index  : KPI names (fixed order)
#   - columns: each date (YYYY-MM-DD) + WEEK_AVG
# ------------------------------------------------------------
def compute_weekly_kpis_from_df(
    df: pd.DataFrame,
    minutes_per_day: int = 480,
    fixed_rooms: int = 10,
    fcot_threshold_mins: int = 5,
    add_exit_or_plus1min_for_turnover: bool = True,
) -> pd.DataFrame:

    KPI_ORDER = [
        "Opened Utilization (%)",
        "Fixed Utilization (%)",
        "Turnover (mins)",
        "Wheels Out to OR Ready (mins)",
        "Patient Throughput",
        "Total Patient",
        "FCOT (%)",
        "Patient in to Start (mins)",
        "Physician Utilization (%)",
        "Physician Turnover (mins)",
        "Physician Daily OR Usage (mins)",
        "Patient Journey Time (mins)",
        "WR Time (mins)",
        "PREOP Time (mins)",
        "OR Time (mins)",
        "Recovery Time(mins)",
    ]

    KPI_FORMAT = {
        "Opened Utilization (%)":        lambda x: round(x, 2),
        "Fixed Utilization (%)":         lambda x: round(x, 2),
        "Turnover (mins)":               lambda x: round(x, 0),
        "Wheels Out to OR Ready (mins)": lambda x: round(x, 0),
        "Patient Throughput":            lambda x: round(x, 3),
        "Total Patient":                 lambda x: int(round(x)),
        "FCOT (%)":                      lambda x: round(x, 0),
        "Patient in to Start (mins)":    lambda x: round(x, 0),
        "Physician Utilization (%)":     lambda x: round(x, 1),
        "Physician Turnover (mins)":     lambda x: round(x, 0),
        "Physician Daily OR Usage (mins)": lambda x: round(x, 0),
        "Patient Journey Time (mins)":   lambda x: round(x, 0),
        "WR Time (mins)":                lambda x: round(x, 0),
        "PREOP Time (mins)":             lambda x: round(x, 0),
        "OR Time (mins)":                lambda x: round(x, 0),
        "Recovery Time(mins)":           lambda x: round(x, 0),
    }

    # -------- Helpers --------
    def _as_series(x):
        return x if isinstance(x, pd.Series) else pd.Series(x)

    def to_dt(s: pd.Series) -> pd.Series:
        """Safe datetime conversion; treats '-' as NaN."""
        s = _as_series(s).astype("string")
        s = s.replace("-", pd.NA)
        return pd.to_datetime(s, errors="coerce")

    def mins_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
        return (later - earlier).dt.total_seconds() / 60

    def safe_col(name: str) -> pd.Series:
        return df_day[name] if name in df_day.columns else pd.Series([pd.NA] * len(df_day))

    # -------- Choose date column --------
    if "Room / Date of Visit" in df.columns:
        date_col = "Room / Date of Visit"
    elif "Scheduled Date" in df.columns:
        date_col = "Scheduled Date"
    else:
        raise ValueError("No date column found. Need 'Room / Date of Visit' or 'Scheduled Date'.")

    df = df.copy()
    df["_date_key"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df.dropna(subset=["_date_key"]).copy()

    # -------- One-day KPI computation --------
    def compute_kpis_for_day(df_day: pd.DataFrame) -> dict:
        # ---------- OR basic ----------
        df_or = df_day[["Operation Room", "Entered OR", "Exited OR"]].copy() \
            if all(c in df_day.columns for c in ["Operation Room", "Entered OR", "Exited OR"]) \
            else pd.DataFrame(columns=["Operation Room", "Entered OR", "Exited OR"])

        df_or = df_or.dropna(subset=["Operation Room"]).copy()
        df_or["Entered OR"] = to_dt(df_or["Entered OR"])
        df_or["Exited OR"]  = to_dt(df_or["Exited OR"])

        df_or["or_time"] = mins_diff(df_or["Exited OR"], df_or["Entered OR"])
        df_or.loc[df_or["or_time"] < 0, "or_time"] = np.nan  # ignore negative per cell
        total_OR_time = float(df_or["or_time"].sum(skipna=True))

        opened_room = int(df_or["Operation Room"].nunique()) if len(df_or) else 0

        # ---------- OR Turnover (positive only) ----------
        Turnover = np.nan
        if len(df_or):
            df_to = df_or[["Operation Room", "Entered OR", "Exited OR"]].copy()
            if add_exit_or_plus1min_for_turnover:
                df_to["Exited OR"] = df_to["Exited OR"] + pd.Timedelta(minutes=1)
            df_to = df_to.sort_values(["Operation Room", "Entered OR"])
            df_to["next_entered"] = df_to.groupby("Operation Room")["Entered OR"].shift(-1)
            df_to["turnover"] = mins_diff(df_to["next_entered"], df_to["Exited OR"])
            df_to = df_to[df_to["turnover"] > 0].copy()
            Turnover = round(df_to["turnover"].mean(), 0) if len(df_to) else np.nan

        # ---------- Wheels Out -> OR Ready Next Patient ----------
        Wheels_out_to_OR_Ready = np.nan
        if all(c in df_day.columns for c in ["Exited OR", "OR Room Ready Next Patient"]):
            tmp = df_day[["Exited OR", "OR Room Ready Next Patient"]].copy()
            tmp["Exited OR"] = to_dt(tmp["Exited OR"])
            tmp["OR Room Ready Next Patient"] = to_dt(tmp["OR Room Ready Next Patient"])
            tmp["w_out_to_ready"] = mins_diff(tmp["OR Room Ready Next Patient"], tmp["Exited OR"])
            # NOTE: if you want to ignore negatives here too, uncomment next line
            tmp.loc[tmp["w_out_to_ready"] < 0, "w_out_to_ready"] = np.nan
            Wheels_out_to_OR_Ready = int(round(tmp["w_out_to_ready"].mean())) if tmp["w_out_to_ready"].notna().any() else np.nan

        # ---------- OR Cycle (matches your original logic + negative ignored per cell) ----------
        Patient_Throughput = np.nan
        Patient_in_to_Start = np.nan
        if all(c in df_day.columns for c in [
            "Operation Room", "Entered OR", "Procedure Start", "Closing Start",
            "Procedure End", "Exited OR", "OR Room Ready"
        ]):
            df3 = df_day[[
                "Operation Room", "Entered OR", "Procedure Start", "Closing Start",
                "Procedure End", "Exited OR", "OR Room Ready"
            ]].copy()

            df3 = df3.dropna(subset=["Operation Room"]).reset_index(drop=True)
            df3["Operation Room"] = df3["Operation Room"].astype(str).str.extract(r"(\d+)")[0]
            df3 = df3.dropna(subset=["Operation Room"]).copy()
            df3["Operation Room"] = df3["Operation Room"].astype(int)

            col = ["Entered OR", "Procedure Start", "Closing Start", "Procedure End", "Exited OR", "OR Room Ready"]
            df3[col] = df3[col].apply(lambda x: pd.to_datetime(x, errors="coerce"))

            df3["pt_prepping"]  = (df3["Procedure Start"] - df3["Entered OR"]).dt.total_seconds() / 60
            df3["procedure"]    = (df3["Closing Start"] - df3["Procedure Start"]).dt.total_seconds() / 60
            df3["closing"]      = (df3["Procedure End"] - df3["Closing Start"]).dt.total_seconds() / 60
            df3["post_closing"] = (df3["Exited OR"] - df3["Procedure End"]).dt.total_seconds() / 60
            df3["cleaning"]     = (df3["OR Room Ready"] - df3["Exited OR"]).dt.total_seconds() / 60

            df3 = df3.sort_values(["Operation Room", "Entered OR"])
            df3["next_entered"] = df3.groupby("Operation Room")["Entered OR"].shift(-1)
            df3["or_prepping"]  = (df3["next_entered"] - df3["OR Room Ready"]).dt.total_seconds() / 60

            # ignore negative values per cell
            cycle_cols = ["pt_prepping", "procedure", "closing", "post_closing", "cleaning", "or_prepping"]
            for c in cycle_cols:
                df3.loc[df3[c] < 0, c] = np.nan

            pt_prepping  = np.floor(df3["pt_prepping"].mean())
            procedure    = np.floor(df3["procedure"].mean())
            closing      = np.floor(df3["closing"].mean())
            post_closing = np.floor(df3["post_closing"].mean())
            cleaning     = np.floor(df3["cleaning"].mean())
            or_prepping  = np.floor(df3["or_prepping"].mean())

            OR_Cycle = pt_prepping + procedure + closing + post_closing + cleaning + or_prepping
            Patient_Throughput = round(60 / OR_Cycle, 3) if OR_Cycle and not np.isnan(OR_Cycle) else np.nan
            Patient_in_to_Start = pt_prepping

        # ---------- FCOT ----------
        FCOT = np.nan
        if all(c in df_day.columns for c in ["Physician", "Scheduled Date", "Entered OR"]):
            df5 = df_day[["Physician", "Scheduled Date", "Entered OR"]].copy()
            df5["Scheduled Date"] = to_dt(df5["Scheduled Date"])
            df5["Entered OR"] = to_dt(df5["Entered OR"])
            df5 = df5.dropna(subset=["Physician", "Scheduled Date", "Entered OR"]).copy()
            if len(df5):
                df5 = df5.sort_values(["Physician", "Entered OR"])
                first_cases = df5.groupby("Physician", as_index=False).first()

                sched_t = pd.to_datetime(first_cases["Scheduled Date"].dt.time.astype(str), errors="coerce")
                ent_t   = pd.to_datetime(first_cases["Entered OR"].dt.time.astype(str), errors="coerce")
                diff = mins_diff(ent_t, sched_t)
                FCOT = ((diff <= fcot_threshold_mins).mean() * 100 if diff.notna().any() else np.nan).round()

        # ---------- Physician Utilization + Daily OR Usage ----------
        phy_util = np.nan
        Physician_Daily_OR_Usage = np.nan
        if all(c in df_day.columns for c in ["Physician", "Entered OR", "Exited OR"]):
            df6 = df_day[["Physician", "Entered OR", "Exited OR"]].copy()
            df6["Entered OR"] = to_dt(df6["Entered OR"])
            df6["Exited OR"]  = to_dt(df6["Exited OR"])
            df6 = df6.dropna(subset=["Physician", "Entered OR", "Exited OR"]).copy()
            if len(df6):
                df6["surgery_mins"] = mins_diff(df6["Exited OR"], df6["Entered OR"])
                df6.loc[df6["surgery_mins"] < 0, "surgery_mins"] = np.nan
                df6 = df6[df6["surgery_mins"].notna()].copy()

                grouped = df6.groupby("Physician").agg(
                    first_entered=("Entered OR", "min"),
                    last_exited=("Exited OR", "max"),
                    total_surgery_time=("surgery_mins", "sum"),
                )
                grouped["total_span_mins"] = mins_diff(grouped["last_exited"], grouped["first_entered"])
                grouped = grouped[grouped["total_span_mins"] > 0].copy()
                grouped["util"] = (grouped["total_surgery_time"] / grouped["total_span_mins"]) * 100

                phy_util = round(grouped["util"].mean(), 2) if len(grouped) else np.nan
                Physician_Daily_OR_Usage = int(grouped["total_surgery_time"].mean()) if len(grouped) else np.nan

        # ---------- Physician Turnover (negatives allowed) ----------
        Physician_Turnover_avg = np.nan
        if all(c in df_day.columns for c in ["Physician", "Procedure Start", "Procedure End"]):
            df_pto = df_day[["Physician", "Procedure Start", "Procedure End"]].copy()
            df_pto["Procedure Start"] = to_dt(df_pto["Procedure Start"])
            df_pto["Procedure End"]   = to_dt(df_pto["Procedure End"])
            df_pto = df_pto.dropna(subset=["Physician", "Procedure Start", "Procedure End"]).copy()
            if len(df_pto):
                df_pto = df_pto.sort_values(["Physician", "Procedure Start"])
                df_pto["next_procedure"] = df_pto.groupby("Physician")["Procedure Start"].shift(-1)
                df_pto["phy_turnover"] = mins_diff(df_pto["next_procedure"], df_pto["Procedure End"])
                Physician_Turnover_avg = np.floor(df_pto["phy_turnover"].mean()) if df_pto["phy_turnover"].notna().any() else np.nan

        # ---------- OR Time (mins) = Wheels In/Out Physician-based avg ----------
        WIW_Physician_Avg = np.nan
        if all(c in df_day.columns for c in ["Physician", "Entered OR", "Exited OR"]):
            tmp = df_day[["Physician", "Entered OR", "Exited OR"]].copy()
            tmp["Entered OR"] = to_dt(tmp["Entered OR"])
            tmp["Exited OR"]  = to_dt(tmp["Exited OR"])
            tmp = tmp.dropna(subset=["Physician", "Entered OR", "Exited OR"]).copy()
            if len(tmp):
                tmp["wiw_mins"] = mins_diff(tmp["Exited OR"], tmp["Entered OR"])
                tmp.loc[tmp["wiw_mins"] < 0, "wiw_mins"] = np.nan
                tmp = tmp[tmp["wiw_mins"].notna()].copy()
                wiw_by_phys = tmp.groupby("Physician")["wiw_mins"].mean()
                WIW_Physician_Avg = int(wiw_by_phys.mean()) if len(wiw_by_phys) else np.nan

        # ---------- Patient Journey / WR / PREOP / PACU ----------
        Patient_Journey_Time = np.nan
        if all(c in df_day.columns for c in ["Admitted", "Discharged"]):
            pj = df_day[["Admitted", "Discharged"]].copy()
            pj["Admitted"] = to_dt(pj["Admitted"])
            pj["Discharged"] = to_dt(pj["Discharged"])
            pj["journey"] = mins_diff(pj["Discharged"], pj["Admitted"])
            pj.loc[pj["journey"] < 0, "journey"] = np.nan
            Patient_Journey_Time = round(pj["journey"].mean(), 0) if pj["journey"].notna().any() else np.nan

        WR = np.nan
        if all(c in df_day.columns for c in ["Entered WR", "Exited WR"]):
            wr = pd.DataFrame({"in": to_dt(df_day["Entered WR"]), "out": to_dt(df_day["Exited WR"])})
            wr["v"] = mins_diff(wr["out"], wr["in"])
            wr.loc[wr["v"] < 0, "v"] = np.nan
            WR = round(wr["v"].mean(), 0) if wr["v"].notna().any() else np.nan

        PREOP = np.nan
        if all(c in df_day.columns for c in ["Entered Pre-Op", "Exited Pre-Op"]):
            pr = pd.DataFrame({"in": to_dt(df_day["Entered Pre-Op"]), "out": to_dt(df_day["Exited Pre-Op"])})
            pr["v"] = mins_diff(pr["out"], pr["in"])
            pr.loc[pr["v"] < 0, "v"] = np.nan
            PREOP = round(pr["v"].mean(), 0) if pr["v"].notna().any() else np.nan

        PACU = np.nan
        if all(c in df_day.columns for c in ["Entered Pacu", "Exited Pacu"]):
            pa = pd.DataFrame({"in": to_dt(df_day["Entered Pacu"]), "out": to_dt(df_day["Exited Pacu"])})
            pa["v"] = mins_diff(pa["out"], pa["in"])
            pa.loc[pa["v"] < 0, "v"] = np.nan
            PACU = round(pa["v"].mean(), 0) if pa["v"].notna().any() else np.nan

        # ---------- Utilization (per day) ----------
        Opened_Util = (
            round(total_OR_time / (opened_room * minutes_per_day) * 100, 2)
            if opened_room
            else np.nan
        )
        Fixed_Util = round(total_OR_time / (fixed_rooms * minutes_per_day) * 100, 2) if minutes_per_day else np.nan

        return {
            "Opened Utilization (%)": Opened_Util,
            "Fixed Utilization (%)": Fixed_Util,
            "Turnover (mins)": Turnover,
            "Wheels Out to OR Ready (mins)": Wheels_out_to_OR_Ready,
            "Patient Throughput": Patient_Throughput,
            "Total Patient": int(df_day.shape[0]),
            "FCOT (%)": FCOT,
            "Patient in to Start (mins)": Patient_in_to_Start,
            "Physician Utilization (%)": phy_util,
            "Physician Turnover (mins)": Physician_Turnover_avg,
            "Physician Daily OR Usage (mins)": Physician_Daily_OR_Usage,
            "Patient Journey Time (mins)": Patient_Journey_Time,
            "WR Time (mins)": WR,
            "PREOP Time (mins)": PREOP,
            "OR Time (mins)": WIW_Physician_Avg,
            "Recovery Time(mins)": PACU,
        }

    # -------- Compute per day --------
    daily_frames = []
    for day, df_day in df.groupby("_date_key"):
        day_str = str(day)
        kpi_dict = compute_kpis_for_day(df_day)
        day_df = pd.DataFrame.from_dict(kpi_dict, orient="index", columns=[day_str])
        daily_frames.append(day_df)

    daily_kpi_df = pd.concat(daily_frames, axis=1)

    # enforce requested order
    daily_kpi_df = daily_kpi_df.reindex(KPI_ORDER)

    # Calculate WEEK_AVG first (raw mean)
    daily_kpi_df["WEEK_AVG"] = daily_kpi_df.mean(axis=1, skipna=True)

    # Apply same formatting as daily values
    for kpi, fmt in KPI_FORMAT.items():
        if kpi in daily_kpi_df.index:
            daily_kpi_df.loc[kpi, "WEEK_AVG"] = fmt(daily_kpi_df.loc[kpi, "WEEK_AVG"])



    return daily_kpi_df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/kpi", methods=["POST"])
def api_kpi():
    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file'."}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file."}), 400

    try:
        df = pd.read_csv(f)
        kpi_df = compute_weekly_kpis_from_df(df)

        return jsonify({
            "index": kpi_df.index.tolist(),
            "columns": kpi_df.columns.tolist(),
            "data": kpi_df.values.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)