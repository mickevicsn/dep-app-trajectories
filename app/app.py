# app/app.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

# =========================
# HARD-CODED PATHS
# =========================
RAW_DIR = Path("data/raw_tracks_2")          # per-flight parquet (ts/lat/lon...)
RESAMPLED_DIR = Path("data/resampled_5s")    # per-flight parquet (dt_utc/lat/lon...)
RTS_DIR = Path("data/smoothed_rts")          # per-flight parquet (dt_utc + lat_rts/lon_rts)
METRICS_PATH = Path("data/flight_metrics.parquet")

MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

# =========================
# DISPLAY / SAFETY SETTINGS
# =========================
MAX_POINTS_ON_MAP = 1500     
MAX_GAP_S = 180              
MAX_JUMP_KM = 250            
MAX_TABLE_ROWS = 120


# -------------------------
# Basic helpers
# -------------------------
def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dt_utc" in df.columns:
        df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    elif "ts" in df.columns:
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df["dt_utc"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _list_flights() -> list[str]:
    base = RESAMPLED_DIR if RESAMPLED_DIR.exists() else RAW_DIR
    if not base.exists():
        return []
    return sorted([p.stem for p in base.glob("*.parquet")])


def _load_flight(folder: Path, flight_id: str) -> Optional[pd.DataFrame]:
    """
    Loads a parquet. If the parquet contains multiple flight_id values,
    it filters to the selected flight_id (this is a common reason for “map ribbon carpets”).
    """
    fp = folder / f"{flight_id}.parquet"
    if not fp.exists():
        return None

    df = pd.read_parquet(fp)
    df = _ensure_dt(df)
    df = _coerce_numeric(df, ["lat", "lon", "lat_rts", "lon_rts", "altitude", "altitude_rts"])

    # If there is a flight_id column and it contains multiple flights, filter.
    if "flight_id" in df.columns:
        df["flight_id"] = df["flight_id"].astype(str)
        if df["flight_id"].nunique(dropna=True) > 1:
            df = df[df["flight_id"] == flight_id].copy()

    df = df.dropna(subset=["dt_utc"]).sort_values("dt_utc").reset_index(drop=True)
    return df


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _clean_track(df: Optional[pd.DataFrame], lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Returns a clean, time-ordered track with dt_utc + lat/lon, and thinned for map display.
    """
    if df is None:
        return pd.DataFrame()

    if "dt_utc" not in df.columns:
        return pd.DataFrame()
    if lat_col not in df.columns or lon_col not in df.columns:
        return pd.DataFrame()

    g = df[["dt_utc", lat_col, lon_col]].copy()
    g["dt_utc"] = pd.to_datetime(g["dt_utc"], utc=True, errors="coerce")
    g[lat_col] = pd.to_numeric(g[lat_col], errors="coerce")
    g[lon_col] = pd.to_numeric(g[lon_col], errors="coerce")

    g = g.dropna(subset=["dt_utc", lat_col, lon_col])
    g = g[np.isfinite(g[lat_col]) & np.isfinite(g[lon_col])]

    # Critical: time order
    g = g.sort_values("dt_utc").reset_index(drop=True)

    # Drop duplicate timestamps (prevents weird “fan” artifacts)
    g = g.drop_duplicates(subset=["dt_utc"])

    # Thin for web map (prevents dense “ribbon carpet”)
    if len(g) > MAX_POINTS_ON_MAP:
        step = max(1, len(g) // MAX_POINTS_ON_MAP)
        g = g.iloc[::step].copy().reset_index(drop=True)

    return g


def _segments_for_line_layer(g: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Build safe segments:
      - don’t connect across time gaps (MAX_GAP_S)
      - don’t connect “teleport” jumps (MAX_JUMP_KM)
    """
    if g.empty or len(g) < 2:
        return pd.DataFrame()

    lon = g[lon_col].to_numpy()
    lat = g[lat_col].to_numpy()
    t = g["dt_utc"].astype("int64").to_numpy() / 1e9

    dt = np.diff(t)
    dkm = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])

    ok = (dt <= MAX_GAP_S) & (dkm <= MAX_JUMP_KM)

    seg = pd.DataFrame(
        {
            "from_lon": lon[:-1],
            "from_lat": lat[:-1],
            "to_lon": lon[1:],
            "to_lat": lat[1:],
        }
    )
    seg = seg[ok].reset_index(drop=True)
    return seg


def _line_layer_from_track(g: pd.DataFrame, lat_col: str, lon_col: str, name: str, color, width_m: int):
    seg = _segments_for_line_layer(g, lat_col, lon_col)
    if seg.empty:
        return None

    seg = seg.copy()
    seg["name"] = name

    return pdk.Layer(
        "LineLayer",
        data=seg,
        get_source_position="[from_lon, from_lat]",
        get_target_position="[to_lon, to_lat]",
        get_color=color,
        get_width=width_m,
        width_units="meters",       # IMPORTANT
        width_min_pixels=1,         # keep visible when zoomed out
        width_max_pixels=3,         # prevent fat lines when zoomed in
        pickable=True,
    )



def _points_layer_from_track(g: pd.DataFrame, lat_col: str, lon_col: str, name: str, color, radius_px: int = 2):
    if g.empty:
        return None

    pts = g.rename(columns={lon_col: "lon", lat_col: "lat"}).copy()
    pts["name"] = name

    return pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position="[lon, lat]",
        get_fill_color=color,
        get_radius=radius_px,
        radius_units="pixels",
        pickable=True,
        stroked=False,
    )


def _view_state_from_track(g: pd.DataFrame, lat_col: str, lon_col: str) -> pdk.ViewState:
    if g.empty:
        return pdk.ViewState(latitude=56.95, longitude=24.10, zoom=5.5, pitch=0)
    lat = float(g[lat_col].median())
    lon = float(g[lon_col].median())
    return pdk.ViewState(latitude=lat, longitude=lon, zoom=5.5, pitch=0)


@st.cache_data(show_spinner=False)
def _load_metrics() -> Optional[pd.DataFrame]:
    if not METRICS_PATH.exists():
        return None
    return pd.read_parquet(METRICS_PATH)


def _safe_kv_table(row: pd.Series) -> pd.DataFrame:
    kv = row.to_dict()
    rows = []
    for k, v in kv.items():
        if isinstance(v, (dict, list, tuple, set)):
            v = str(v)
        try:
            if pd.isna(v):
                v = None
        except Exception:
            pass
        if not (isinstance(v, (int, float, str, bool)) or v is None):
            v = str(v)
        rows.append({"metric": str(k), "value": v})
    return pd.DataFrame(rows)


def _fallback_lonlat_plot(raw_g, res_g, rts_g):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    drawn = 0

    if not raw_g.empty:
        ax.plot(raw_g["lon"], raw_g["lat"], label="Raw")
        drawn += 1
    if not res_g.empty:
        ax.plot(res_g["lon"], res_g["lat"], label="Resampled (5s)")
        drawn += 1
    if not rts_g.empty:
        ax.plot(rts_g["lon"], rts_g["lat"], label="RTS Smoothed")
        drawn += 1

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Trajectory overlay (fallback lon/lat)")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    if drawn:
        ax.legend()
    return fig


# =========================
# APP
# =========================
def main():
    st.set_page_config(page_title="Trajectory Quality Dashboard", layout="wide")
    st.title("Aircraft Trajectory Quality Dashboard")
    st.caption("Inspect Raw vs 5s Resampled vs RTS-smoothed trajectories + metrics (MO4 data product).")

    flights = _list_flights()
    if not flights:
        st.error("No flight parquet files found. Check your data folders.")
        st.stop()

    with st.sidebar:
        st.header("Data folders")
        st.write(f"Raw: `{RAW_DIR}`")
        st.write(f"Resampled: `{RESAMPLED_DIR}`")
        st.write(f"RTS: `{RTS_DIR}`")
        st.write(f"Metrics: `{METRICS_PATH}`")

        st.divider()
        flight_id = st.selectbox("Select flight", flights, index=0)

        st.divider()
        show_raw = st.checkbox("Show Raw", True)
        show_res = st.checkbox("Show Resampled (5s)", True)
        show_rts = st.checkbox("Show RTS Smoothed", True)

        # Points OFF by default to avoid “ribbon carpet”
        show_points = st.checkbox("Show points (debug)", False)

    raw_df = _load_flight(RAW_DIR, flight_id) if show_raw else None
    res_df = _load_flight(RESAMPLED_DIR, flight_id) if show_res else None
    rts_df = _load_flight(RTS_DIR, flight_id) if show_rts else None

    # Build clean tracks with consistent columns for plotting/map
    raw_g = _clean_track(raw_df, "lat", "lon")
    res_g = _clean_track(res_df, "lat", "lon")

    # RTS: prefer lat_rts/lon_rts; fall back to lat/lon if needed
    if rts_df is not None and ("lat_rts" in rts_df.columns and "lon_rts" in rts_df.columns):
        rts_g0 = _clean_track(rts_df, "lat_rts", "lon_rts")
        rts_g = rts_g0.rename(columns={"lat_rts": "lat", "lon_rts": "lon"})
        rts_lat_col, rts_lon_col = "lat", "lon"
    else:
        rts_g = _clean_track(rts_df, "lat", "lon")
        rts_lat_col, rts_lon_col = "lat", "lon"

    # Sidebar debug so you can verify it’s truly one flight / not insane density
    with st.sidebar:
        st.divider()
        st.subheader("Debug")
        st.write(f"Raw points on map: {len(raw_g)}")
        st.write(f"Res points on map: {len(res_g)}")
        st.write(f"RTS points on map: {len(rts_g)}")
        if rts_df is not None and "flight_id" in rts_df.columns:
            st.write(f"RTS file flight_id nunique: {rts_df['flight_id'].astype(str).nunique(dropna=True)}")

    col_map, col_info = st.columns([2, 1], gap="large")

    with col_map:
        st.subheader("Trajectory overlay map")

        # Choose view based on best available track
        if not rts_g.empty:
            view = _view_state_from_track(rts_g, "lat", "lon")
        elif not res_g.empty:
            view = _view_state_from_track(res_g, "lat", "lon")
        elif not raw_g.empty:
            view = _view_state_from_track(raw_g, "lat", "lon")
        else:
            st.warning("No valid coordinates found for selected layers.")
            st.stop()

        layers = []

        # POINTS FIRST (optional), LINES LAST (lines remain visible)
        if show_points:
            if not raw_g.empty:
                pt = _points_layer_from_track(raw_g, "lat", "lon", "Raw pts", [255, 60, 60, 120], radius_px=2)
                if pt is not None:
                    layers.append(pt)
            if not res_g.empty:
                pt = _points_layer_from_track(res_g, "lat", "lon", "Resampled pts", [80, 140, 255, 120], radius_px=2)
                if pt is not None:
                    layers.append(pt)
            if not rts_g.empty:
                pt = _points_layer_from_track(rts_g, "lat", "lon", "RTS pts", [60, 220, 120, 120], radius_px=2)
                if pt is not None:
                    layers.append(pt)

        # LINES (these should look like a single track, not a ribbon)
        if not raw_g.empty:
            lyr = _line_layer_from_track(raw_g, "lat", "lon", "Raw", [255, 60, 60, 230], 200)
            if lyr is not None:
                layers.append(lyr)

        if not res_g.empty:
            lyr = _line_layer_from_track(res_g, "lat", "lon", "Resampled (5s)", [80, 140, 255, 240], 250)
            if lyr is not None:
                layers.append(lyr)

        if not rts_g.empty:
            lyr = _line_layer_from_track(rts_g, "lat", "lon", "RTS Smoothed", [60, 220, 120, 255], 300)
            if lyr is not None:
                layers.append(lyr)


            deck = pdk.Deck(
                layers=layers,
                initial_view_state=view,
                map_style=MAP_STYLE,
                tooltip={"text": "{name}"},
            )
            st.pydeck_chart(deck, use_container_width=True)

            st.markdown("**Legend:** 🟥 Raw | 🟦 Resampled (5s) | 🟩 RTS Smoothed")

            st.subheader("Fallback plot (always correct)")
            fig = _fallback_lonlat_plot(raw_g, res_g, rts_g)
            st.pyplot(fig, use_container_width=True)

    with col_info:
        st.subheader("Metrics (datamart)")
        metrics = _load_metrics()
        if metrics is None:
            st.info("No metrics parquet found yet.")
        else:
            if "flight_id" in metrics.columns:
                row = metrics[metrics["flight_id"].astype(str) == str(flight_id)]
                if row.empty:
                    st.info("No metrics for this flight_id.")
                else:
                    kv_df = _safe_kv_table(row.iloc[0])
                    st.dataframe(kv_df.head(MAX_TABLE_ROWS), width="stretch", hide_index=True)
            else:
                st.warning("Metrics parquet exists but has no 'flight_id' column.")

        st.subheader("Download cleaned output")
        if not rts_g.empty:
            export = rts_g[["dt_utc", "lat", "lon"]].copy()
            st.download_button(
                "Download RTS-smoothed CSV",
                data=export.to_csv(index=False).encode("utf-8"),
                file_name=f"{flight_id}_rts.csv",
                mime="text/csv",
            )
        elif not res_g.empty:
            export = res_g[["dt_utc", "lat", "lon"]].copy()
            st.download_button(
                "Download resampled CSV",
                data=export.to_csv(index=False).encode("utf-8"),
                file_name=f"{flight_id}_resampled.csv",
                mime="text/csv",
            )
        elif not raw_g.empty:
            export = raw_g[["dt_utc", "lat", "lon"]].copy()
            st.download_button(
                "Download raw CSV",
                data=export.to_csv(index=False).encode("utf-8"),
                file_name=f"{flight_id}_raw.csv",
                mime="text/csv",
            )
        else:
            st.warning("Nothing to download for this flight.")


if __name__ == "__main__":
    main()
