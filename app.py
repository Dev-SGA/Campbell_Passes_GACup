import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Pass Map Dashboard (Interactive)")

# ==========================
# Configuration
# ==========================
MINUTES_PLAYED = 100

st.title("Pass Map Dashboard")
st.caption(f"⏱️ {MINUTES_PLAYED} minutes played — Click the start dot to select the pass event.")

FINAL_THIRD_LINE_X = 80

BOX_X_MIN = 102
BOX_Y_MIN = 18
BOX_Y_MAX = 62

GOAL_X = 120
GOAL_Y = 40

# Limiares oficiais Opta para passes progressivos
PROG_OWN_HALF_THRESHOLD = 30
PROG_CROSS_HALF_THRESHOLD = 15
PROG_OPP_HALF_THRESHOLD = 10

# Cores ajustadas: sucesso mais claro, falha mais escura
COLOR_SUCCESS = "#B0B0B0"
COLOR_FAIL = "#D45B5B"
COLOR_PROGRESSIVE = "#2F80ED"

# ==========================
# DATA
# ==========================
matches_data = {
    "Vs Dallas": [
        ("PASS WON", 40.88, 68.26, 55.35, 70.09, None),
        ("PASS WON", 64.49, 73.08, 77.95, 66.43, None),
        ("PASS WON", 64.99, 69.76, 69.48, 58.62, None),
        ("PASS WON", 66.98, 54.13, 50.36, 52.97, None),
        ("PASS WON", 91.75, 71.09, 82.94, 56.29, None),
        ("PASS LOST", 116.69, 11.74, 110.54, 33.18, None),
        ("PASS LOST", 57.01, 73.58, 112.03, 62.28, None),
        ("PASS LOST", 66.15, 69.76, 93.91, 64.27, None),
    ],
    "Vs Nagoya": [
        ("PASS WON", 14.45, 68.26, 9.30, 55.79, None),
        ("PASS WON", 16.95, 64.10, 11.29, 50.47, None),
        ("PASS WON", 12.46, 68.76, 29.58, 78.57, None),
        ("PASS WON", 27.42, 52.14, 22.26, 42.83, None),
        ("PASS WON", 49.36, 63.61, 67.65, 67.26, None),
        ("PASS WON", 99.56, 73.41, 110.87, 65.93, None),
        ("PASS WON", 27.92, 65.27, 33.40, 75.24, None),
        ("PASS WON", 27.92, 71.09, 43.04, 75.41, None),
        ("PASS WON", 25.59, 67.43, 37.06, 68.92, None),
        ("PASS WON", 32.07, 70.25, 23.93, 55.96, None),
        ("PASS WON", 29.41, 72.42, 38.39, 75.74, None),
        ("PASS WON", 31.41, 70.75, 26.42, 62.77, None),
        ("PASS WON", 32.07, 77.40, 41.88, 70.25, None),
        ("PASS WON", 33.07, 70.92, 54.35, 47.31, None),
        ("PASS LOST", 12.12, 77.74, 19.61, 77.90, None),
        ("PASS LOST", 14.95, 73.08, 25.92, 70.75, None),
        ("PASS LOST", 53.35, 63.61, 55.35, 53.63, None),
        ("PASS LOST", 55.01, 60.45, 41.55, 43.32, None),
        ("PASS LOST", 58.17, 72.08, 100.39, 38.84, None),
    ],
    "Vs Busan Park": [
        ("PASS WON", 23.76, 77.82, 33.74, 72.66, None),
        ("PASS LOST", 56.84, 74.74, 81.11, 57.95, None),
    ],
    "Vs Atlanta": [
        ("PASS WON", 13.95, 72.75, 18.61, 76.41, None),
        ("PASS WON", 18.61, 71.58, 27.92, 75.41, None),
        ("PASS WON", 32.57, 66.76, 46.20, 72.58, None),
        ("PASS WON", 51.85, 73.58, 44.04, 39.83, None),
        ("PASS WON", 50.52, 71.09, 43.71, 46.32, None),
        ("PASS WON", 74.13, 50.81, 86.43, 53.13, None),
        ("PASS WON", 98.73, 71.75, 114.69, 71.09, None),
        ("PASS WON", 118.51, 63.61, 96.74, 39.83, None),
        ("PASS WON", 93.75, 72.91, 98.57, 68.92, None),
        ("PASS WON", 76.62, 64.60, 89.92, 77.74, None),
        ("PASS WON", 70.64, 65.93, 89.09, 73.25, None),
        ("PASS WON", 66.82, 70.75, 85.43, 68.26, None),
        ("PASS LOST", 107.54, 69.92, 108.54, 39.34, None),
    ],
}

# ==========================
# Helpers
# ==========================
def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""


def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def is_progressive_pass(x_start, y_start, x_end, y_end) -> bool:
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    gain = start_dist - end_dist

    start_own_half = x_start < 60
    end_own_half = x_end < 60
    end_opp_half = x_end >= 60
    start_opp_half = x_start >= 60

    if start_own_half and end_own_half:
        return gain >= PROG_OWN_HALF_THRESHOLD
    elif start_own_half and end_opp_half:
        return gain >= PROG_CROSS_HALF_THRESHOLD
    elif start_opp_half and end_opp_half:
        return gain >= PROG_OPP_HALF_THRESHOLD
    else:
        return False


def per90(value: int, minutes: int) -> str:
    """Retorna o valor normalizado por 90 minutos, formatado."""
    if minutes <= 0:
        return "—"
    result = value / minutes * 90
    if result == int(result):
        return f"{int(result)}"
    return f"{result:.1f}"


# ==========================
# Build DataFrames
# ==========================
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(
        events,
        columns=["type", "x_start", "y_start", "x_end", "y_end", "video"],
    )
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["progressive"] = dfm.apply(
        lambda row: is_progressive_pass(
            row["x_start"], row["y_start"], row["x_end"], row["y_end"]
        ),
        axis=1,
    )
    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All Matches": df_all}
full_data.update(dfs_by_match)

# ==========================
# Stats
# ==========================
def compute_stats(df: pd.DataFrame) -> dict:
    total_passes = len(df)
    successful = int(df["type"].str.contains("WON", case=False).sum())
    unsuccessful = int(df["type"].str.contains("LOST", case=False).sum())
    accuracy = (successful / total_passes * 100.0) if total_passes else 0.0

    progressive_total = int(df["progressive"].sum())
    progressive_successful = int(
        (df["progressive"] & df["type"].str.contains("WON", case=False)).sum()
    )
    progressive_accuracy = (
        progressive_successful / progressive_total * 100.0
        if progressive_total
        else 0.0
    )

    key_passes = int(df["video"].apply(has_video_value).sum())

    in_final_third = df["x_end"] >= FINAL_THIRD_LINE_X
    final_third_total = int(in_final_third.sum())
    final_third_success = int(
        (in_final_third & df["type"].str.contains("WON", case=False)).sum()
    )
    final_third_unsuccess = int(
        (in_final_third & df["type"].str.contains("LOST", case=False)).sum()
    )
    final_third_accuracy = (
        (final_third_success / final_third_total * 100.0) if final_third_total else 0.0
    )

    to_box = (
        (df["x_end"] >= BOX_X_MIN)
        & (df["y_end"] >= BOX_Y_MIN)
        & (df["y_end"] <= BOX_Y_MAX)
    )
    box_total = int(to_box.sum())
    box_success = int(
        (to_box & df["type"].str.contains("WON", case=False)).sum()
    )
    box_unsuccess = int(
        (to_box & df["type"].str.contains("LOST", case=False)).sum()
    )
    box_accuracy = (box_success / box_total * 100.0) if box_total else 0.0

    return {
        "total_passes": total_passes,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "key_passes": key_passes,
        "progressive_passes": progressive_total,
        "progressive_successful_passes": progressive_successful,
        "progressive_accuracy_pct": round(progressive_accuracy, 2),
        "final_third_total": final_third_total,
        "final_third_success": final_third_success,
        "final_third_unsuccess": final_third_unsuccess,
        "final_third_accuracy_pct": round(final_third_accuracy, 2),
        "box_total": box_total,
        "box_success": box_success,
        "box_unsuccess": box_unsuccess,
        "box_accuracy_pct": round(box_accuracy, 2),
    }


# ==========================
# Metric helper — mostra valor + p90 embaixo
# ==========================
def metric_with_p90(container, label: str, value, minutes: int, show_p90: bool = True):
    """
    Exibe um st.metric e logo abaixo o valor per 90 em texto pequeno.
    Para percentuais (accuracy) não faz sentido mostrar p90.
    """
    container.metric(label, value)
    if show_p90 and isinstance(value, (int, np.integer)):
        container.caption(f"p90: {per90(value, minutes)}")


# ==========================
# Draw pass map
# ==========================
FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110


def draw_pass_map(df: pd.DataFrame, title: str):
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#f5f5f5",
        line_color="#4a4a4a",
    )
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_dpi(FIG_DPI)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.2, alpha=0.25)

    START_DOT_SIZE = 45

    for _, row in df.iterrows():
        is_lost = "LOST" in row["type"].upper()
        is_progressive_success = bool(row["progressive"]) and not is_lost
        has_vid = has_video_value(row["video"])

        if is_lost:
            color = COLOR_FAIL
            alpha = 0.55
        elif is_progressive_success:
            color = COLOR_PROGRESSIVE
            alpha = 0.82
        else:
            color = COLOR_SUCCESS
            alpha = 0.80

        pitch.arrows(
            row["x_start"], row["y_start"],
            row["x_end"], row["y_end"],
            color=color, width=1.55, headwidth=2.25, headlength=2.25,
            ax=ax, zorder=3, alpha=alpha,
        )

        if has_vid:
            pitch.scatter(
                row["x_start"], row["y_start"],
                s=95, marker="o", facecolors="none",
                edgecolors="#FFD54F", linewidths=2.0, ax=ax, zorder=4,
            )

        pitch.scatter(
            row["x_start"], row["y_start"],
            s=START_DOT_SIZE, marker="o", color=color,
            edgecolors="white", linewidths=0.8, ax=ax, zorder=5, alpha=alpha,
        )

    ax.set_title(title, fontsize=12)

    legend_elements = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.5, label="Successful Pass"),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.5, label="Unsuccessful Pass"),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.5,
               label="Successful Progressive Pass (Opta)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=6, label="Start point (click)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="#FFD54F", markeredgewidth=2, markersize=7,
               label="Has video"),
    ]

    legend = ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="white", edgecolor="#cccccc", shadow=False,
        fontsize="x-small", labelspacing=0.5, borderpad=0.5,
    )
    legend.get_frame().set_alpha(1.0)

    arrow = FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#333333",
    )
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction",
             ha="center", va="center", fontsize=9, color="#333333")

    fig.tight_layout()
    fig.canvas.draw()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI)
    buf.seek(0)
    img_obj = Image.open(buf)
    return img_obj, ax, fig


# ==========================
# Sidebar
# ==========================
st.sidebar.header("Match Selection")
selected_match = st.sidebar.radio(
    "Choose the match", list(full_data.keys()), index=0
)

st.sidebar.header("Pass Filter")
pass_filter = st.sidebar.radio(
    "Filter passes",
    ["All Passes", "Successful Only", "Unsuccessful Only", "Progressive Only"],
    index=0,
)

df = full_data[selected_match].copy()

if pass_filter == "Successful Only":
    df = df[df["type"].str.contains("WON", case=False)].reset_index(drop=True)
elif pass_filter == "Unsuccessful Only":
    df = df[df["type"].str.contains("LOST", case=False)].reset_index(drop=True)
elif pass_filter == "Progressive Only":
    df = df[
        df["progressive"] & df["type"].str.contains("WON", case=False)
    ].reset_index(drop=True)

stats = compute_stats(df)
mins = MINUTES_PLAYED

# ==========================
# Layout
# ==========================
col_stats, col_right = st.columns([1, 2], gap="large")

with col_stats:
    st.subheader("Statistics")

    c1, c2, c3 = st.columns(3)
    metric_with_p90(c1, "Total Passes", stats["total_passes"], mins)
    metric_with_p90(c2, "Successful", stats["successful_passes"], mins)
    c3.metric("Accuracy", f'{stats["accuracy_pct"]:.1f}%')

    st.divider()

    st.subheader("Progressive Passes")
    p1, p2, p3 = st.columns(3)
    metric_with_p90(p1, "Total", stats["progressive_passes"], mins)
    metric_with_p90(p2, "Successful", stats["progressive_successful_passes"], mins)
    p3.metric("Accuracy", f'{stats["progressive_accuracy_pct"]:.1f}%')

    st.divider()

    st.subheader("Final Third")
    c7, c8, c9 = st.columns(3)
    metric_with_p90(c7, "Total", stats["final_third_total"], mins)
    metric_with_p90(c8, "Successful", stats["final_third_success"], mins)
    metric_with_p90(c9, "Unsuccessful", stats["final_third_unsuccess"], mins)
    st.metric("Accuracy", f'{stats["final_third_accuracy_pct"]:.1f}%')

    st.divider()

    st.subheader("Passes Into the Box")
    d1, d2, d3 = st.columns(3)
    metric_with_p90(d1, "Total", stats["box_total"], mins)
    metric_with_p90(d2, "Successful", stats["box_success"], mins)
    metric_with_p90(d3, "Unsuccessful", stats["box_unsuccess"], mins)
    st.metric("Accuracy", f'{stats["box_accuracy_pct"]:.1f}%')

with col_right:
    st.subheader("Pass Map (click the start dot)")

    img_obj, ax, fig = draw_pass_map(df, title=f"Pass Map — {selected_match}")

    DISPLAY_WIDTH = 780
    click = streamlit_image_coordinates(img_obj, width=DISPLAY_WIDTH)

    selected_pass = None

    if click is not None:
        real_w, real_h = img_obj.size
        disp_w = click["width"]
        disp_h = click["height"]

        pixel_x = click["x"] * (real_w / disp_w)
        pixel_y = click["y"] * (real_h / disp_h)
        mpl_pixel_y = real_h - pixel_y

        field_x, field_y = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))

        df_sel = df.copy()
        df_sel["dist"] = np.sqrt(
            (df_sel["x_start"] - field_x) ** 2
            + (df_sel["y_start"] - field_y) ** 2
        )

        RADIUS = 5.0
        candidates = df_sel[df_sel["dist"] < RADIUS].copy()

        if not candidates.empty:
            candidates = candidates.sort_values(by="dist", ascending=True)
            selected_pass = candidates.iloc[0]

    plt.close(fig)

    st.divider()
    st.subheader("Selected Event")

    if selected_pass is None:
        st.info("Click the start dot to inspect the pass details.")
    else:
        st.success(
            f"Selected pass: #{int(selected_pass['number'])} ({selected_pass['type']})"
        )
        st.write(
            f"Start: ({selected_pass['x_start']:.2f}, {selected_pass['y_start']:.2f})  \n"
            f"End: ({selected_pass['x_end']:.2f}, {selected_pass['y_end']:.2f})"
        )
        st.write(f"Progressive: {'Yes' if selected_pass['progressive'] else 'No'}")

        if has_video_value(selected_pass["video"]):
            try:
                st.video(selected_pass["video"])
            except Exception:
                st.error(f"Video file not found: {selected_pass['video']}")
        else:
            st.warning("No video is attached to this event.")
