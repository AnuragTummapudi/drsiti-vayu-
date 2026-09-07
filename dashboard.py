import streamlit as st
import json
from pathlib import Path

# ============================================================
# MINEGUARD CONTROL ROOM
# Prototype Dashboard
# ============================================================

EVENT_FILE = Path("mineguard_events/events.jsonl")

st.set_page_config(
    page_title="MineGuard Control Room",
    page_icon="⛏️",
    layout="wide",
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #111417;
    color: #E6E9EC;
}

.block-container {
    padding-top: 1rem;
}

h1, h2, h3 {
    color: #F1F3F5;
}

.vehicle {
    background-color: #191D21;
    border: 1px solid #30363B;
    padding: 12px;
    margin-bottom: 8px;
    border-radius: 6px;
}

.normal {
    border-left: 4px solid #43A047;
}

.warning {
    border-left: 4px solid #FBC02D;
}

.elevated {
    border-left: 4px solid #F57C00;
}

.critical {
    border-left: 4px solid #E53935;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD EVENTS
# ============================================================

def load_events():

    if not EVENT_FILE.exists():
        return []

    events = []

    with open(EVENT_FILE, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return events


events = load_events()

latest = events[-1] if events else None


# ============================================================
# SIMULATED FLEET
# ============================================================

fleet = [
    ["TRK-07", "Heavy Dump Truck", 18, "North Haul Road", "NORMAL"],
    ["TRK-08", "Heavy Dump Truck", 14, "North Haul Road", "WARNING"],
    ["TRK-12", "Mining Truck", 9, "Fog Zone A", "ELEVATED"],
    ["TRK-15", "Heavy Dump Truck", 21, "South Haul Road", "NORMAL"],
    ["EXC-03", "Excavator", 3, "Loading Area", "NORMAL"],
]


# ============================================================
# APPLY REAL EVENT TO TRK-07
# ============================================================

if latest:

    fleet[0][4] = latest.get(
        "risk_level",
        "NORMAL"
    )


# ============================================================
# HEADER
# ============================================================

title, status = st.columns([5, 1])

with title:

    st.title("MINEGUARD")

    st.write(
        "Edge-AI Mine Safety & Situational Awareness Platform"
    )

with status:

    st.success("SYSTEM ONLINE")


st.divider()


# ============================================================
# GLOBAL STATUS
# ============================================================

active = len(fleet)

warnings = sum(
    1 for v in fleet
    if v[4] == "WARNING"
)

elevated = sum(
    1 for v in fleet
    if v[4] == "ELEVATED"
)

critical = sum(
    1 for v in fleet
    if v[4] == "CRITICAL"
)

visibility = (
    latest.get("visibility", "NORMAL")
    if latest else "NORMAL"
)

risk = (
    latest.get("risk_level", "NORMAL")
    if latest else "NORMAL"
)

score = (
    latest.get("risk_score", 0)
    if latest else 0
)


a, b, c, d, e = st.columns(5)

a.metric("ACTIVE VEHICLES", active)
b.metric("WARNING", warnings)
c.metric("ELEVATED", elevated)
d.metric("CRITICAL", critical)
e.metric("VISIBILITY", visibility)


st.divider()


# ============================================================
# THREE COLUMNS
# ============================================================

left, center, right = st.columns([1.2, 2.5, 1.3])


# ============================================================
# FLEET
# ============================================================

with left:

    st.subheader("FLEET")

    for vehicle in fleet:

        vehicle_id = vehicle[0]
        vehicle_type = vehicle[1]
        speed = vehicle[2]
        zone = vehicle[3]
        vehicle_risk = vehicle[4]

        if vehicle_risk == "CRITICAL":
            icon = "🔴"

        elif vehicle_risk == "ELEVATED":
            icon = "🟠"

        elif vehicle_risk == "WARNING":
            icon = "🟡"

        else:
            icon = "🟢"

        st.markdown(
            f"""
            <div class="vehicle {vehicle_risk.lower()}">
                <b>{icon} {vehicle_id}</b><br>
                {vehicle_type}<br>
                {speed} km/h<br>
                {zone}<br>
                <b>Risk: {vehicle_risk}</b>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MAP
# ============================================================

with center:

    st.subheader("MINE OPERATIONS MAP")

    st.markdown(
        """
        ### ⛰️ OPEN-CAST MINE

        **LOADING AREA**

        🟢 TRK-07 ───────────── 🟡 TRK-08

        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        **NORTH HAUL ROAD**

        ──────────────────────────────

        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        🟠 TRK-12

        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        **⚠ LOW VISIBILITY ZONE**

        ──────────────────────────────

        🟢 TRK-15 ────────────────

        **CRUSHER**

        **MAINTENANCE AREA**

        ---

        **Legend**

        🟢 Normal &nbsp;&nbsp;
        🟡 Warning &nbsp;&nbsp;
        🟠 Elevated &nbsp;&nbsp;
        🔴 Critical
        """
    )

    st.caption(
        "Prototype operational map — production positions "
        "would be supplied by GNSS/DGPS."
    )


# ============================================================
# SELECTED ASSET
# ============================================================

with right:

    st.subheader("SELECTED ASSET")

    selected = fleet[0]

    st.markdown(
        f"""
        ## {selected[0]}

        **Type:**  
        {selected[1]}

        **Speed:**  
        {selected[2]} km/h

        **Zone:**  
        {selected[3]}

        **Risk:**  
        {selected[4]}
        """
    )

    st.divider()

    st.subheader("EDGE SYSTEM")

    st.success("Camera ONLINE")
    st.success("Edge AI ONLINE")
    st.success("Network ONLINE")

    if latest:

        st.divider()

        st.subheader("LATEST EVENT")

        st.write(
            f"Risk: {score}/100"
        )

        st.write(
            latest.get(
                "risk_reason",
                "No active event."
            )
        )


# ============================================================
# EVENTS
# ============================================================

st.divider()

st.subheader("SAFETY EVENTS")

if not events:

    st.info("No safety events recorded.")

else:

    for event in reversed(events[-10:]):

        level = event.get(
            "risk_level",
            "NORMAL"
        )

        score = event.get(
            "risk_score",
            0
        )

        timestamp = event.get(
            "timestamp",
            ""
        )

        visibility = event.get(
            "visibility",
            "NORMAL"
        )

        reason = event.get(
            "risk_reason",
            ""
        )

        if level == "CRITICAL":
            icon = "🔴"

        elif level == "ELEVATED":
            icon = "🟠"

        elif level == "WARNING":
            icon = "🟡"

        else:
            icon = "🟢"

        st.write(
            f"{icon} **{level}** | "
            f"Risk {score}/100 | "
            f"Visibility: {visibility} | "
            f"{timestamp}"
        )

        st.write(reason)

        st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "MINEGUARD • Edge Perception • Risk Intelligence "
    "• Fleet Situational Awareness"
)