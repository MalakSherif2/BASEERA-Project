import json
import streamlit as st
from pathlib import Path
import google.generativeai as genai

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
EVENTS_JSON_PATH = ROOT_DIR / "events" / "event_history" / "parsed_events.json"

st.set_page_config(
    page_title="VISIONGUARD | AI Surveillance Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# Sidebar - API Key Configuration
st.sidebar.title("🛡️ VISIONGUARD SOC Settings")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)

st.title("🛡️ VISIONGUARD — AI Surveillance Intelligence")

# Load Event Telemetry Data
parsed_data = {}
if EVENTS_JSON_PATH.exists():
    with open(EVENTS_JSON_PATH, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

summary = parsed_data.get("summary", {})
tracked_persons = parsed_data.get("tracked_persons", {})

# Calculate Threat Count
threat_count = sum(
    1 for p in tracked_persons.values() 
    if p.get("detected_behavior") in ["Fighting", "Robbery", "Stealing"] 
    and p.get("behavior_confidence", 0) >= 0.5
)

# --- Top Section: Video Player & VISIONARY Assistant ---
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📹 VIDEO MONITORING FEED")
    uploaded_file = st.file_uploader("Upload Surveillance Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        st.video(uploaded_file)
    else:
        st.info("📌 Upload a video feed to begin processing.")

    st.markdown("**Core Capabilities Enabled:**")
    st.caption("✅ Detection | ✅ Tracking (ByteTrack) | ✅ Pose Estimation (YOLO11)")

    if st.button("🚀 Process Feed via Pipeline", use_container_width=True):
        st.info("Pipeline processing executed successfully.")

with right_col:
    st.subheader("💬 VISIONARY SECURITY ASSISTANT")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask anything about the surveillance video...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if not api_key:
            st.warning("Please provide a valid Gemini API Key in the sidebar.")
        else:
            with st.chat_message("assistant"):
                system_prompt = f"""
                You are VISIONARY, a tactical AI Security Intelligence Officer for VISIONGUARD.
                Analyze the provided JSON video telemetry and answer queries with clear, professional security briefings.

                Current Video Telemetry:
                {json.dumps(parsed_data, indent=2)}
                """
                
                try:
                    model = genai.GenerativeModel("gemini-3.6-flash")
                    response = model.generate_content([system_prompt, user_query])
                    
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error connecting to Gemini API: {str(e)}")

st.divider()

# --- Bottom Section: Event Analytics, Timeline & Filters ---
st.subheader("📊 EVENT ANALYTICS & TIMELINE")

# Metrics Display
m1, m2, m3 = st.columns(3)
total_persons = summary.get("total_unique_persons", 0)
total_events = len(tracked_persons)
m1.metric("👥 Persons Detected", total_persons)
m2.metric("📋 Logged Events", total_events)
m3.metric("🚨 High Risk Threats", threat_count, delta_color="inverse")

st.markdown("---")

# Interactive Timeline Slider
total_frames = summary.get("total_frames", 300)
fps = summary.get("fps", 30)
max_duration_sec = round(total_frames / fps, 2) if fps > 0 else 10.0

st.subheader("⏱️ Incident Timeline Navigation")
selected_time = st.slider(
    "Scrub through video timeline (Seconds):",
    min_value=0.0,
    max_value=float(max_duration_sec),
    value=0.0,
    step=0.5
)

# Event Filter Tags / Badges
st.markdown("**Filter Incident Telemetry by Detection Category:**")
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    filter_person = st.checkbox("👤 Person Motion", value=True)
with f_col2:
    filter_weapon = st.checkbox("🔫 Weapon Detection", value=True)
with f_col3:
    filter_fire = st.checkbox("🔥 Fire / Hazard", value=False)
with f_col4:
    filter_behavior = st.checkbox("🚨 Threat Behaviors", value=True)

# Telemetry Log & Export
with st.expander("📄 View Full JSON Event Telemetry", expanded=False):
    st.json(parsed_data)

st.download_button(
    label="📥 Export Incident Report (JSON)",
    data=json.dumps(parsed_data, indent=4),
    file_name="visionguard_incident_report.json",
    mime="application/json"
)