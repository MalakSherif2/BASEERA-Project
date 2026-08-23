import streamlit as st
import json
import subprocess
import sys
import time
from pathlib import Path
import plotly.graph_objects as go
import os


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(html_string: str):
    lines = html_string.split("\n")
    cleaned = "\n".join(line.strip() for line in lines)
    st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# OPTIONAL GEMINI
# ============================================================

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BASEERA | Intelligent Video Understanding",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

PIPELINE_FILE = ROOT_DIR / "src" / "pipeline.py"

INPUT_DIR = ROOT_DIR / "data" / "inputs"
OUTPUT_DIR = ROOT_DIR / "data" / "outputs"

EVENT_DIR = ROOT_DIR / "events" / "event_history"

PARSED_EVENTS = EVENT_DIR / "parsed_events.json"
ADVANCED_EVENTS = EVENT_DIR / "advanced_event_history.json"

OUTPUT_VIDEO = OUTPUT_DIR / "annotated_pipeline_out.mp4"
WEB_VIDEO = OUTPUT_DIR / "annotated_pipeline_web.mp4"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INITIALIZE SESSION STATE & CLEAR OLD CACHE ON FIRST LOAD
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
    if PARSED_EVENTS.exists():
        try:
            PARSED_EVENTS.unlink()
        except Exception:
            pass
    if ADVANCED_EVENTS.exists():
        try:
            ADVANCED_EVENTS.unlink()
        except Exception:
            pass


# ============================================================
# FFMPEG
# ============================================================

FFMPEG_PATH = (
    r"C:\Users\dell\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0-full_build\bin\ffmpeg.exe"
)


def get_ffmpeg():
    if Path(FFMPEG_PATH).exists():
        return FFMPEG_PATH
    return "ffmpeg"


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None

if GEMINI_API_KEY and GEMINI_AVAILABLE:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )
    except Exception:
        gemini_client = None


# ============================================================
# CSS (Baseera Purple Theme)
# ============================================================

render_html(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 5% 0%,
                rgba(140,80,255,0.08),
                transparent 25%
            ),
            radial-gradient(
                circle at 95% 5%,
                rgba(100,50,220,0.10),
                transparent 30%
            ),
            #0b0813;

        color: #f5f7fa;
    }


    /* SIDEBAR */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0d0a17 0%,
                #07050d 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] * {
        color: #dce3ec;
    }


    /* HEADER */

    .vg-header {
        padding: 8px 0 26px 0;
    }

    .vg-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.05;
        margin: 0;
        color: #ffffff;
    }

    .vg-subtitle {
        color: #9d98b8;
        font-size: 14px;
        margin-top: 8px;
    }

    .online {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 15px;
        padding: 7px 13px;
        border-radius: 999px;
        background:
            rgba(160,100,255,0.08);
        border:
            1px solid rgba(160,100,255,0.22);
        color: #c49eff;
        font-size: 12px;
        font-weight: 700;
    }

    .dot {
        width: 8px;
        height: 8px;
        background: #a855f7;
        border-radius: 50%;
        box-shadow:
            0 0 12px #a855f7;
    }


    /* METRICS */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.04),
                rgba(255,255,255,0.01)
            );

        border:
            1px solid rgba(168,85,247,0.15);

        border-radius: 18px;
        padding: 18px;
        min-height: 125px;

        box-shadow:
            0 12px 35px rgba(0,0,0,0.20);
    }

    .metric-label {
        color: #a399c2;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        margin-top: 9px;
        color: #ffffff;
    }

    .metric-small {
        color: #7b719b;
        font-size: 11px;
        margin-top: 5px;
    }


    /* SECTIONS */

    .section-title {
        font-size: 21px;
        font-weight: 800;
        margin: 28px 0 14px 0;
        color: #e2d9f3;
    }


    /* PANELS */

    .panel {
        background:
            rgba(255,255,255,0.02);

        border:
            1px solid rgba(168,85,247,0.15);

        border-radius: 18px;
        padding: 18px;
    }


    /* PERSON */

    .person-card {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.03),
                rgba(255,255,255,0.01)
            );

        border:
            1px solid rgba(168,85,247,0.15);

        border-radius: 18px;
        padding: 18px;
        margin-bottom: 12px;
    }


    /* EVENTS */

    .event-critical {
        border-left:
            4px solid #ff3b55;

        background:
            rgba(255,59,85,0.055);

        padding: 15px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .event-low {
        border-left:
            4px solid #a855f7;

        background:
            rgba(168,85,247,0.05);

        padding: 15px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
    }


    /* BADGES */

    .badge-critical {
        color: #ff6476;
        font-weight: 800;
    }

    .badge-low {
        color: #c49eff;
        font-weight: 800;
    }


    /* ANALYST */

    .analyst-box {
        background:
            linear-gradient(
                145deg,
                rgba(140,80,255,0.08),
                rgba(80,40,180,0.04)
            );

        border:
            1px solid rgba(168,85,247,0.25);

        border-radius: 22px;
        padding: 22px;
        margin-top: 10px;
    }

    .analyst-title {
        font-size: 25px;
        font-weight: 800;
        color: #ffffff;
    }

    .analyst-subtitle {
        color: #9d98b8;
        margin-top: 6px;
    }

    .ai-online {
        display: inline-block;
        margin-top: 14px;
        padding: 7px 12px;
        border-radius: 999px;
        background:
            rgba(168,85,247,0.1);
        color: #c49eff;
        font-size: 12px;
        font-weight: 700;
    }


    /* REPORT */

    .report-box {
        background:
            rgba(255,255,255,0.02);

        border:
            1px solid rgba(168,85,247,0.15);

        border-radius: 18px;
        padding: 24px;
        line-height: 1.7;
    }


    /* CHAT */

    .chat-box {
        background:
            linear-gradient(
                145deg,
                rgba(168,85,247,0.04),
                rgba(90,80,255,0.06)
            );

        border:
            1px solid rgba(168,85,247,0.15);

        border-radius: 20px;
        padding: 22px;
        margin-top: 10px;
    }


    /* FOOTER */

    .footer {
        text-align: center;
        color: #6b5f8a;
        font-size: 11px;
        padding: 40px 0 15px 0;
    }

    </style>
    """
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_uploaded_video(uploaded_file):
    destination = INPUT_DIR / uploaded_file.name
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return destination


def run_pipeline(video_path):
    process = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_FILE),
            str(video_path),
        ],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",      
        errors="ignore" 
    )
    return process


def convert_video_for_web():
    if not OUTPUT_VIDEO.exists():
        return False

    ffmpeg = get_ffmpeg()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(OUTPUT_VIDEO),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(WEB_VIDEO),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr)
        return (
            result.returncode == 0
            and WEB_VIDEO.exists()
        )
    except Exception as e:
        print(f"FFmpeg Error: {e}")
        return False


def metric_card(label, value, description=""):
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>
            <div class="metric-value">
                {value}
            </div>
            <div class="metric-small">
                {description}
            </div>
        </div>
        """
    )


def ask_gemini(prompt):
    if gemini_client is None:
        return (
            "Gemini is not connected. "
            "Please verify GEMINI_API_KEY."
        )
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Gemini Analyst Error: {e}"


def event_to_text(event):
    return json.dumps(
        event,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# HEADER
# ============================================================

render_html(
    """
    <div class="vg-header">
        <div class="vg-title">
            👁️ BASEERA
        </div>
        <div class="vg-subtitle">
            Intelligent Video Understanding & Campus Security System
            &nbsp;•&nbsp;
            Behavioral Intelligence
            &nbsp;•&nbsp;
            Threat Detection
        </div>
        <div class="online">
            <span class="dot"></span>
            SYSTEM ONLINE
        </div>
    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 👁️ BASEERA COMMAND")

    st.caption(
        "Intelligent video monitoring & analysis"
    )

    st.divider()

    st.markdown(
        "### 🎥 Video Analysis"
    )

    uploaded_file = st.file_uploader(
        "Upload surveillance video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm",
            "mpeg4",
        ],
    )

    analyze_button = st.button(
        "🚀 RUN FULL AI ANALYSIS",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    st.markdown(
        "### AI Stack"
    )

    st.success("YOLO11 Pose")
    st.success("ByteTrack")
    st.success("LSTM")
    st.success("Temporal Transformer")
    st.success("Spatio-Temporal Fusion")
    st.success("Weapon Detection")
    st.success("SAM ViT-B")
    st.success("Event Intelligence")

    st.divider()

    if gemini_client:
        st.success(
            " BASEERA AI Assistant ONLINE"
        )
    else:
        st.warning(
            "AI Assistant OFFLINE"
        )

    st.divider()

    st.caption(
        "BASEERA v1.0"
    )

    st.caption(
        "Intelligent Video Understanding"
    )


# ============================================================
# UPLOAD + PIPELINE
# ============================================================

if analyze_button:

    if uploaded_file is None:
        st.warning(
            "⚠️ Please upload a surveillance video first."
        )
    else:
        st.session_state["analysis_done"] = False
        if PARSED_EVENTS.exists():
            PARSED_EVENTS.unlink()
        if ADVANCED_EVENTS.exists():
            ADVANCED_EVENTS.unlink()

        with st.status(
            " Initializing BASEERA AI Pipeline...",
            expanded=True,
        ) as status:

            st.write(
                " Saving uploaded video..."
            )

            video_path = save_uploaded_video(
                uploaded_file
            )

            st.write(
                "👤 YOLO11 Pose Detection..."
            )
            time.sleep(0.2)

            st.write(
                " ByteTrack Identity Tracking..."
            )
            time.sleep(0.2)

            st.write(
                " LSTM Behavioral Analysis..."
            )
            time.sleep(0.2)

            st.write(
                " Temporal Transformer..."
            )
            time.sleep(0.2)

            st.write(
                "🔗 Spatio-Temporal Fusion..."
            )
            time.sleep(0.2)

            st.write(
                " Weapon Detection..."
            )
            time.sleep(0.2)

            st.write(
                " SAM Segmentation..."
            )
            time.sleep(0.2)

            st.write(
                " Advanced Event Intelligence..."
            )

            result = run_pipeline(
                video_path
            )

            if result.returncode == 0:

                st.write(
                    " Optimizing annotated video for web playback..."
                )

                video_ready = convert_video_for_web()

                if video_ready:
                    st.write(
                        " H.264 web-compatible video generated."
                    )
                else:
                    st.warning(
                        "⚠️ Web conversion failed. "
                        "Original video will be used."
                    )

                status.update(
                    label=" BASEERA Analysis Complete",
                    state="complete",
                )

                st.session_state["analysis_done"] = True
                st.session_state["uploaded_video_name"] = uploaded_file.name

            else:

                status.update(
                    label="❌ Pipeline Failed",
                    state="error",
                )

                st.error(
                    result.stderr
                )

                if result.stdout:
                    with st.expander(
                        "Pipeline Output"
                    ):
                        st.code(
                            result.stdout
                        )


# ============================================================
# LOAD RESULTS
# ============================================================

data = None
advanced_events = None

if st.session_state.get("analysis_done", False):
    data = load_json(PARSED_EVENTS)
    advanced_events = load_json(ADVANCED_EVENTS)


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.get("analysis_done", False) or data is None:

    render_html(
        """
        <div class="panel">
            <h2>
                BASEERA Command Center
            </h2>
            <p style="color:#9d98b8;">
                Upload a surveillance video and click <b>RUN FULL AI ANALYSIS</b> to activate
                the complete AI security pipeline.
            </p>
            <p style="color:#c49eff;">
                YOLO11 Pose →
                ByteTrack →
                LSTM →
                Temporal Transformer →
                Spatio-Temporal Fusion →
                Weapon Detection →
                SAM →
                Event Intelligence →
                BASEERA Assistant
            </p>
        </div>
        """
    )

    st.stop()


# ============================================================
# DATA EXTRACTION
# ============================================================

summary = data.get("summary", {})
components = data.get("pipeline_components", {})
persons = data.get("tracked_persons", {})

total_frames = summary.get("total_frames", 0)
persons_count = summary.get("total_unique_persons", 0)
weapon_frames = summary.get("weapon_detection_frames", 0)
critical = summary.get("critical_events", 0)
high = summary.get("high_events", 0)
medium = summary.get("medium_events", 0)
transformer_predictions = summary.get("transformer_predictions", 0)
fusion_predictions = summary.get("fusion_predictions", 0)


# ============================================================
# KPI
# ============================================================

render_html(
    """
    <div class="section-title">
        📡 Threat Intelligence Overview
    </div>
    """
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    metric_card("Frames", total_frames, "Frames analyzed")

with c2:
    metric_card("Persons", persons_count, "Tracked identities")

with c3:
    metric_card("Weapon Frames", weapon_frames, "Weapon detections")

with c4:
    metric_card("Critical", critical, "Critical incidents")

with c5:
    metric_card("Transformer", transformer_predictions, "Predictions")

with c6:
    metric_card("Fusion", fusion_predictions, "Predictions")


# ============================================================
# THREAT LEVEL
# ============================================================

if critical > 0:
    threat = "CRITICAL"
elif high > 0:
    threat = "HIGH"
elif medium > 0:
    threat = "MEDIUM"
else:
    threat = "LOW"

render_html(
    """
    <div class="section-title">
        🚨 Current Threat Level
    </div>
    """
)

if threat == "CRITICAL":
    st.error("🔴 CRITICAL — Immediate security attention recommended.")
elif threat == "HIGH":
    st.warning("🟠 HIGH — Suspicious activity detected.")
elif threat == "MEDIUM":
    st.warning("🟡 MEDIUM — Behavioral anomaly detected.")
else:
    st.success("🟢 LOW — No critical threat detected.")


# ============================================================
# VIDEO
# ============================================================

render_html(
    """
    <div class="section-title">
        🎬 Annotated Surveillance Video
    </div>
    """
)

video_to_show = None

if WEB_VIDEO.exists():
    video_to_show = WEB_VIDEO
elif OUTPUT_VIDEO.exists():
    video_to_show = OUTPUT_VIDEO

if video_to_show:
    st.video(str(video_to_show))
else:
    st.warning("Annotated video is not available yet.")


# ============================================================
# PIPELINE ARCHITECTURE
# ============================================================

render_html(
    """
    <div class="section-title">
        🧠 AI Pipeline Architecture
    </div>
    """
)

pipeline_items = [
    (" Pose Detection", components.get("pose_detection", "YOLO11n-Pose")),
    (" Fusion", components.get("spatiotemporal_fusion", "LOADED")),
    (" Tracking", components.get("tracking", "ByteTrack")),
    (" Weapon", components.get("weapon_detection", "Custom Weapon YOLO")),
    (" Behavior", components.get("temporal_behavior", "LSTM Sliding Window")),
    (" Segmentation", components.get("segmentation", "SAM ViT-B")),
    (" Transformer", components.get("temporal_transformer", "LOADED")),
    (" Event Engine", components.get("event_engine", "Advanced Event Intelligence")),
]

pipeline_cols = st.columns(4)

for i, (title, value) in enumerate(pipeline_items):
    with pipeline_cols[i % 4]:
        render_html(
            f"""
            <div class="panel">
                <b>{title}</b>
                <br><br>
                <span style="color:#c49eff;">
                    ● {value}
                </span>
            </div>
            """
        )


# ============================================================
# PERSON INTELLIGENCE
# ============================================================

render_html(
    """
    <div class="section-title">
        👥 Tracked Person Intelligence
    </div>
    """
)

for key, person in persons.items():
    person_id = person.get("person_id", key)
    behavior = person.get("detected_behavior", "Unknown")
    confidence = person.get("behavior_confidence", 0)
    weapon = person.get("weapon_detected", False)
    weapon_frames_person = person.get("weapon_frames", 0)
    severity = person.get("severity", "LOW")
    start = person.get("start_time_sec", 0)
    end = person.get("end_time_sec", 0)
    transformer_behavior = person.get("transformer_behavior", "Not Available")
    transformer_confidence = person.get("transformer_confidence", 0)
    fusion_behavior = person.get("fusion_behavior", "Not Available")
    fusion_confidence = person.get("fusion_confidence", 0)

    if severity == "CRITICAL":
        icon = "🔴"
        badge = "badge-critical"
    else:
        icon = "🟢"
        badge = "badge-low"

    render_html(
        f"""
        <div class="person-card">
            <h3>
                {icon} Person {person_id} — {behavior} — <span class="{badge}">{severity}</span>
            </h3>
            <hr style="border-color: rgba(255,255,255,0.06);">
            <b>Behavior:</b> {behavior} &nbsp;&nbsp; | &nbsp;&nbsp;
            <b>Confidence:</b> {confidence:.2f} &nbsp;&nbsp; | &nbsp;&nbsp;
            <b>Weapon:</b> {"YES " if weapon else "NO"}
            <br><br>
            <b>Weapon Frames:</b> {weapon_frames_person} &nbsp;&nbsp; | &nbsp;&nbsp;
            <b>Timeline:</b> {start:.2f}s → {end:.2f}s
            <br><br>
            <b>🤖 Transformer:</b> {transformer_behavior} ({transformer_confidence:.2f}) &nbsp;&nbsp; | &nbsp;&nbsp;
            <b>🔗 Fusion:</b> {fusion_behavior} ({fusion_confidence:.2f})
        </div>
        """
    )


# ============================================================
# ADVANCED EVENTS
# ============================================================

render_html(
    """
    <div class="section-title">
        🚨 Advanced Event Intelligence
    </div>
    """
)

if advanced_events:
    for event in advanced_events:
        severity = event.get("severity", "LOW")
        event_id = event.get("event_id", "UNKNOWN")
        person_id = event.get("person_id", "—")
        behavior = event.get("behavior", "Unknown")
        weapon = event.get("weapon_detected", False)
        confidence = event.get("confidence", 0)

        if severity == "CRITICAL":
            render_html(
                f"""
                <div class="event-critical">
                    🔴 <b>{event_id}</b> &nbsp; | &nbsp;
                    Person <b>{person_id}</b> &nbsp; | &nbsp;
                    Behavior: <b>{behavior}</b> &nbsp; | &nbsp;
                    Weapon: <b>{"YES " if weapon else "NO"}</b> &nbsp; | &nbsp;
                    Confidence: <b>{confidence:.2f}</b> &nbsp; | &nbsp;
                    Severity: <b>CRITICAL</b>
                </div>
                """
            )
        else:
            render_html(
                f"""
                <div class="event-low">
                    🟢 <b>{event_id}</b> &nbsp; | &nbsp;
                    Person <b>{person_id}</b> &nbsp; | &nbsp;
                    Behavior: <b>{behavior}</b> &nbsp; | &nbsp;
                    Weapon: <b>{"YES " if weapon else "NO"}</b> &nbsp; | &nbsp;
                    Severity: <b>{severity}</b>
                </div>
                """
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

render_html(
    """
    <div class="section-title">
        Multi-Model Intelligence
    </div>
    """
)

model_data = {
    "LSTM": summary.get("sliding_window_predictions", 0),
    "Temporal Transformer": transformer_predictions,
    "Spatio-Temporal Fusion": fusion_predictions,
}

fig = go.Figure(
    data=[
        go.Bar(
            x=list(model_data.keys()),
            y=list(model_data.values()),
            text=list(model_data.values()),
            textposition="auto",
            marker_color="#a855f7",
        )
    ]
)

fig.update_layout(
    height=350,
    margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# BASEERA AI ASSISTANT
# ============================================================

render_html(
    """
    <div class="analyst-box">
        <div class="analyst-title">
            BASEERA AI ASSISTANT
        </div>
        <div class="analyst-subtitle">
            Ask the intelligence layer about detected
            events, behaviors, weapons, persons and
            model decisions.
        </div>
        <div class="ai-online">
            ● BASEERA AI Assistant ONLINE
            — Connected to Event Intelligence
        </div>
    </div>
    """
)


# ============================================================
# EVENT DEEP ANALYSIS
# ============================================================

if advanced_events:
    event_labels = []
    event_map = {}

    for event in advanced_events:
        event_id = event.get("event_id", "UNKNOWN")
        person_id = event.get("person_id", "—")
        behavior = event.get("behavior", "Unknown")
        label = f"{event_id} | Person {person_id} | {behavior}"
        event_labels.append(label)
        event_map[label] = event

    st.markdown("### Select an Event for Deep Analysis")

    selected_label = st.selectbox(
        "Event",
        event_labels,
        label_visibility="collapsed",
        key="event_selector",
    )

    selected_event = event_map[selected_label]

    st.markdown("### ⚡ Quick Intelligence Queries")

    quick_question = st.selectbox(
        "Choose a security question",
        [
            "Why was this event classified as CRITICAL?",
            "What evidence supports this event classification?",
            "How reliable is this alert?",
            "What are the main uncertainties?",
            "Compare the model decisions.",
            "What should a security operator do next?",
        ],
        key="quick_security_question",
    )

    if st.button(
        " Analyze Selected Event",
        use_container_width=True,
        key="analyze_selected_event",
    ):
        event_json = event_to_text(selected_event)
        prompt = f"""
You are BASEERA AI Security Analyst.
You analyze machine-generated surveillance evidence.
Selected Event:
{event_json}
Security Question:
{quick_question}
Provide a professional security intelligence analysis.
"""

        with st.spinner(" BASEERA is analyzing the incident..."):
            answer = ask_gemini(prompt)

        st.markdown("### BASEERA AI Security Intelligence Analysis")
        st.markdown(answer)