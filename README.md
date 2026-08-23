# 👀BASEERA Project
AI Security Command Center
Multi-model computer vision pipeline for behavioral intelligence, threat detection, and AI-assisted security analysis.

BASEERA is an AI-powered video surveillance and behavioral intelligence platform. It analyzes surveillance footage by combining pose estimation, multi-object tracking, temporal behavior modeling, weapon detection, segmentation, and an LLM-based security analyst into a single end-to-end pipeline, presented through an interactive Streamlit command center.

## 🎯 Project Overview
Problem
Traditional CCTV and surveillance systems are largely passive — footage is recorded but rarely analyzed in real time, and operators are left to manually review hours of video to spot suspicious activity, weapons, or altercations.

## Motivation
BASEERA explores how multiple, purpose-built Computer Vision and Deep Learning models can be combined into a single pipeline that goes beyond simple object detection — tracking individuals over time, interpreting their behavior, correlating detected weapons with specific people, and summarizing findings in a way a human security operator can quickly act on.

## What BASEERA Solves
Turns raw surveillance video into structured, per-person intelligence (identity, behavior, weapon association, severity).
Reduces the need to manually scrub through footage by surfacing ranked events (CRITICAL → LOW).
Adds a natural-language security analyst layer (Gemini) that explains why an event was flagged, grounded strictly in the pipeline's own telemetry.
How Multiple AI Models Cooperate
Rather than relying on a single model, BASEERA layers several specialized components — pose tracking, weapon detection, and multiple temporal behavior models (LSTM, Temporal Transformer, Spatio-Temporal Fusion) — so that behavior classification and weapon-to-person association can be cross-checked instead of relying on one signal alone.

## 🎬 System Demo
---
## 🧠 System Architecture
flowchart TD
    A[Input Surveillance Video] --> B[YOLO11 Pose Detection]
    B --> C[ByteTrack Identity Tracking]
    C --> D[Temporal Keypoint Sequence per Person]

    D --> E[LSTM Behavioral Classifier]
    D --> F[Temporal Transformer]
    D --> G[Spatio-Temporal Fusion Model]

    A --> H[Custom Weapon Detection - YOLO]
    H --> I[Weapon–Person Association<br/>BBox Center + Nearest Distance]
    C --> I

    I --> J[SAM ViT-B Segmentation<br/>on Weapon-Associated Persons]

    E --> K[Event Severity Scoring]
    F --> K
    G --> K
    I --> K

    K --> L[Advanced Event Intelligence Engine]

    L --> M[Gemini AI Security Analyst]
    L --> N[Streamlit Security Dashboard]

    M --> N
    N --> O[AI-Generated Incident Report]

## 🔄 End-to-End Workflow
|   Step | Stage                              | Description                                                                                                                                                                                       |
| -----: | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  **1** | **Upload**                         | A surveillance video is uploaded through the Streamlit dashboard (`dashboard/app.py`) or passed directly to `src/pipeline.py`.                                                                    |
|  **2** | **Pose Detection & Tracking**      | YOLO11-Pose (`yolo11n-pose.pt`) detects people frame-by-frame, while ByteTrack maintains identities and assigns a stable `person_id`.                                                             |
|  **3** | **Keypoint Sequencing**            | For each tracked person, 2D keypoints are accumulated into a temporal sequence containing **34 values** (17 keypoints × x,y).                                                                     |
|  **4** | **Weapon Detection**               | A custom-trained YOLO model (`weapon_detect_best.pt`) scans each frame for weapons.                                                                                                               |
|  **5** | **Weapon–Person Association**      | Detected weapons are matched to the nearest tracked person whose bounding box contains the weapon's center point, preventing every person in the frame from being flagged.                        |
|  **6** | **Behavior Classification**        | Once enough keypoint frames are accumulated, the LSTM (`temporal_behavior_model.pt`) predicts **Fighting, Robbery, Stealing, or Normal Movement** with a confidence score using a sliding window. |
|  **7** | **Segmentation (Conditional)**     | When SAM ViT-B is loaded and a weapon association is active, segmentation masks are periodically generated to highlight the relevant region.                                                      |
|  **8** | **Severity Scoring**               | Each tracked person's final record receives a severity level: **CRITICAL, HIGH, MEDIUM, or LOW**, based on weapon detection and behavior confidence.                                              |
|  **9** | **Event Aggregation**              | Results are consolidated into `events/event_history/parsed_events.json` and, when active, `advanced_event_history.json`.                                                                          |
| **10** | **Dashboard Rendering**            | The Streamlit dashboard loads the JSON output and displays KPIs, threat level, annotated video, per-person cards, and event timelines.                                                            |
| **11** | **AI Security Analyst (Optional)** | When a Gemini API key is configured, Gemini uses the machine-generated evidence to provide event explanations, answer security questions, and generate a complete incident report.                |


## 🤖 AI Components
| Component                     | Purpose                                                                                                 | Output                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **YOLO11 Pose**               | Detects people and estimates 2D body keypoints per frame                                                | Bounding boxes + 17 keypoints per person                                     |
| **ByteTrack**                 | Assigns and maintains a stable identity for each person across frames                                   | `person_id` with continuous tracking                                         |
| **LSTM (Temporal Behavior)**  | Classifies short-term behavior from a sliding window of keypoint sequences                              | Behavior label + confidence: Fighting / Robbery / Stealing / Normal Movement |
| **Temporal Transformer**      | Alternative temporal model for behavior classification using attention                                  | Behavior prediction                                                          |
| **Spatio-Temporal Fusion**    | Combines spatial and temporal cues for behavior/threat interpretation                                   | Fused behavior + confidence signal                                           |
| **Custom Weapon Detection**   | Detects weapons in each frame using a YOLO model trained on a custom weapon dataset                     | Weapon bounding boxes                                                        |
| **Weapon–Person Association** | Associates detected weapons with the most likely person using containment and nearest-distance matching | `weapon_detected` flag + `weapon_frames` count per person                    |
| **SAM ViT-B**                 | Generates a segmentation mask for persons associated with a detected weapon                             | Person segmentation mask overlay                                             |
| **Event Intelligence Engine** | Aggregates frame-level detections into structured, severity-ranked events                               | JSON event records                                                           |
| **Gemini Security Analyst**   | Interprets generated evidence in natural language based strictly on the provided telemetry              | Event explanations, chat answers, incident reports                           |

## 🚨 Threat & Event Intelligence
| Severity        | Condition                                                                                                                               | Meaning                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| 🔴 **CRITICAL** | A weapon is associated with the person, **regardless of behavior**, OR a weapon is detected together with `Fighting` behavior           | Immediate/highest-priority threat       |
| 🟠 **HIGH**     | `Fighting` behavior with confidence **≥ 0.70**                                                                                          | High-confidence aggressive behavior     |
| 🟡 **MEDIUM**   | `Fighting`, `Robbery`, or `Stealing` with confidence **< 0.70**                                                                         | Suspicious behavior requiring attention |
| 🟢 **LOW**      | No weapon association and no flagged behavior above the defined thresholds; includes `Normal Movement` and `Insufficient Temporal Data` | No significant threat detected          |

## 👤 Person Intelligence
For every tracked identity, VISIONGUARD records:

| Field                                              | Description                                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `person_id`                                        | Stable identity assigned by ByteTrack                                                       |
| `detected_behavior`                                | Best behavior label observed across the person's track using the LSTM sliding-window output |
| `behavior_confidence`                              | Confidence score associated with the detected behavior                                      |
| `weapon_detected`                                  | Indicates whether a weapon was ever associated with this person                             |
| `weapon_frames` / `weapon_detection_ratio`         | Number and proportion of frames in which a weapon was associated with the person            |
| `start_time_sec` / `end_time_sec` / `duration_sec` | Timeline showing when the person appeared and how long they were present in the video       |
| `severity`                                         | Final rule-based severity classification of the detected event                              |
| `transformer_behavior` / `transformer_confidence`  | Behavior prediction and confidence from the Temporal Transformer, when available            |
| `fusion_behavior` / `fusion_confidence`            | Behavior prediction and confidence from the Spatio-Temporal Fusion model, when available    |


## 🧠 Multi-Model Intelligence
The dashboard includes a Multi-Model Intelligence comparison chart (built with Plotly) that visualizes prediction counts side-by-side for:

LSTM — sliding-window prediction count
Temporal Transformer — prediction count
Spatio-Temporal Fusion — prediction count
This is intended to make it visible when models agree or diverge on a given clip, rather than presenting a single model's output as ground truth.

## 🧠 Gemini Security Analyst
BASEERA integrates Google's Gemini API (via the google-genai SDK) as an optional natural-language security analyst layered on top of the machine-generated evidence. It does not replace the detection pipeline — it only interprets the JSON telemetry the pipeline already produced.

Capabilities exposed in the dashboard:

Event-specific analysis — select any recorded event and ask a predefined security question (e.g. "Why was this event classified as CRITICAL?").
Quick Intelligence Queries — a curated set of common security questions.
Security Chat — free-form questions about any tracked person, behavior, or event in the current run.
Incident Report Generator — produces a structured, multi-section report (Executive Summary, Threat Assessment, Person Intelligence, Weapon Analysis, Behavioral Analysis, Multi-Model comparison, Event Timeline, Risk Assessment, Recommended Actions, AI Limitations).
Evidence-grounded prompting: Every prompt sent to Gemini explicitly instructs the model not to invent people, weapons, behaviors, timestamps, or confidence values, and to clearly separate detected evidence from model interpretation and uncertainty.

Uncertainty handling: Responses are expected to flag low-confidence or conflicting model outputs and recommend human verification rather than asserting certainty.

Configuring GEMINI_API_KEY safely
Gemini access is entirely optional — the dashboard runs without it, simply showing the analyst as offline.

PowerShell (current session only):

$env:GEMINI_API_KEY="YOUR_API_KEY"
Recommended (persistent, local-only): create a .env file in the project root:

GEMINI_API_KEY=YOUR_API_KEY
and ensure it is excluded via .gitignore. Never commit an API key to source control.

## 🖥️ Streamlit Dashboard
The dashboard (dashboard/app.py) is the primary interface for VISIONGUARD, titled "VISIONGUARD — AI Security Command Center".

Screenshot Gallery
Replace the placeholders below with your own captured screenshots (see the Screenshots I Should Capture section in the project notes).

Command Center Overview VISIONGUARD Dashboard The main command center: header, system status, video upload, and AI stack status in the sidebar.

Threat Intelligence Overview Threat Intelligence Overview Frame count, tracked persons, weapon detection frames, critical events, and model prediction counts at a glance.

Person Intelligence Person Intelligence Per-person cards showing behavior, confidence, weapon association, timeline, and multi-model outputs.

Advanced Event Intelligence Event Intelligence Severity-ranked event feed generated from the Advanced Event Engine.

Multi-Model Comparison Model Comparison Side-by-side prediction counts for LSTM, Temporal Transformer, and Spatio-Temporal Fusion.

Gemini Security Analyst Gemini Analyst Event-specific analysis and free-form security chat grounded in pipeline telemetry.

Incident Report Incident Report AI-generated, structured incident report with export to .txt.

## 📊 Outputs
Each pipeline run produces:

| Output                     | Location                                           | Description                                                     |
| -------------------------- | -------------------------------------------------- | --------------------------------------------------------------- |
| **Annotated video**        | `data/outputs/annotated_pipeline_out.mp4`          | Original video with pose, weapon, and severity overlays         |
| **Web-optimized video**    | `data/outputs/annotated_pipeline_web.mp4`          | H.264-encoded version generated via FFmpeg for browser playback |
| **Tracking results**       | `data/outputs/tracking_results.csv` / `.json`      | Raw per-frame tracking data                                     |
| **Parsed events**          | `events/event_history/parsed_events.json`          | Structured per-person summary consumed by the dashboard         |
| **Advanced event history** | `events/event_history/advanced_event_history.json` | Output of the Advanced Event Intelligence Engine, when active   |
| **Pipeline report**        | `events/pipeline_final_report.json`                | Run-level summary report                                        |
put of the Advanced Event Intelligence Engine, when active
Pipeline report	events/pipeline_final_report.json	Run-level summary report
## 📁 Project Structure
BASEERA/
│   app.py
│   detector.py
│   run_pipeline.py
│   server.py
│   requirements.txt
│   Dockerfile
│   docker-compose.yml
│   yolo11n-pose.pt / yolo11n.pt / yolov8n.pt
│
├── dashboard/
│   └── app.py                  # Streamlit AI Security Command Center
│
├── data/
│   ├── inputs/                 # Sample surveillance videos and images
│   ├── outputs/                # Annotated video and tracking outputs
│   └── processed_features/     # Extracted keypoint features by behavior class
│
├── events/
│   └── event_history/          # Parsed and advanced event JSON records
│
├── models/                     # Trained weights (pose, weapon, LSTM, transformer, fusion, SAM)
│
├── notebooks/
│   └── Pose_Estimation.ipynb
│
├── runs/detect/train/          # YOLO training run artifacts
│
├── src/
│   ├── pipeline.py             # Main end-to-end inference pipeline
│   ├── event_engine.py         # Advanced Event Intelligence Engine
│   ├── compare_models.py
│   ├── threat_viz.py
│   ├── detection/               # YOLO weapon detection + tracking scripts
│   ├── events/                  # Event building utilities
│   ├── features/                # Keypoint feature extraction
│   ├── models/                  # Model definitions (LSTM, Transformer, Fusion, SAM, ViT)
│   ├── pose/                    # Pose estimation test scripts
│   └── visionary/                # LLM integration test scripts
│
├── tests/
│   └── test_data/
│
└── Weapon-yolo8-1/              # Roboflow-exported YOLO weapon detection dataset
    ├── data.yaml
    ├── train/ valid/ test/
(Cache directories such as __pycache__ and the local venv/ virtual environment are omitted for clarity.)

## ⚙️ Installation
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd VISIONGUARD

python -m venv venv
Activate the virtual environment (Windows):

venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
## 🔑 Gemini Configuration
Gemini is optional. To enable the AI Security Analyst, set your API key before launching the dashboard.

PowerShell:

$env:GEMINI_API_KEY="YOUR_API_KEY"
Without a configured key, the dashboard will run normally with the Gemini analyst shown as offline.

## ▶️ Running the Application
python -m streamlit run dashboard\app.py
Then, in the Streamlit interface:

Open the sidebar and upload a surveillance video (.mp4, .avi, .mov, .mkv, .webm, or .mpeg4).
Click "🚀 RUN FULL AI ANALYSIS".
Wait for the pipeline to process the video and generate outputs.
Review the Threat Intelligence Overview, annotated video, person cards, and events.
Optionally, use the Gemini Security Analyst to query the results or generate an incident report.
## 🧪 Testing
The repository includes:

tests/test_data/event.json — sample event data for validating event-handling logic.
src/detection/test_detection.py — weapon/object detection test script.
src/pose/test_pose.py — pose estimation test script.
src/visionary/test_llm.py — LLM integration test script.
(No automated test runner configuration, such as pytest.ini, is currently present in the repository — tests are standalone scripts.)

## 📈 Results
Training artifacts are available under runs/detect/train/ (results.csv, training batch visualizations, and final weights best.pt / last.pt) for the custom weapon detection model.

No finalized, independently-verified accuracy/precision/recall figures are included in this document. Quantitative performance should be assessed directly from runs/detect/train/results.csv and reported separately once validated, rather than restated here without verification.

## 🔐 Security & Privacy
API keys (e.g. GEMINI_API_KEY) must be stored as environment variables or in a local, git-ignored .env file — never committed to source control.
Surveillance video may contain personally identifiable information; ensure it is stored, processed, and shared in compliance with applicable privacy regulations and organizational policy.
VISIONGUARD should only be deployed with appropriate authorization from the premises/asset owner.
All AI predictions (behavior classification, weapon association, severity scoring, Gemini analysis) are decision support, not definitive determinations, and should be verified by a human operator — particularly for CRITICAL-severity events.
## ⚠️ Limitations
Model confidence scores reflect statistical certainty of the model, not ground-truth certainty of an actual event.
Detection errors (false positives/negatives) can occur in pose detection, weapon detection, and behavior classification.
Temporal behavior models (LSTM, Transformer, Fusion) require a minimum number of accumulated frames per person; short tracks may be reported as having insufficient temporal data.
The Gemini Security Analyst depends on API key configuration and external service availability; its output quality depends entirely on the accuracy of the upstream detection/tracking pipeline.
SAM segmentation quality depends on the accuracy of the underlying weapon/person detections that trigger it.
Human verification is recommended before acting on any CRITICAL-severity alert.
## 🚀 Future Improvements
Real-time CCTV / RTSP streaming support (currently file-based video processing).
Larger and more diverse weapon and behavior training datasets.
Improved temporal event localization (precise start/end boundaries of an incident).
Model optimization for faster inference (quantization, batching, GPU scheduling).
Edge deployment support for on-premises hardware.
Configurable alert notifications (email/SMS/webhook) for CRITICAL events.
Multi-camera / multi-stream support with a unified dashboard.
Authentication and role-based access control for the dashboard.
More rigorous, published evaluation metrics across all models.
## 👨‍💻 Project Team
Malak Sherif - [LinkedIn Profile](https://www.linkedin.com/in/malak-sherif-b03138357)
Shahinaz Salah - [LinkedIn Profile](https://www.linkedin.com/in/shahinaz-salah-67409a2a4)
