# SAI Edge Scout Platform - Architecture & Scaffolding Guide

## 1. Exact Terminal Commands

### Flutter Edge App (Mobile)
Run these commands in PowerShell or Git Bash once Flutter is installed and added to `PATH`:

```powershell
# 1. Initialize the Flutter project with Android/iOS support
flutter create --org in.gov.sai edge_app
cd edge_app

# 2. Add required dependencies for ML tracking, local vault, and API
flutter pub add google_mlkit_pose_detection camera sqflite path_provider http

# 3. Add state management and responsive UI utilities (Optional but recommended)
flutter pub add provider get_it
```

### Streamlit Command Center (Web Backend)
Run these commands to set up the Python environment:

```powershell
# 1. Create the backend directory
mkdir command_center
cd command_center

# 2. Initialize a separate Python Virtual Environment
python -m venv venv

# 3. Activate the Virtual Environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# (Use `source venv/bin/activate` if on Mac/Linux)

# 4. Install the required Data, Dashboard, and Video dependencies
pip install streamlit pymongo moviepy pandas folium pydeck python-dotenv
```

---

## 2. Optimal Folder Structures

### Flutter Mobile App (`edge_app/lib/`)
Designed for an offline-first MVVM/Clean Architecture approach:
```text
edge_app/
├── android/                  # Android specific configurations (Permissions for Camera, Network, etc.)
├── ios/                      # iOS specific configurations (Info.plist for Camera usage)
├── lib/
│   ├── core/
│   │   ├── utils/
│   │   │   ├── jitter_filter.dart     # Custom 5-frame low-pass smoothing filter algorithm
│   │   │   └── math_utils.dart        # Z-axis depth & jump metric calculators
│   │   ├── theme/
│   │   │   └── app_theme.dart         # Material 3 Saffron & Navy configurations
│   │   └── database/
│   │       ├── local_vault.dart       # SQLite database initialization & query engine
│   │       └── sync_manager.dart      # Background auto-sync logic for MongoDB Atlas
│   ├── models/
│   │   └── athlete_profile.dart       # Dart data class mapped precisely to the PRD JSON Schema
│   ├── services/
│   │   ├── pose_detection_service.dart # Wrapper for google_mlkit_pose_detection
│   │   └── api_service.dart           # REST interface for pushing data to Command Center
│   ├── ui/
│   │   ├── screens/
│   │   │   ├── tactical_hud/
│   │   │   │   └── hud_screen.dart    # Camera interface with real-time pose overlays
│   │   │   ├── manual_entry/
│   │   │   │   └── form_screen.dart   # Annexure A manual data input UI
│   │   │   └── calibration/
│   │   │       └── baseline_screen.dart # Auto-calculate height UI state
│   │   └── widgets/
│   │       └── state_indicator.dart   # (Waiting -> Calibrating -> Active) HUD widget
│   └── main.dart                      # App entry point, offline initialization
└── pubspec.yaml
```

### Streamlit Dashboard & Backend (`command_center/`)
Designed to be stateless and fast for real-time map visualization and data ingestion:
```text
command_center/
├── app.py                      # Main Streamlit dashboard entry point
├── requirements.txt            # Python dependencies lists
├── .env                        # MongoDB credentials and secret keys
├── components/
│   ├── leaderboard.py          # Pandas-driven leaderboard widget
│   └── heatmap.py              # Pydeck/Folium geographic talent density mapper
├── services/
│   ├── mongo_client.py         # PyMongo wrapper for connecting to MongoDB Atlas
│   └── transcoder.py           # Moviepy background video transcoder
├── api/
│   └── server.py               # Lightweight FastAPI or Flask route for Flutter to Sync to (optional, or streamline via Streamlit/Mongo direct if securely configured, but API approach is better)
├── data/                       # Temporary folder for local caching, if needed
└── utils/
    └── olympic_analytics.py    # Predictive analytics for age-to-metric ratios
```

---

## 3. QA / Audit Results

- **Python Environment**: `command_center` structure initialized successfully via script.
- **Flutter Environment**: `flutter` command was not detected on the system PATH. The `edge_app` scaffolding was aborted locally, but exact setup commands are provided above. Wait until Flutter is installed to run them.
