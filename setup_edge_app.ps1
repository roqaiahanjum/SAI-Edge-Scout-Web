# setup_edge_app.ps1
# Run this script to scaffold the Flutter project once Flutter is installed on the system PATH.

Write-Host "Checking for Flutter..."
if (Get-Command flutter -ErrorAction SilentlyContinue) {
    Write-Host "Flutter found, creating edge_app..."
    flutter create --org in.gov.sai edge_app
    cd edge_app
    
    Write-Host "Adding dependencies..."
    flutter pub add google_mlkit_pose_detection camera sqflite path_provider http
    
    Write-Host "Creating optimal folder structure..."
    mkdir -Force lib\core\utils
    mkdir -Force lib\core\theme
    mkdir -Force lib\core\database
    mkdir -Force lib\models
    mkdir -Force lib\services
    mkdir -Force lib\ui\screens\tactical_hud
    mkdir -Force lib\ui\screens\manual_entry
    mkdir -Force lib\ui\screens\calibration
    mkdir -Force lib\ui\widgets
    
    # Create empty placeholders
    New-Item -ItemType File -Force lib\core\utils\jitter_filter.dart
    New-Item -ItemType File -Force lib\core\database\local_vault.dart
    New-Item -ItemType File -Force lib\models\athlete_profile.dart
    New-Item -ItemType File -Force lib\services\pose_detection_service.dart

    Write-Host "Flutter edge_app setup complete!"
} else {
    Write-Host "Flutter is not installed or not in PATH. Please install Flutter and run this script." -ForegroundColor Red
}
