import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLabel, QProgressBar, QComboBox,
    QGroupBox, QListWidget, QListWidgetItem, QMessageBox, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon
import whisper
import fcntl
import tempfile
import time
from datetime import timedelta

# Detect if running in PyInstaller bundle
def get_ffmpeg_path():
    """Get the path to ffmpeg, whether bundled or system-installed"""
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        bundle_dir = sys._MEIPASS
        ffmpeg_path = os.path.join(bundle_dir, 'ffmpeg')
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
    # Fallback to system ffmpeg
    return 'ffmpeg'

# Set ffmpeg path for whisper
FFMPEG_PATH = get_ffmpeg_path()
if FFMPEG_PATH != 'ffmpeg':
    # Only set if we found a bundled version
    os.environ['PATH'] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ.get('PATH', '')



class TranscriptionThread(QThread):
    """Thread for running Whisper transcription without blocking the UI"""
    finished = pyqtSignal(str)
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    time_remaining = pyqtSignal(str)
    error = pyqtSignal(str)

    # Estimated processing time per second of audio for different models
    MODEL_SPEED = {
        'tiny': 0.1,    # 0.1 seconds per second of audio
        'base': 0.2,
        'small': 0.5,
        'medium': 1.0,
        'large': 2.0
    }

    def __init__(self, audio_file, model_name="base", language=None):
        super().__init__()
        self.audio_file = audio_file
        self.model_name = model_name
        self.language = language
        self._is_running = True

    def run(self):
        try:
            start_time = time.time()

            # Step 1: Loading model (10% of progress)
            self.progress.emit("Loading Whisper model...")
            self.progress_percent.emit(5)
            model = whisper.load_model(self.model_name)
            self.progress_percent.emit(10)

            # Get audio duration for time estimation
            try:
                import subprocess
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', self.audio_file],
                    capture_output=True, text=True, timeout=5
                )
                audio_duration = float(result.stdout.strip())
            except:
                audio_duration = 60  # Default estimate if we can't get duration

            # Estimate total time based on model
            speed_factor = self.MODEL_SPEED.get(self.model_name, 0.5)
            estimated_total_time = audio_duration * speed_factor

            # Step 2: Start transcription (10% to 90%)
            self.progress.emit("Transcribing audio...")
            self.progress_percent.emit(15)

            # Start a timer thread to update progress during transcription
            transcription_start = time.time()

            # Run transcription in a way that allows us to update progress
            # We'll simulate progress since Whisper doesn't provide callbacks
            def update_progress():
                while self._is_running:
                    elapsed = time.time() - transcription_start
                    progress = min(85, int(15 + (elapsed / estimated_total_time) * 70))
                    self.progress_percent.emit(progress)

                    remaining = max(0, estimated_total_time - elapsed)
                    remaining_str = str(timedelta(seconds=int(remaining))).split('.')[0]
                    self.time_remaining.emit(f"Est. time remaining: {remaining_str}")

                    time.sleep(0.5)

            # Start progress updater in background
            from threading import Thread
            progress_thread = Thread(target=update_progress, daemon=True)
            progress_thread.start()

            # Actually run transcription
            if self.language and self.language != "Auto-detect":
                result = model.transcribe(self.audio_file, language=self.language)
            else:
                result = model.transcribe(self.audio_file)

            # Stop progress updates
            self._is_running = False
            progress_thread.join(timeout=1)

            # Step 3: Finalizing (90% to 100%)
            self.progress.emit("Finalizing transcription...")
            self.progress_percent.emit(95)
            self.time_remaining.emit("Almost done...")

            self.progress_percent.emit(100)
            self.time_remaining.emit("Complete!")
            self.finished.emit(result["text"])

        except Exception as e:
            self._is_running = False
            self.error.emit(f"Error during transcription: {str(e)}")


class ModelDownloadThread(QThread):
    """Thread for downloading Whisper models without blocking the UI"""
    finished = pyqtSignal(str)
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    use_indeterminate = pyqtSignal(bool)  # Signal to switch to indeterminate mode
    error = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self._is_running = True
        self._last_percent = 0
        self._real_progress_working = False

    def run(self):
        try:
            self.progress.emit(f"Preparing download: {self.model_name}")

            # Check if model is already cached
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "whisper"
            model_path = cache_dir / f"{self.model_name}.pt"

            if model_path.exists():
                # Model already exists, just load it
                self.progress.emit("Model already cached, loading...")
                self.use_indeterminate.emit(True)
                whisper.load_model(self.model_name)
                self.progress_percent.emit(100)
                self.finished.emit(f"Model {self.model_name} loaded from cache")
                return

            # Start with indeterminate progress (spinner)
            self.use_indeterminate.emit(True)
            self.progress.emit(f"Downloading {self.model_name} model...")

            # Try to patch torch hub download to track real progress
            try:
                import torch.hub
                import urllib.request

                original_download = torch.hub.download_url_to_file

                def progress_hook(block_num, block_size, total_size):
                    """Real download progress hook - NO SIMULATION"""
                    if not self._real_progress_working and total_size > 0:
                        # First time we get real data, switch to percentage mode
                        self._real_progress_working = True
                        self.use_indeterminate.emit(False)

                    if total_size > 0:
                        downloaded = block_num * block_size
                        percent = min(99, int((downloaded / total_size) * 100))
                        if percent > self._last_percent:
                            self._last_percent = percent
                            self.progress_percent.emit(percent)
                            # Show MB downloaded
                            mb_down = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            self.progress.emit(f"Downloading: {mb_down:.1f} / {mb_total:.1f} MB ({percent}%)")

                def patched_download(url, dst, hash_prefix=None, progress=True):
                    """Patched download with real progress hook"""
                    try:
                        urllib.request.urlretrieve(url, dst, reporthook=progress_hook)
                    except Exception:
                        # Fallback to original - still no simulation
                        original_download(url, dst, hash_prefix, progress)

                # Replace the download function temporarily
                torch.hub.download_url_to_file = patched_download

                try:
                    whisper.load_model(self.model_name)
                finally:
                    # Always restore original
                    torch.hub.download_url_to_file = original_download

            except Exception:
                # If patching fails, download with indeterminate progress
                self.progress.emit(f"Downloading {self.model_name} (progress unavailable)...")
                whisper.load_model(self.model_name)

            self._is_running = False
            self.progress_percent.emit(100)
            self.progress.emit("Download complete!")
            self.finished.emit(f"Successfully downloaded {self.model_name} model")

        except Exception as e:
            self._is_running = False
            self.error.emit(f"Error downloading model: {str(e)}")


class WhispersViolinsApp(QMainWindow):
    # Whisper supported languages
    LANGUAGES = {
        "Auto-detect": None,
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Dutch": "nl",
        "Russian": "ru",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Arabic": "ar",
        "Hindi": "hi",
    }

    def __init__(self):
        super().__init__()
        self.audio_file = None
        self.transcription_thread = None
        self.download_thread = None
        self.init_ui()

    def closeEvent(self, event):
        """Handle window close event - stop all running threads and QUIT"""
        # Stop transcription thread if running
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread._is_running = False
            self.transcription_thread.wait(1000)  # Wait up to 1 second
            if self.transcription_thread.isRunning():
                self.transcription_thread.terminate()

        # Stop download thread if running
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread._is_running = False
            self.download_thread.wait(1000)
            if self.download_thread.isRunning():
                self.download_thread.terminate()

        # Accept the close event
        event.accept()

        # Force application to quit - don't allow relaunch
        QApplication.instance().quit()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("whispersViolins - Audio Transcription")
        self.setMinimumSize(900, 700)

        # Set window icon
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🎻 whispersViolins")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Arial", 48, QFont.Weight.Bold)
        title.setFont(title_font)
        main_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Powered by OpenAI Whisper")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont("Arial", 24)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #666;")
        main_layout.addWidget(subtitle)

        # Create tabs for main interface and model management
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 30))
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
            }
            QTabBar::tab {
                padding: 15px 30px;
                margin: 2px;
                background-color: #e0e0e0;
                color: #666666;
                border: 1px solid #ccc;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border: 1px solid #2196F3;
            }
            QTabBar::tab:hover:!selected {
                background-color: #d0d0d0;
            }
        """)

        # Transcription tab
        transcription_tab = QWidget()
        transcription_layout = QVBoxLayout(transcription_tab)
        transcription_layout.setSpacing(15)

        # Settings row (Model + Language)
        settings_layout = QHBoxLayout()

        # Model selection (will be populated dynamically)
        model_label = QLabel("Model:")
        model_label.setFont(QFont("Arial", 22))
        settings_layout.addWidget(model_label)

        # Container for model selector or "Go to Model Management" button
        self.model_selector_container = QWidget()
        self.model_selector_layout = QHBoxLayout(self.model_selector_container)
        self.model_selector_layout.setContentsMargins(0, 0, 0, 0)

        # Model combo (will be shown/hidden based on downloaded models)
        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("Arial", 28))
        self.model_combo.setMinimumHeight(50)
        self.model_selector_layout.addWidget(self.model_combo)

        # "Go to Model Management" button (shown when no models downloaded)
        self.go_to_models_btn = QPushButton("Download Models First")
        self.go_to_models_btn.setFont(QFont("Arial", 22))
        self.go_to_models_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.go_to_models_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))  # Switch to Model Management tab
        self.model_selector_layout.addWidget(self.go_to_models_btn)
        self.go_to_models_btn.setVisible(False)  # Hidden by default

        settings_layout.addWidget(self.model_selector_container)

        settings_layout.addSpacing(20)

        # Language selection
        language_label = QLabel("Language:")
        language_label.setFont(QFont("Arial", 22))
        self.language_combo = QComboBox()
        self.language_combo.addItems(list(self.LANGUAGES.keys()))
        self.language_combo.setCurrentText("Auto-detect")
        self.language_combo.setFont(QFont("Arial", 28))
        self.language_combo.setMinimumHeight(50)
        settings_layout.addWidget(language_label)
        settings_layout.addWidget(self.language_combo)

        settings_layout.addStretch()
        transcription_layout.addLayout(settings_layout)

        # File selection area
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setFont(QFont("Arial", 22))
        self.file_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        file_layout.addWidget(self.file_label, 1)

        self.select_file_btn = QPushButton("Select Audio File")
        self.select_file_btn.setFont(QFont("Arial", 22))
        self.select_file_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.select_file_btn.clicked.connect(self.select_audio_file)
        file_layout.addWidget(self.select_file_btn)
        transcription_layout.addLayout(file_layout)

        # Transcribe button
        self.transcribe_btn = QPushButton("Start Transcription")
        self.transcribe_btn.setFont(QFont("Arial", 26, QFont.Weight.Bold))
        self.transcribe_btn.setMinimumHeight(50)
        self.transcribe_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0a6bc4;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.transcribe_btn.setEnabled(False)
        transcription_layout.addWidget(self.transcribe_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        transcription_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Arial", 20))
        self.status_label.setStyleSheet("color: #666;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        transcription_layout.addWidget(self.status_label)

        # Transcription output area
        output_label = QLabel("Transcription:")
        output_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        transcription_layout.addWidget(output_label)

        self.output_text = QTextEdit()
        self.output_text.setFont(QFont("Arial", 22))
        self.output_text.setPlaceholderText("Transcribed text will appear here...")
        self.output_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: #fafafa;
                color: #000000;
            }
        """)
        transcription_layout.addWidget(self.output_text, 1)

        # Buttons row
        buttons_layout = QHBoxLayout()

        self.copy_btn = QPushButton("Copy Text")
        self.copy_btn.setFont(QFont("Arial", 30))
        self.copy_btn.setMinimumHeight(52)
        self.copy_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        self.copy_btn.clicked.connect(self.copy_text)
        buttons_layout.addWidget(self.copy_btn)

        self.save_btn = QPushButton("Save to File")
        self.save_btn.setFont(QFont("Arial", 30))
        self.save_btn.setMinimumHeight(52)
        self.save_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.save_btn.clicked.connect(self.save_to_file)
        buttons_layout.addWidget(self.save_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFont(QFont("Arial", 30))
        self.clear_btn.setMinimumHeight(52)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_output)
        buttons_layout.addWidget(self.clear_btn)

        buttons_layout.addStretch()
        transcription_layout.addLayout(buttons_layout)

        # Add transcription tab
        self.tabs.addTab(transcription_tab, "Transcription")

        # Model Management tab
        self.create_model_management_tab()

        # Add tabs to main layout
        main_layout.addWidget(self.tabs)

    def create_model_management_tab(self):
        """Create the model management tab"""
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        model_layout.setSpacing(15)
        model_layout.setContentsMargins(15, 15, 15, 15)

        # Info section
        info_label = QLabel("Manage Whisper Models")
        info_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        model_layout.addWidget(info_label)

        info_text = QLabel("Download and manage Whisper AI models. Larger models provide better accuracy but require more disk space and processing time.")
        info_text.setFont(QFont("Arial", 20))
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #666; margin-bottom: 10px;")
        model_layout.addWidget(info_text)

        # Model info group
        info_group = QGroupBox("Model Information")
        info_group.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        info_group_layout = QVBoxLayout()

        model_info_text = """<table style='width: 100%;'>
        <tr><td><b>tiny</b></td><td>~39 MB</td><td>Fastest, lowest accuracy</td></tr>
        <tr><td><b>base</b></td><td>~74 MB</td><td>Fast, good for testing</td></tr>
        <tr><td><b>small</b></td><td>~244 MB</td><td>Balanced speed/accuracy</td></tr>
        <tr><td><b>medium</b></td><td>~769 MB</td><td>High accuracy, slower</td></tr>
        <tr><td><b>large</b></td><td>~1550 MB</td><td>Best accuracy, slowest</td></tr>
        </table>"""

        model_info_label = QLabel(model_info_text)
        model_info_label.setFont(QFont("Arial", 20))
        model_info_label.setWordWrap(True)
        info_group_layout.addWidget(model_info_label)
        info_group.setLayout(info_group_layout)
        model_layout.addWidget(info_group)

        # Available models section
        available_label = QLabel("Download Model:")
        available_label.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        model_layout.addWidget(available_label)

        download_layout = QHBoxLayout()

        self.download_model_combo = QComboBox()
        self.download_model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.download_model_combo.setFont(QFont("Arial", 28))
        self.download_model_combo.setMinimumHeight(50)
        download_layout.addWidget(self.download_model_combo, 1)

        self.download_model_btn = QPushButton("Download")
        self.download_model_btn.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        self.download_model_btn.setMinimumHeight(40)
        self.download_model_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 30px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.download_model_btn.clicked.connect(self.download_model)
        download_layout.addWidget(self.download_model_btn)

        model_layout.addLayout(download_layout)

        # Download progress
        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setVisible(False)
        self.download_progress_bar.setTextVisible(True)
        self.download_progress_bar.setFormat("%p%")
        self.download_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        model_layout.addWidget(self.download_progress_bar)

        self.download_progress_label = QLabel("")
        self.download_progress_label.setFont(QFont("Arial", 20))
        self.download_progress_label.setStyleSheet("color: #2196F3;")
        model_layout.addWidget(self.download_progress_label)

        # Downloaded models section
        downloaded_label = QLabel("Downloaded Models:")
        downloaded_label.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        model_layout.addWidget(downloaded_label)

        self.models_list = QListWidget()
        self.models_list.setFont(QFont("Arial", 20))
        self.models_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: #fafafa;
                color: #000000;
            }
        """)
        model_layout.addWidget(self.models_list, 1)

        # Delete button (no more refresh button)
        model_actions_layout = QHBoxLayout()

        self.delete_model_btn = QPushButton("Delete Selected")
        self.delete_model_btn.setFont(QFont("Arial", 20))
        self.delete_model_btn.setMinimumHeight(35)
        self.delete_model_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.delete_model_btn.clicked.connect(self.delete_model)
        model_actions_layout.addWidget(self.delete_model_btn)

        model_actions_layout.addStretch()
        model_layout.addLayout(model_actions_layout)

        # Disk space info
        self.disk_space_label = QLabel("")
        self.disk_space_label.setFont(QFont("Arial", 18))
        self.disk_space_label.setStyleSheet("color: #666;")
        model_layout.addWidget(self.disk_space_label)

        # Add model management tab
        self.tabs.addTab(model_tab, "Model Management")

        # Initialize the models list
        self.refresh_models_list()

        # Update the transcription tab model selector
        self.update_model_selector()

    def update_model_selector(self):
        """Update the model selector based on downloaded models"""
        models = self.get_downloaded_models()

        if not models:
            # No models downloaded - show button to go to model management
            self.model_combo.setVisible(False)
            self.go_to_models_btn.setVisible(True)
            self.transcribe_btn.setEnabled(False)
        else:
            # Models available - show dropdown
            self.model_combo.setVisible(True)
            self.go_to_models_btn.setVisible(False)

            # Extract unique model names (base name without hash/extension)
            model_names = set()
            for model in models:
                # Model filenames are like "base.pt" or "base.en.pt" etc
                name = model['name'].split('.')[0]
                model_names.add(name)

            # Update combo box
            current_selection = self.model_combo.currentText() if self.model_combo.count() > 0 else None
            self.model_combo.clear()
            self.model_combo.addItems(sorted(model_names))

            # Restore previous selection if still available
            if current_selection and current_selection in model_names:
                self.model_combo.setCurrentText(current_selection)

            # Enable transcribe button only if a file is selected
            if self.audio_file:
                self.transcribe_btn.setEnabled(True)

    def select_audio_file(self):
        """Open file dialog to select an audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac);;All Files (*)"
        )

        if file_path:
            self.audio_file = file_path
            self.file_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.file_label.setStyleSheet("padding: 10px; border: 2px solid #4CAF50; border-radius: 5px; color: #4CAF50;")

            # Only enable transcribe button if models are available
            models = self.get_downloaded_models()
            if models:
                self.transcribe_btn.setEnabled(True)
                self.status_label.setText("Ready to transcribe")
            else:
                self.transcribe_btn.setEnabled(False)
                self.status_label.setText("Please download a model first")

    def start_transcription(self):
        """Start the transcription process in a separate thread"""
        if not self.audio_file:
            return

        # Disable buttons during transcription
        self.transcribe_btn.setEnabled(False)
        self.select_file_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.language_combo.setEnabled(False)

        # Show and start progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)  # 0-100% progress
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")  # Show percentage

        # Clear previous output
        self.output_text.clear()

        # Get selected language
        selected_language_name = self.language_combo.currentText()
        language_code = self.LANGUAGES.get(selected_language_name)

        # Create and start transcription thread
        model_name = self.model_combo.currentText()
        self.transcription_thread = TranscriptionThread(self.audio_file, model_name, language_code)
        self.transcription_thread.finished.connect(self.on_transcription_finished)
        self.transcription_thread.progress.connect(self.on_transcription_progress)
        self.transcription_thread.progress_percent.connect(self.on_progress_percent_update)
        self.transcription_thread.time_remaining.connect(self.on_time_remaining_update)
        self.transcription_thread.error.connect(self.on_transcription_error)
        self.transcription_thread.start()

    def on_transcription_progress(self, message):
        """Update progress status"""
        self.status_label.setText(message)

    def on_progress_percent_update(self, percent):
        """Update progress bar percentage"""
        self.progress_bar.setValue(percent)

    def on_time_remaining_update(self, time_str):
        """Update time remaining display"""
        current_text = self.status_label.text()
        # Update status to show time remaining alongside current status
        if "time remaining" in time_str.lower() or "complete" in time_str.lower() or "almost" in time_str.lower():
            self.status_label.setText(time_str)

    def on_transcription_finished(self, text):
        """Handle completed transcription"""
        self.output_text.setPlainText(text)
        self.status_label.setText("Transcription complete!")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

        # Set progress bar to 100%
        self.progress_bar.setValue(100)

        # Re-enable buttons
        self.transcribe_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_transcription_error(self, error_message):
        """Handle transcription errors"""
        self.output_text.setPlainText(f"Error: {error_message}")
        self.status_label.setText("Transcription failed")
        self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")

        # Re-enable buttons
        self.transcribe_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.progress_bar.setVisible(False)

    def copy_text(self):
        """Copy transcribed text to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.output_text.toPlainText())
        self.status_label.setText("Text copied to clipboard!")
        self.status_label.setStyleSheet("color: #2196F3;")

    def save_to_file(self):
        """Save transcribed text to a file"""
        text = self.output_text.toPlainText()
        if not text:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcription",
            "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.status_label.setText(f"Saved to {os.path.basename(file_path)}")
                self.status_label.setStyleSheet("color: #4CAF50;")
            except Exception as e:
                self.status_label.setText(f"Error saving file: {str(e)}")
                self.status_label.setStyleSheet("color: #f44336;")

    def clear_output(self):
        """Clear the output text area"""
        self.output_text.clear()
        self.status_label.setText("")
        self.status_label.setStyleSheet("color: #666;")

    def get_models_cache_dir(self):
        """Get the Whisper models cache directory"""
        # Whisper stores models in ~/.cache/whisper on Unix-like systems
        home = Path.home()
        cache_dir = home / ".cache" / "whisper"
        return cache_dir

    def get_downloaded_models(self):
        """Get list of downloaded Whisper models"""
        cache_dir = self.get_models_cache_dir()
        if not cache_dir.exists():
            return []

        models = []
        for file in cache_dir.glob("*.pt"):
            # Extract model name from filename
            name = file.stem
            size = file.stat().st_size
            size_mb = size / (1024 * 1024)
            models.append({"name": name, "path": file, "size_mb": size_mb})

        return models

    def refresh_models_list(self):
        """Refresh the list of downloaded models"""
        self.models_list.clear()
        models = self.get_downloaded_models()

        if not models:
            item = QListWidgetItem("No models downloaded yet")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.models_list.addItem(item)
        else:
            for model in models:
                item_text = f"{model['name']} - {model['size_mb']:.1f} MB"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, model)
                self.models_list.addItem(item)

        # Update disk space info
        cache_dir = self.get_models_cache_dir()
        if cache_dir.exists():
            total_size = sum(m['size_mb'] for m in models)
            self.disk_space_label.setText(f"Total cache size: {total_size:.1f} MB ({len(models)} models)")
        else:
            self.disk_space_label.setText("Cache directory not created yet")

        # Update the transcription tab model selector
        self.update_model_selector()

    def download_model(self):
        """Download a Whisper model"""
        model_name = self.download_model_combo.currentText()

        # Check if already downloaded
        models = self.get_downloaded_models()
        if any(model_name in m['name'] for m in models):
            reply = QMessageBox.question(
                self,
                "Model Already Downloaded",
                f"The {model_name} model appears to be already downloaded. Download again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Disable download button and show progress bar
        self.download_model_btn.setEnabled(False)
        self.download_progress_bar.setVisible(True)
        self.download_progress_bar.setValue(0)
        self.download_progress_bar.setRange(0, 100)
        self.download_progress_label.setText(f"Downloading {model_name} model...")

        # Start download in thread
        self.download_thread = ModelDownloadThread(model_name)
        self.download_thread.finished.connect(self.on_download_finished)
        self.download_thread.progress.connect(self.on_download_progress)
        self.download_thread.progress_percent.connect(self.on_download_progress_percent)
        self.download_thread.use_indeterminate.connect(self.on_download_use_indeterminate)
        self.download_thread.error.connect(self.on_download_error)
        self.download_thread.start()

    def on_download_progress(self, message):
        """Update download progress"""
        self.download_progress_label.setText(message)

    def on_download_progress_percent(self, percent):
        """Update download progress bar percentage"""
        self.download_progress_bar.setValue(percent)

    def on_download_use_indeterminate(self, use_indeterminate):
        """Switch between indeterminate and percentage progress modes"""
        if use_indeterminate:
            # Switch to indeterminate (spinner) mode
            self.download_progress_bar.setRange(0, 0)
        else:
            # Switch to percentage mode
            self.download_progress_bar.setRange(0, 100)

    def on_download_finished(self, message):
        """Handle completed download"""
        self.download_progress_label.setText(message)
        self.download_progress_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.download_model_btn.setEnabled(True)
        self.download_progress_bar.setValue(100)
        self.download_progress_bar.setVisible(False)

        # Auto-refresh the models list
        self.refresh_models_list()

        # Reset label after processing
        QApplication.processEvents()

    def on_download_error(self, error_message):
        """Handle download errors"""
        self.download_progress_label.setText(error_message)
        self.download_progress_label.setStyleSheet("color: #f44336; font-weight: bold;")
        self.download_model_btn.setEnabled(True)
        self.download_progress_bar.setVisible(False)

    def delete_model(self):
        """Delete a selected model"""
        selected_items = self.models_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a model to delete.")
            return

        item = selected_items[0]
        model_data = item.data(Qt.ItemDataRole.UserRole)

        if not model_data:
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {model_data['name']} ({model_data['size_mb']:.1f} MB)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                model_data['path'].unlink()
                self.refresh_models_list()
                QMessageBox.information(self, "Success", f"Model {model_data['name']} deleted successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete model: {str(e)}")


def main():
    # Single instance lock using file locking
    lock_file_path = os.path.join(tempfile.gettempdir(), 'whispersViolins.lock')
    lock_file = None

    try:
        # Try to acquire an exclusive lock
        lock_file = open(lock_file_path, 'w')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        # Another instance is running - exit silently
        print("Another instance is already running. Exiting.")
        if lock_file:
            lock_file.close()
        return

    try:
        # Create application
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)

        # Set application style
        app.setStyle("Fusion")

        # Set organization and application name for proper macOS behavior
        app.setOrganizationName("whispersViolins")
        app.setApplicationName("whispersViolins")

        # Create and show main window
        window = WhispersViolinsApp()
        window.show()
        window.raise_()
        window.activateWindow()

        # Run the application
        exit_code = app.exec()

    finally:
        # Release the lock when app exits
        if lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            try:
                os.remove(lock_file_path)
            except:
                pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
