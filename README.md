# 🎻 whisperViolins

A beautiful macOS audio transcription application powered by OpenAI's Whisper AI and built with PyQt6 in Antigravity.

## Features

- 🎯 **Multi-Language Support**: Transcribe audio in 13+ languages including English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Korean, Arabic, and Hindi
- 🤖 **Multiple AI Models**: Choose from 5 Whisper models (tiny, base, small, medium, large) to balance speed and accuracy
- 🎨 **Modern GUI**: Clean, intuitive interface with large, readable fonts
- 📦 **Model Management**: Download, view, and delete Whisper models directly from the app
- 💾 **Flexible Export**: Copy transcriptions to clipboard or save as text files
- ⚡ **Async Processing**: Non-blocking transcription with progress feedback
- 🎵 **Wide Format Support**: MP3, WAV, M4A, FLAC, OGG, AAC

## Installation

### Prerequisites
- macOS
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone then navigate to the project directory:
```bash
cd /path/to/whisperViolins
```

2. Install dependencies (automatically handled by uv):
```bash
uv sync
```

## Running the Application

### From Source
```bash
uv run python main.py
```

## Building an Executable App

To create a standalone macOS application bundle that you can distribute or move to your Applications folder:

### Prerequisites: Install ffmpeg

Whisper requires ffmpeg to decode audio files. Install it first:

```bash
# Using Homebrew
brew install ffmpeg
```

### Step 1: Install PyInstaller
```bash
uv add pyinstaller
```

### Step 2: Build the App Bundle

The build needs to include ffmpeg and Whisper's data files. Use this command:

```bash
uv run pyinstaller --name="whisperViolins" \
  --windowed \
  --noconfirm \
  --onefile \
  --icon=icon.icns \
  --add-data="icon.png:." \
  --add-binary="/opt/homebrew/bin/ffmpeg:." \
  --add-binary="/opt/homebrew/bin/ffprobe:." \
  --collect-data whisper \
  --osx-bundle-identifier=com.whispersviolins.app \
  main.py
```

**What this does:**
- `--collect-data whisper`: Bundles Whisper's asset files (mel_filters.npz, etc.)
- `--add-binary`: Includes ffmpeg and ffprobe for audio processing
- `--add-data`: Includes the app icon

**Note**: If your ffmpeg is installed elsewhere (e.g., via MacPorts or from source), adjust the paths accordingly. You can find the location with `which ffmpeg` and `which ffprobe`.


### Step 3: Locate Your App
The application will be created in the `dist/` folder:
```bash
dist/whispersViolins.app
```

### Step 4: Move to Applications (Optional)
```bash
cp -r dist/whispersViolins.app /Applications/
```

**Note**: The first time you build, PyInstaller will bundle all dependencies including PyTorch and Whisper. This may take 5-10 minutes and the resulting `.app` bundle will be ~2-3GB. The build is large because it includes:
- PyTorch ML framework (~1GB)
- Whisper models cache support
- ffmpeg audio processing (~100MB)
- All Python dependencies

**Tip**: To reduce size, use `--onefile` mode and only download the models you need within the app.


## Usage

### First Time Setup
1. Launch the application
2. Navigate to the "Model Management" tab
3. Select a model (start with "base" for testing)
4. Click "Download" and wait for the model to download

### Transcribing Audio
1. Switch to the "Transcription" tab
2. Select your desired model from the dropdown
3. Choose the language (or leave as "Auto-detect")
4. Click "Select Audio File" and choose your audio file
5. Click "Start Transcription"
6. Wait for the transcription to complete
7. Copy or save the results

### Model Information
- **tiny** (~39 MB): Fastest, lowest accuracy
- **base** (~74 MB): Good balance for testing
- **small** (~244 MB): Balanced speed and accuracy
- **medium** (~769 MB): High accuracy, slower
- **large** (~1550 MB): Best accuracy, slowest

## Project Structure

```
whispersViolins/
├── main.py              # Main application code
├── icon.png             # Application icon
├── pyproject.toml       # Project dependencies
├── README.md            # This file
└── .python-version      # Python version specification
```

## Dependencies

Key dependencies (automatically managed by uv):
- **PyQt6**: GUI framework
- **openai-whisper**: AI transcription model
- **torch**: PyTorch for ML operations
- **pyinstaller**: For building executables (optional)

## Troubleshooting

### App won't open after building
If macOS blocks the app due to security settings:
1. Right-click the app in Finder
2. Select "Open"
3. Click "Open" in the security dialog

### Models not downloading
- Check your internet connection
- Ensure you have sufficient disk space
- Models are stored in `~/.cache/whisper`

### Transcription errors
- Ensure the selected model is downloaded
- Check that your audio file format is supported
- Try a smaller model first (base or tiny)

## License

This project uses OpenAI's Whisper model. Please refer to the [Whisper repository](https://github.com/openai/whisper) for licensing information.

## Credits

- **Antigravity (google)**: [https://antigravity.google/](https://antigravity.google/)
- **OpenAI Whisper**: [github.com/openai/whisper](https://github.com/openai/whisper)
- **PyQt6**: [riverbankcomputing.com/software/pyqt](https://www.riverbankcomputing.com/software/pyqt/)
