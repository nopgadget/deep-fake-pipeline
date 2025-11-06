# Real-Time Deepfake Pipeline

A comprehensive toolkit for creating real-time deepfakes by combining face swapping and voice cloning technologies. This pipeline integrates multiple state-of-the-art tools to enable full audiovisual transformations.

> ⚠️ **IMPORTANT ETHICAL NOTICE**: This software is intended for educational, research, and legitimate creative purposes only. Misuse of deepfake technology can cause serious harm. **We are not responsible for end-user actions.** Misuse of this software may result in legal consequences including criminal charges in many jurisdictions.

---

## 🎯 Pipeline Components & Function

This pipeline combines several powerful tools:

1. **[Deep-Live-Cam](./Deep-Live-Cam/)** - Real-time face swapping with single image input
2. **[Seed-VC](./seed-vc/)** - Zero-shot voice cloning and conversion
3. **[Video-Voice-Surgeon](./video-voice-surgeon/)** - Video processing tool for applying voice conversion

Together, these tools enable you to:
- Swap faces in real-time via webcam
- Clone and convert voices with minimal reference audio
- Process videos with both face and voice transformations
- Create deepfakes for entertainment, animation, and creative content

---
## 🎬 Features

### Real-Time Face Swapping
- **Single Image Input**: Swap faces using just one photo
- **Live Webcam Mode**: Real-time face replacement for video calls, streaming
### Voice Cloning & Conversion
- **Zero-Shot Voice Cloning**: Clone a voice from 1-30 seconds of reference audio
- **Real-Time Voice Conversion**: ~300ms latency for live applications
- **Singing Voice Conversion**: Specialized models for musical content
- **Accent & Emotion Transfer**: Advanced V2 model preserves speaking style
- **Fine-tuning Support**: Train on custom data with minimal requirements

### Video Processing
- **Automated Pipeline**: Combine face swap + voice conversion on videos
- **Singing Mode**: Optimized settings for musical performances
- **Pitch & Speed Control**: Adjust voice characteristics
- **Quality Preservation**: Maintains original video quality

---

## 📦 Installation

> 💡 **Note**: This pipeline has been tested and verified with **Python 3.11** using **Conda** for environment management. While Python 3.10 may work for some components, we recommend Python 3.11 for the best compatibility.

### System Requirements

**Minimum:**
- Python 3.11 (tested and recommended)
- Conda/Miniconda for environment management
- 8GB RAM
- 4GB+ GPU (for real-time processing)
- FFmpeg

### Quick Start

1. **Install Conda (if not already installed):**

2. **Clone this repository:**
```bash
git clone --recurse-submodules https://github.com/nopgadget/deep-fake-pipeline
cd deep-fake-pipeline
```

3. **Install FFmpeg:**
```bash
conda install conda-forge::ffmpeg
```

4. **Install Deep-Live-Cam:**
```bash
cd Deep-Live-Cam
conda create -n deep-live-cam python=3.11
conda activate deep-live-cam

# Install dependencies
pip install -r requirements.txt

# Download required models (place in models/ folder):
# - GFPGANv1.4.pth
# - inswapper_128_fp16.onnx
# From: https://huggingface.co/hacksider/deep-live-cam
```

5. **Install Seed-VC:**
```bash
cd seed-vc
conda create -n seed-vc python=3.11
conda activate seed-vc
pip install -r requirements.txt
# Models auto-download on first run
```

6. **Verify Installation:**
```bash
# Test Deep-Live-Cam
cd Deep-Live-Cam
conda activate deep-live-cam
python run.py

# Test Seed-VC (in a new terminal)
cd seed-vc
conda activate seed-vc
python app.py --enable-v1 --enable-v2
```

### GPU Acceleration (Recommended)

#### NVIDIA CUDA
```bash
# For Deep-Live-Cam
conda activate deep-live-cam
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install onnxruntime-gpu==1.21.0

# Seed-VC will use CUDA automatically if PyTorch CUDA is available
```

See individual tool READMEs for detailed GPU setup instructions and troubleshooting.

### Why Separate Conda Environments?

We use separate conda environments for each tool to:
- **Avoid Dependency Conflicts**: Different tools may require different versions of the same packages
- **Easy Management**: Activate only what you need when you need it
- **Clean Troubleshooting**: Issues are isolated to specific environments
- **Independent Updates**: Update one tool without affecting the other

You can run both tools simultaneously by opening separate terminals and activating different environments in each.

---

## 🚀 Usage

### 1. Real-Time Face Swap (Webcam)

```bash
cd Deep-Live-Cam
conda activate deep-live-cam
python run.py
```

1. Click "Select a face" and choose a source image
2. Click "Live" to start webcam feed
3. Wait 10-30 seconds for preview
4. Use OBS or similar to capture output

**Command line:**
```bash
conda activate deep-live-cam
python run.py --execution-provider cuda --many-faces --live-mirror
```

### 2. Face Swap on Video/Image

```bash
conda activate deep-live-cam
python run.py --source face.jpg --target input_video.mp4 --output output_video.mp4
```

### 3. Real-Time Voice Conversion

```bash
cd seed-vc
conda activate seed-vc
python real-time-gui.py
```

1. Set **Input Device**: Your physical microphone
2. Set **Output Device**: VB-CABLE Input (virtual audio cable)
3. Upload 3-10 second reference audio of target voice
4. Adjust diffusion steps (10 for real-time, 30 for quality)
5. Click "Start Conversion"

The converted audio is piped to VB-CABLE, which then appears as a virtual microphone (VB-CABLE Output) that you can select in any application (Zoom, Discord, OBS, etc.).

**Setup Virtual Audio Cable:**
- Download [VB-CABLE](https://vb-audio.com/Cable/) (Windows/Mac)
- Linux users can use `pulseaudio-module-loopback` or `jack`
- After installation, VB-CABLE appears as an audio device in your system

### 4. Voice Conversion on Audio

```bash
conda activate seed-vc
python inference.py \
  --source original_speech.wav \
  --target target_voice_reference.wav \
  --output converted_speech.wav \
  --diffusion-steps 30 \
  --inference-cfg-rate 0.7
```

### 5. Voice Conversion on Video

```bash
cd video-voice-surgeon
conda activate seed-vc  # Video-Voice-Surgeon uses the Seed-VC environment
python main.py \
  --seedvc-dir ../seed-vc \
  --input video.mp4 \
  --target voice_reference.wav \
  --output converted_video.mp4 \
  --diffusion-steps 30
```

**For singing:**
```bash
  --singing \
  --f0-condition True
```

### 6. Combined Face + Voice Pipeline (Video)

**Step 1: Face Swap**
```bash
cd Deep-Live-Cam
conda activate deep-live-cam
python run.py --source face.jpg --target video.mp4 --output face_swapped.mp4
```

**Step 2: Voice Conversion**
```bash
cd ../video-voice-surgeon
conda activate seed-vc
python main.py \
  --seedvc-dir ../seed-vc \
  --input ../Deep-Live-Cam/face_swapped.mp4 \
  --target voice.wav \
  --output final_deepfake.mp4
```

### 7. Real-Time Combined Face + Voice (Advanced)

For live streaming/video calls with both face and voice conversion:

**Setup Requirements:**
- [OBS Studio](https://obsproject.com/)
- [VB-CABLE](https://vb-audio.com/Cable/) (or similar virtual audio cable)
- Decent GPU (RTX 3060 or better recommended)
- Two separate terminals (one for each tool)

**Step-by-step Setup:**

1. **Start Deep-Live-Cam (Terminal 1):**
```bash
cd Deep-Live-Cam
conda activate dlc
python run.py --execution-provider cuda
```
   - Select your source face image
   - Click "Live" to start webcam feed
   - The preview window will show the face-swapped output

2. **Start Seed-VC Real-Time Conversion (Terminal 2):**
```bash
cd seed-vc
conda activate seedvc
python real-time-gui.py
```
   - Set **Input Device**: Your microphone
   - Set **Output Device**: VB-CABLE Input (or your virtual audio cable)
   - Upload reference audio of target voice
   - Set diffusion steps to 10 for real-time performance
   - Click "Start Conversion"

3. **Configure OBS Studio:**
   - Add **Display Capture** source → Select Deep-Live-Cam preview window
   - Crop to just the preview area (remove window borders)
   - Add **Audio Input Capture** → Select VB-CABLE Output as the source
   - **Important**: Add a **delay filter** to the video source:
     - Right-click video source → Filters → Add "Render Delay"
     - Set delay to match Seed-VC latency (~430ms typically)
     - This synchronizes the face swap with the voice conversion
   - Enable **Virtual Camera** in OBS

4. **Use in Applications:**
   - In Zoom/Discord/etc., select:
     - **Camera**: OBS Virtual Camera
     - **Microphone**: VB-CABLE Output (the virtual audio cable)
   - You now have synchronized face and voice conversion!

**Timing Calibration:**
- Seed-VC algorithmic delay is typically ~300-430ms (depends on settings)
- Adjust OBS render delay to match your actual measured latency
- Test with a video to ensure audio/video sync before going live

**Performance Notes:**
- Both tools running simultaneously is GPU-intensive
- Monitor GPU usage and temperature
- Close unnecessary applications
- Consider lowering resolution or diffusion steps if lagging

**Signal Flow Diagram:**
```
Real-Time Deepfake Pipeline

VIDEO PATH:
Your Webcam → Deep-Live-Cam (face swap) → Preview Window → OBS Display Capture
                                                              (with ~430ms delay)
                                                                      ↓
                                                              OBS Virtual Camera
                                                                      ↓
                                                           Zoom/Discord/App

AUDIO PATH:
Your Microphone → Seed-VC (voice conversion) → VB-CABLE Input → VB-CABLE Output
                     (~430ms latency)                              (virtual mic)
                                                                      ↓
                                                           Zoom/Discord/App
                                                                      
Note: OBS video delay matches Seed-VC audio delay for perfect sync!
```

---

## 🎛️ Performance Tips

### Real-Time Face Swapping
- **GPU Essential**: CPU mode too slow for real-time
- **Resolution**: Lower webcam resolution for better FPS
### Real-Time Voice Conversion
- **Model Choice**: Use `seed-uvit-xlsr-tiny` for lowest latency
- **Diffusion Steps**: 4-10 for real-time, 30+ for quality
- **Block Time**: Must be > inference time per chunk
- **Background Tasks**: Close GPU-intensive apps

### Video Processing
- **Batch Processing**: Process videos separately, combine later
- **Quality Settings**: 
  - Fast: `--diffusion-steps 15 --fp16 True`
  - Balanced: `--diffusion-steps 30`
  - High Quality: `--diffusion-steps 50 --f0-condition True`

---

## 🔧 Troubleshooting

### Common Issues

**Conda Environment Management**
```bash
conda info --envs
conda activate deep-live-cam  # or seed-vc
conda deactivate

# Remove an environment if you need to reinstall
conda env remove -n deep-live-cam
conda env remove -n seed-vc
```

**Deep-Live-Cam Won't Start**
- Ensure correct conda environment is activated: `conda activate deep-live-cam`
- Ensure models are in `models/` folder
- Install Visual Studio C++ Redistributables (Windows)
- Verify Python version: `python --version` (should be 3.11.x)

**CUDA Out of Memory**
- Use `--fp16 True`
- Reduce batch size
- Close other GPU applications
- Lower diffusion steps

**Real-Time Conversion Stuttering**
- Reduce diffusion steps to 4-6
- Decrease max prompt length
- Increase block time
- Ensure inference time < block time

---

## 🤝 Credits & Acknowledgments

### Deep-Live-Cam
- [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)
- [deepinsight/insightface](https://github.com/deepinsight/insightface)
- [s0md3v/roop](https://github.com/s0md3v/roop) (original base)

### Seed-VC
- [Plachta/Seed-VC](https://github.com/Plachta/Seed-VC)
- [Amphion](https://github.com/open-mmlab/Amphion) framework
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

### Technologies
- FFmpeg for media processing
- PyTorch & ONNX for neural networks
- Gradio for web interfaces

---

## 📜 License

This pipeline combines multiple open-source projects:

- **Deep-Live-Cam**: See [Deep-Live-Cam/LICENSE](./Deep-Live-Cam/LICENSE)
- **Seed-VC**: See [seed-vc/LICENSE](./seed-vc/LICENSE)
- **InsightFace models**: Non-commercial research purposes only

**Important**: The `inswapper` model used by Deep-Live-Cam is for non-commercial research purposes only.


