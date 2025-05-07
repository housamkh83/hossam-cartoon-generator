@echo off
setlocal EnableDelayedExpansion

echo ===================================
echo Cartoon Generator Setup (Arabic Support)
echo ===================================

REM Check Python version
echo [1/8] Checking Python installation...
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Setup virtual environment
echo [2/8] Setting up virtual environment...
if exist venv (
    echo Cleaning existing environment...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

REM Activate and upgrade pip
echo [3/8] Activating environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

echo [4/8] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo [5/8] Installing PyTorch with CUDA support...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [WARNING] CUDA installation might have failed. Continuing with CPU version...
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
)

REM Install core dependencies with specific versions to avoid conflicts
echo [6/8] Installing core dependencies...
pip install --no-cache-dir ^
    diffusers==0.21.4 ^
    transformers==4.27.4 ^
    accelerate==0.25.0 ^
    huggingface-hub==0.19.4 ^
    numpy==1.24.0 ^
    Pillow==10.0.0 ^
    gradio==3.50.2 ^
    opencv-python==4.8.0 ^
    controlnet_aux==0.0.6 ^
    mediapipe==0.10.0

REM Install Arabic TTS and additional requirements
echo [7/8] Installing Arabic TTS support...
pip install --no-cache-dir ^
    gTTS==2.3.2 ^
    requests==2.31.0 ^
    GitPython==3.1.0 ^
    PyYAML==6.0.1 ^
    omegaconf==2.3.0 ^
    face-alignment==1.3.1 ^
    basicsr==1.4.2 ^
    imageio==2.31.1 ^
    imageio-ffmpeg==0.4.8 ^
    librosa==0.10.1 ^
    moviepy==1.0.3

REM Install remaining requirements from file if needed
echo [8/8] Checking for additional requirements...
if exist requirements.txt (
    echo Installing additional requirements from requirements.txt...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found, skipping.
)

REM Setup animation tools
echo Setting up animation tools...

REM Setup DAWN
if not exist DAWN (
    echo Setting up DAWN...
    git clone https://github.com/dawnauracle/DAWN.git
    cd DAWN
    pip install -r requirements.txt
    cd ..
)

REM Setup MEMO
if not exist memo (
    echo Setting up MEMO...
    git clone https://github.com/memoavatar/memo.git
    cd memo
    pip install -r requirements.txt
    cd ..
)

echo Animation tools setup complete!

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    echo HF_TOKEN=your_huggingface_token_here > .env
    echo D_ID_API_KEY=your_d_id_api_key_here >> .env
    echo HEYGEN_API_KEY=your_heygen_api_key_here >> .env
    echo Please update the .env file with your API keys
)

REM Check for Arabic text-to-speech support
echo Testing Arabic text-to-speech support...
python -c "from gtts import gTTS; gTTS(text='مرحبا بالعالم', lang='ar').save('test_arabic.mp3'); print('Arabic TTS test successful!')" 2>nul
if errorlevel 1 (
    echo [WARNING] Arabic text-to-speech test failed. Please check your internet connection.
) else (
    echo Arabic text-to-speech is working!
    del test_arabic.mp3
)

REM Setup complete
echo.
echo ===================================
echo Setup Complete! (مكتمل)
echo -----------------------------------
echo To start the application:
echo 1. Make sure you're in the project directory
echo 2. Run: python app.py
echo ===================================
echo.

pause

@echo off
echo ===================================
echo Fixing Dependencies
echo ===================================

call venv\Scripts\activate

echo Installing correct OpenCV version...
pip uninstall -y opencv-python
pip install opencv-python==4.8.0.76

echo Installing MEMO dependencies...
pip install omegaconf face-alignment basicsr librosa

echo Installing MoviePy for video generation...
pip install moviepy imageio-ffmpeg

echo Dependencies fixed successfully!
pause

@echo off
echo ===================================
echo Instalando dependencias
echo ===================================

call venv\Scripts\activate

echo Instalando gradio...
pip install gradio==3.50.2

echo Instalando bibliotecas principales...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate huggingface-hub
pip install Pillow numpy opencv-python

echo Instalando bibliotecas adicionales...
pip install moviepy imageio imageio-ffmpeg
pip install controlnet_aux 
pip install gTTS
pip install requests GitPython PyYAML
pip install omegaconf

echo Instalando face-enhancement (si es posible)...
pip install gfpgan facexlib realesrgan

echo ===================================
echo Instalación completada
echo ===================================
pause

@echo off
echo ===================================
echo Fixing Library Versions
echo ===================================

call venv\Scripts\activate

echo Uninstalling incompatible packages...
pip uninstall -y diffusers transformers huggingface_hub

echo Installing compatible versions...
pip install huggingface_hub==0.14.1
pip install transformers==4.27.4
pip install diffusers==0.21.4

echo Fixing missing code in app.py...
powershell -Command "(Get-Content app.py) -replace 'if __name == \"__main__\":', 'if __name__ == \"__main__\":' | Set-Content app.py"

echo Removing problematic code at end of file...
powershell -Command "$content = Get-Content app.py -Raw; $newContent = $content -replace '# save as fix_code.*?backticks\)', ''; Set-Content -Path app.py -Value $newContent"

echo ===================================
echo Dependencies fixed successfully!
echo ===================================
pause