import os
import sys
import subprocess
from typing import Optional

class MEMOHelper:
    def __init__(self, memo_path: str = "memo"):
        self.memo_path = memo_path
        self.config_path = os.path.join(memo_path, "configs/inference.yaml")
        
    def install_dependencies(self) -> bool:
        """Install required dependencies for MEMO"""
        try:
            # Install core dependencies
            subprocess.run([
                "pip", "install",
                "omegaconf>=2.3.0",
                "pytorch3d",
                "face-alignment",
                "mediapipe",
                "imageio",
                "imageio-ffmpeg"
            ], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            return False

    def setup_memo(self) -> bool:
        """Sets up MEMO if not already installed"""
        if not os.path.exists(self.memo_path):
            try:
                # Clone MEMO repository
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/memoavatar/memo.git",
                    self.memo_path
                ], check=True)
                
                # Install requirements
                subprocess.run([
                    "pip", "install", "-r",
                    os.path.join(self.memo_path, "requirements.txt")
                ], check=True)
                
                # Install additional dependencies that might be missing
                subprocess.run([
                    "pip", "install", "omegaconf", "moviepy", "imageio-ffmpeg"
                ], check=True)
                
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error setting up MEMO: {e}")
                return False
        return True

    def generate_video(
        self,
        input_image: str,
        input_audio: str,
        output_dir: str
    ) -> Optional[str]:
        """Generates video using MEMO"""
        # First make sure the dependencies are installed
        try:
            import omegaconf
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "omegaconf"])
            
        try:
            output_path = os.path.join(output_dir, "memo_output.mp4")
            print("Running MEMO inference...")
            subprocess.run([
                sys.executable,  # Use the current Python interpreter
                os.path.join(self.memo_path, "inference.py"),
                "--config", self.config_path,
                "--input_image", input_image,
                "--input_audio", input_audio,
                "--output_dir", output_dir
            ], check=True)
            
            return output_path if os.path.exists(output_path) else None
        except subprocess.CalledProcessError as e:
            print(f"Error generating video with MEMO: {e}")
            return None
