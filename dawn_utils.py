import os
import subprocess
from typing import Optional
from pathlib import Path

class DAWNHelper:
    def __init__(self, dawn_path: str = "DAWN"):
        self.dawn_path = Path(dawn_path)
        self.config_path = self.dawn_path / "configs/inference.yaml"
        
    def setup_dawn(self) -> bool:
        """Sets up DAWN if not already installed"""
        if not self.dawn_path.exists():
            try:
                # Clone DAWN repository
                subprocess.run([
                    "git", "clone",
                    "https://github.com/dawn-diffusion/DAWN.git",
                    str(self.dawn_path)
                ], check=True)
                
                # Install requirements
                subprocess.run([
                    "pip", "install", "-r",
                    str(self.dawn_path / "requirements.txt")
                ], check=True)
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error setting up DAWN: {e}")
                return False
        return True

    def generate_talking_video(
        self,
        input_image: str,
        input_audio: str,
        output_dir: str,
        expression: str = "neutral",
        language: str = "ar"  # Default to Arabic
    ) -> Optional[str]:
        """Generates expressive talking video using DAWN"""
        try:
            output_path = os.path.join(output_dir, "dawn_output.mp4")
            
            subprocess.run([
                "python",
                str(self.dawn_path / "inference.py"),
                "--config", str(self.config_path),
                "--source_image", input_image,
                "--driving_audio", input_audio,
                "--expression", expression,
                "--language", language,
                "--output_path", output_path
            ], check=True)
            
            return output_path if os.path.exists(output_path) else None
            
        except subprocess.CalledProcessError as e:
            print(f"Error generating video with DAWN: {e}")
            return None