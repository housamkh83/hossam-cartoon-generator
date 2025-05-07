import os
import subprocess
import time
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
        try:
            if not os.path.exists(self.memo_path):
                print("Cloning MEMO repository...")
                subprocess.run([
                    "git", "clone",
                    "https://github.com/memoavatar/memo.git",
                    self.memo_path
                ], check=True)
                
                # Install dependencies first
                print("Installing dependencies...")
                if not self.install_dependencies():
                    return False

                # Wait for files to be available
                time.sleep(2)
                
                # Create configs directory if it doesn't exist
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # Create default config
                with open(self.config_path, "w") as f:
                    f.write("""
model:
  name: memo
  checkpoint: pretrained/memo.pth

data:
  image_size: 256
  fps: 25

inference:
  batch_size: 1
  num_workers: 0
""")
                
                return True
            return True

        except Exception as e:
            print(f"Error during MEMO setup: {e}")
            return False

    def generate_video(
        self,
        input_image: str,
        input_audio: str,
        output_dir: str
    ) -> Optional[str]:
        """Generates video using MEMO"""
        try:
            if not os.path.exists(self.memo_path):
                print("MEMO not installed. Running setup...")
                if not self.setup_memo():
                    return None
                
            output_path = os.path.join(output_dir, "memo_output.mp4")
            
            print("Running MEMO inference...")
            subprocess.run([
                "python",
                os.path.join(self.memo_path, "inference.py"),
                "--config", self.config_path,
                "--input_image", input_image,
                "--input_audio", input_audio,
                "--output_dir", output_dir
            ], check=True)
            
            if os.path.exists(output_path):
                print(f"Video generated successfully at: {output_path}")
                return output_path
            else:
                print("Video file not found after generation")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error generating video with MEMO: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during video generation: {e}")
            return None