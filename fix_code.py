# --------------------------------------------------
# المرحلة 1: إعداد بيئة العمل واستيراد المكتبات
# --------------------------------------------------
import os
import shutil
import zipfile
from datetime import datetime
import gradio as gr
from PIL import Image, ImageDraw  # Add ImageDraw
import cv2 # OpenCV
import numpy as np
import requests # For APIs
import torch # Assuming PyTorch backend for local models
import subprocess
from typing import Optional
import sys  # Add this to imports section

# --- AI Model Libraries (Choose based on approach) ---
# --- AI Model Libraries ---
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
# حذف السطر التالي لتجنب التحذير:
# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import StableDiffusionControlNetPipeline
from transformers import pipeline # Potentially for other tasks

# Option 2: TTS Libraries (Local)
from TTS.api import TTS # Example: XTTS
# import soundfile as sf

# Option 3: Lip Sync Libraries (Local)
# Requires separate installation and potentially complex setup (e.g., Wav2Lip, SadTalker)
# Need helper functions/scripts to call these tools

# Add to imports section
from memo_utils import MEMOHelper
from dawn_utils import DAWNHelper

# Add this function near the beginning of your file, after the imports

def create_image_grid(images, rows, cols):
    """Create a grid of images"""
    if not images:
        return None
    
    # Make sure all images are PIL images
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            pil_images.append(Image.fromarray(img))
        else:
            pil_images.append(img)
    
    # Get dimensions
    width, height = pil_images[0].size
    grid_width = width * cols
    grid_height = height * rows
    
    # Create a blank grid
    grid_img = Image.new('RGB', (grid_width, grid_height))
    
    # Paste images into grid
    for i, img in enumerate(pil_images):
        if i >= rows * cols:
            break  # Don't exceed grid size
        
        row = i // cols
        col = i % cols
        grid_img.paste(img, (col * width, row * height))
    
    return grid_img

def get_device_settings():
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        slow_mode = vram < 8  # Activate slow mode for GPUs with less than 8GB
        device = "cuda"
    else:
        slow_mode = True
        device = "cpu"
    return device, slow_mode

# --- تحسين VAE للوجوه ---
def get_improved_vae():
    """تهيئة VAE محسّن لتفاصيل الوجه"""
    try:
        from diffusers import AutoencoderKL
        improved_vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
        )
        if torch.__version__ >= "2.0.0":
            improved_vae = torch.compile(improved_vae)
        return improved_vae
    except Exception as e:
        print(f"فشل تحميل VAE المحسن: {e}")
        return None

# --- تحسين الوجه باستخدام GFPGAN ---
gfpgan_enhancer = None

def initialize_gfpgan():
    global gfpgan_enhancer
    
    try:
        # First try to install gfpgan if not already installed
        try:
            import gfpgan
            print("GFPGAN module already installed")
        except ImportError:
            print("GFPGAN module not found, skipping face enhancement")
            return None
        
        # Now try to initialize the face enhancer
        from gfpgan import GFPGANer
        gfpgan_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=1,  # لا نريد تكبير الصورة، فقط تحسين الوجه
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None  # لا نحتاج لمحسن خلفية
        )
        print("تم تهيئة GFPGAN بنجاح.")
    except Exception as e:
        print(f"فشل تهيئة GFPGAN: {e}")
        gfpgan_enhancer = None

def enhance_face_with_gfpgan(pil_image):
    global gfpgan_enhancer
    
    if gfpgan_enhancer is None:
        initialize_gfpgan()
    
    if gfpgan_enhancer is None:  # إذا فشل التهيئة
        print("GFPGAN not available, returning original image")
        return pil_image

    try:
        img_np = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        _, _, restored_img = gfpgan_enhancer.enhance(img_np, has_aligned=False, only_center_face=False, paste_to_face=True)
        if restored_img is not None:
            return Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"خطأ أثناء تحسين الوجه بـ GFPGAN: {e}")
    
    return pil_image  # إرجاع الصورة الأصلية في حالة الخطأ

# الأوامر النصية المحسّنة للوجوه الكرتونية
POSITIVE_FACE_PROMPT = """3D Pixar style cartoon character with highly detailed facial features,
perfect face, symmetrical face, detailed eyes, clear eyes, expressive eyes, well-defined lips and nose,
appealing eyes, beautiful face, flawless skin texture, professional 3D rendering, high quality,
detailed animation style, cinematic lighting, ultra realistic textures, volumetric lighting"""

NEGATIVE_FACE_PROMPT = """deformed face, distorted face, disfigured, bad eyes, crossed eyes, 
bad anatomy, ugly, blurry, low quality, text, watermark, logo, malformed face, asymmetrical eyes, 
blurry eyes, missing pupils, extra eyes, poorly drawn face, distorted lips, wonky eyes, 
extra limbs, duplicate, multiple bodies, extra fingers, mutated hands"""

# --- Configuration ---
# API Keys (Load from environment variables or config file for security)
D_ID_API_KEY = os.getenv("D_ID_API_KEY")
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
# Other API keys as needed...

# Model Paths (if using local models)
SD_BASE_MODEL = "runwayml/stable-diffusion-v1-5" # Or a fine-tuned cartoon model
SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting" # Or equivalent
# Add paths for ControlNet, TTS, Lip Sync models if used locally

# Add to Configuration section
MEMO_HELPER = MEMOHelper()
DAWN_HELPER = DAWNHelper()

# --- Global Variables / State (for Gradio) ---
current_cartoon_image = None
generated_expressions = {} # Store expression images { 'happy': Image, ... }
generated_poses = [] # Store pose images [Image, Image, ...]
edited_image = None
final_video_path = None
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
all_generated_files = [] # Keep track of all saved files for zipping

# --------------------------------------------------
# المرحلة 2: تحويل الصورة إلى كرتونية 3D
# --------------------------------------------------
# تحديث دالة cartoonify_image لتحسين جودة الوجه

def cartoonify_image(input_image, api_choice="Local Stable Diffusion", progress=gr.Progress()):
    progress(0, desc="Initializing")
    global current_cartoon_image, all_generated_files
    print(f"Cartoonifying using: {api_choice}")

    try:
        # Convert input to PIL Image
        pil_image = Image.fromarray(input_image).convert("RGB")
        pil_image = pil_image.resize((512, 512))

        if api_choice == "Local Stable Diffusion":
            if not torch.cuda.is_available():
                return None, "Error: CUDA GPU required for local Stable Diffusion."

            try:
                # تحميل VAE المحسّن
                improved_vae = get_improved_vae()
                
                # Initialize ControlNet with better settings
                use_controlnet = True
                try:
                    # Only use Canny and Depth models (skip face control due to DLL issues)
                    controlnet_canny = ControlNetModel.from_pretrained(
                        "lllyasviel/sd-controlnet-canny",
                        torch_dtype=torch.float16
                    )
                    controlnet_depth = ControlNetModel.from_pretrained(
                        "lllyasviel/sd-controlnet-depth",
                        torch_dtype=torch.float16
                    )
                    # Use only two controlnets to avoid mismatch error
                    controlnets = [controlnet_canny, controlnet_depth]
                    
                except Exception as e:
                    print(f"ControlNet not available: {e}")
                    use_controlnet = False

                if use_controlnet:
                    try:
                        # Initialize pipeline with multiple ControlNets and improved VAE
                        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                            SD_BASE_MODEL,
                            controlnet=controlnets,
                            vae=improved_vae,  # استخدام VAE المحسن
                            torch_dtype=torch.float16
                        ).to("cuda")

                        # Create control images
                        image_np = np.array(pil_image)
                        
                        # Canny edge map
                        canny_map = cv2.Canny(image_np, 100, 200)
                        canny_map = np.stack([canny_map] * 3, axis=-1)
                        canny_image = Image.fromarray(canny_map)
                        
                        # Depth map using Midas
                        from controlnet_aux import MidasDetector
                        midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")
                        depth_map = midas(image_np)
                        depth_image = Image.fromarray(depth_map)
                        
                        # Just use the two control images we have
                        control_images = [canny_image, depth_image]
                        control_weights = [0.8, 0.6]  # زيادة وزن Canny للحفاظ على ملامح الوجه
                        
                        # Skip face landmarks due to DLL issues
                        # Just use simple face feature extraction for the prompt
                        face_features = "detailed face, high quality facial features"
                        try:
                            # Basic face analysis without MediaPipe
                            face_features = "accurate facial proportions, realistic eye details, natural smile, well-defined lips"
                        except Exception as e:
                            print(f"Using default face features: {e}")
                            
                        # استخدام الأوامر النصية المحسنة للوجه
                        prompt = f"{POSITIVE_FACE_PROMPT}, {face_features}"
                        
                        # Generate with multiple ControlNets
                        output = pipe(
                            prompt=prompt,
                            negative_prompt=NEGATIVE_FACE_PROMPT,
                            image=pil_image,
                            control_image=control_images,  # Now matches the number of controlnets
                            controlnet_conditioning_scale=control_weights,
                            strength=0.58,  # تقليل قليلا للحفاظ على الهوية أكثر
                            guidance_scale=8.5,  # Higher for better prompt following
                            num_inference_steps=45  # زيادة عدد الخطوات لجودة أفضل
                        )

                        output_image = output.images[0]
                        
                        # تحسين الوجه باستخدام GFPGAN - skip if not available
                        try:
                            if 'gfpgan_enhancer' in globals() and gfpgan_enhancer is not None:
                                output_image = enhance_face_with_gfpgan(output_image)
                        except Exception as e:
                            print(f"Face enhancement failed, continuing with original: {e}")
                        
                        # Save result
                        if output_image:
                            current_cartoon_image = output_image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = os.path.join(output_dir, f"cartoon_{timestamp}.png")
                            output_image.save(save_path)
                            all_generated_files.append(save_path)
                            return output_image, "Cartoon generated successfully!"

                    finally:
                        if 'pipe' in locals():
                            del pipe
                        if 'controlnets' in locals():
                            del controlnets
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error during cartoonification: {e}")
                return None, f"Error: {e}"

    except Exception as e:
        print(f"Error in image processing: {e}")
        return None, f"Error: {e}"

    return None, "Failed to generate cartoon."

def analyze_face(image_np):
    """
    تحليل مفصل للوجه لاستخراج الخصائص - نسخة بسيطة لا تعتمد على MediaPipe
    """
    try:
        # Simple face analysis without MediaPipe
        # Basic detection using OpenCV's Haar Cascade
        # This is less accurate but more stable than MediaPipe
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            return "accurate facial proportions, realistic eye details, natural smile, well-defined lips, expressive face"
        
        return "person with natural facial features and expression"
    except Exception as e:
        print(f"Simple face analysis error: {e}")
        return "person with detailed face"

def analyze_image(image_np):
    """
    تحليل بسيط للصورة لاستخراج الخصائص الأساسية
    """
    try:
        # يمكن استخدام face_recognition أو dlib هنا
        # للتبسيط، نعيد وصفاً عاماً
        return "person with natural features and expression"
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return "person"

# --------------------------------------------------
# المرحلة 3: إنشاء تعبيرات وجه مختلفة
# --------------------------------------------------
# تحديث دالة generate_expressions_batch لتحسين جودة الوجه

def generate_expressions_batch(base_cartoon_img, api_choice="Local Stable Diffusion"):
    global generated_expressions, all_generated_files
    
    try:
        if base_cartoon_img is None:
            return None, "Error: No input image provided. Please generate a cartoon image first."

        # Convert input to PIL Image
        try:
            if isinstance(base_cartoon_img, np.ndarray):
                base_pil_image = Image.fromarray(base_cartoon_img).convert("RGB")
            elif isinstance(base_cartoon_img, Image.Image):
                base_pil_image = base_cartoon_img.convert("RGB")
            else:
                return None, f"Error: Unexpected image type: {type(base_cartoon_img)}"
            
            base_pil_image = base_pil_image.resize((512, 512))
            print("Image prepared for processing")
            
        except Exception as e:
            print(f"Error during image conversion: {e}")
            return None, f"Failed to process input image: {str(e)}"

        if api_choice == "Local Stable Diffusion":
            if not torch.cuda.is_available():
                return None, "Error: CUDA GPU required for local Stable Diffusion."
            
            try:
                torch.cuda.empty_cache()
                print("CUDA cache cleared")

                # تحميل VAE المحسّن
                improved_vae = get_improved_vae()

                # First initialize base pipeline to get scheduler config
                base_pipe = StableDiffusionPipeline.from_pretrained(
                    SD_BASE_MODEL,
                    torch_dtype=torch.float16,
                    vae=improved_vae  # استخدام VAE المحسن
                )
                scheduler = UniPCMultistepScheduler.from_config(base_pipe.scheduler.config)
                del base_pipe  # Clean up memory

                # Initialize ControlNet with scheduler
                controlnet_canny = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny",
                    torch_dtype=torch.float16
                ).to("cuda")
                
                # إضافة ControlNet للعمق للحفاظ على بنية الوجه ثلاثية الأبعاد
                try:
                    controlnet_depth = ControlNetModel.from_pretrained(
                        "lllyasviel/sd-controlnet-depth",
                        torch_dtype=torch.float16
                    ).to("cuda")
                    controlnets = [controlnet_canny, controlnet_depth]
                    use_depth = True
                except Exception as e:
                    print(f"Depth ControlNet not available: {e}")
                    controlnets = [controlnet_canny]
                    use_depth = False

                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    SD_BASE_MODEL,
                    controlnet=controlnets,
                    torch_dtype=torch.float16,
                    scheduler=scheduler,
                    vae=improved_vae  # استخدام VAE المحسن
                ).to("cuda")

                expressions = ["happy", "sad", "angry", "surprised", "neutral", "thinking"]
                results = {}

                # Create Canny edge map once for consistency
                image_np = np.array(base_pil_image)
                canny_map = cv2.Canny(image_np, 100, 200)
                canny_map = np.stack([canny_map] * 3, axis=-1)
                control_image_canny = Image.fromarray(canny_map)
                
                # إنشاء خريطة العمق إذا تم تحميل ControlNet الخاص بها
                if use_depth:
                    try:
                        from controlnet_aux import MidasDetector
                        midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")
                        depth_map = midas(image_np)
                        control_image_depth = Image.fromarray(depth_map)
                    except Exception as e:
                        print(f"Failed to create depth map: {e}")
                        use_depth = False

                for expr in expressions:
                    print(f"Generating {expr} expression...")
                    try:
                        # تحسين الأوامر النصية لكل تعبير
                        prompt = f"{POSITIVE_FACE_PROMPT}, {expr} expression, emotional, expressive {expr} face"
                        
                        # إعداد صور التحكم وأوزانها
                        if use_depth:
                            control_images = [control_image_canny, control_image_depth]
                            control_weights = [0.7, 0.5]  # وزن أعلى للـ Canny للحفاظ على تفاصيل الوجه
                        else:
                            control_images = control_image_canny
                            control_weights = 0.7
                        
                        output = pipe(
                            prompt=prompt,
                            negative_prompt=NEGATIVE_FACE_PROMPT,
                            image=base_pil_image,
                            control_image=control_images,
                            strength=0.45,  # قيمة جيدة للحفاظ على الهوية
                            controlnet_conditioning_scale=control_weights,
                            guidance_scale=8.0,  # زيادة التوجيه للحصول على تعبيرات أفضل
                            num_inference_steps=40  # زيادة عدد الخطوات للحصول على جودة أفضل
                        )
                        
                        expr_image = output.images[0]
                        
                        # تحسين الوجه باستخدام GFPGAN
                        try:
                            expr_image = enhance_face_with_gfpgan(expr_image)
                        except Exception as e:
                            print(f"Face enhancement failed for {expr} expression: {e}")
                        
                        results[expr] = expr_image

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(output_dir, f"expr_{expr}_{timestamp}.png")
                        expr_image.save(save_path)
                        all_generated_files.append(save_path)
                        print(f"Saved {expr} expression")

                    except Exception as e:
                        print(f"Error generating {expr} expression: {e}")
                        continue

            except Exception as e:
                print(f"Pipeline initialization error: {e}")
                return None, f"Failed to initialize Stable Diffusion: {str(e)}"
            
            finally:
                if 'pipe' in locals():
                    del pipe
                if 'controlnets' in locals():
                    del controlnets
                torch.cuda.empty_cache()
                print("Cleaned up GPU resources")

            # Create and save grid if we have results
            if results:
                try:
                    grid_img = create_image_grid(list(results.values()), rows=2, cols=3)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    grid_save_path = os.path.join(output_dir, f"expressions_grid_{timestamp}.png")
                    grid_img.save(grid_save_path)
                    all_generated_files.append(grid_save_path)
                    generated_expressions = results
                    return grid_img, f"Successfully generated {len(results)} expressions"
                except Exception as e:
                    print(f"Error creating grid: {e}")
                    return None, f"Generated expressions but failed to create grid: {str(e)}"

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, f"Unexpected error: {str(e)}"

    return None, "Failed to generate expressions"

# --------------------------------------------------
# المرحلة 4: توليد وضعيات للجسم
# --------------------------------------------------
def generate_body_poses(base_cartoon_img, api_choice="Local Stable Diffusion"):
    global generated_poses, all_generated_files
    
    try:
        # Input validation and conversion
        if base_cartoon_img is None:
            return None, "Error: Please generate the base cartoon image first (Stage 2)."

        # Convert input to PIL Image
        if isinstance(base_cartoon_img, np.ndarray):
            base_pil_image = Image.fromarray(base_cartoon_img).convert("RGB")
        elif isinstance(base_cartoon_img, Image.Image):
            base_pil_image = base_cartoon_img.convert("RGB")
        else:
            return None, f"Error: Invalid image type: {type(base_cartoon_img)}"

        # Ensure proper image size for full body
        base_pil_image = base_pil_image.resize((512, 768))  # Larger height for full body
        
        # Create smaller version for face analysis
        image_np_for_face_analysis = np.array(base_pil_image.resize((512, 512)))
        face_description_from_input = analyze_face(image_np_for_face_analysis)

        if api_choice == "Local Stable Diffusion":
            if not torch.cuda.is_available():
                return None, "Error: CUDA GPU required for local Stable Diffusion."

            try:
                torch.cuda.empty_cache()
                print("CUDA cache cleared")

                # تحميل VAE المحسّن
                improved_vae = get_improved_vae()

                # Initialize ControlNets
                controlnet_pose = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose",
                    torch_dtype=torch.float16
                ).to("cuda")

                # Initialize Canny ControlNet for additional structure preservation
                controlnet_canny = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny",
                    torch_dtype=torch.float16
                ).to("cuda")

                # Get base pipeline config
                base_pipe = StableDiffusionPipeline.from_pretrained(
                    SD_BASE_MODEL,
                    torch_dtype=torch.float16,
                    vae=improved_vae
                )
                scheduler = UniPCMultistepScheduler.from_config(base_pipe.scheduler.config)
                del base_pipe

                # Initialize pipeline with both ControlNets
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    SD_BASE_MODEL,
                    controlnet=[controlnet_pose, controlnet_canny],
                    scheduler=scheduler,
                    vae=improved_vae,
                    torch_dtype=torch.float16
                ).to("cuda")

                # Define poses with reference points and better prompts
                poses = [
                    {
                        "name": "standing",
                        "reference_pose": create_standing_pose()
                    },
                    {
                        "name": "sitting",
                        "reference_pose": create_sitting_pose()
                    },
                    {
                        "name": "walking",
                        "reference_pose": create_walking_pose()
                    }
                ]

                results = []
                
                # Create Canny map once for consistency
                image_np = np.array(base_pil_image)
                canny_map = cv2.Canny(image_np, 100, 200)
                canny_map = np.stack([canny_map] * 3, axis=-1)
                canny_image = Image.fromarray(canny_map)

                for pose_info in poses:
                    print(f"Generating {pose_info['name']} pose...")
                    
                    # تحسين الأوامر النصية للحفاظ على جودة الوجه
                    prompt = f"{POSITIVE_FACE_PROMPT.replace('cartoon character', 'full body cartoon character')}, {face_description_from_input}, {pose_info['name']} pose, maintaining original facial identity and high detail, full body shot, perfect body proportions, professional 3D rendering"
                    
                    # دمج الأوامر السلبية للوجه والجسم
                    negative_prompt = (
                        f"{NEGATIVE_FACE_PROMPT}, bad anatomy body, bad proportions body, extra limbs, cloned face on body, "
                        f"duplicate bodies, multiple bodies, extra fingers, mutated hands, "
                        f"poorly drawn hands, poorly drawn feet, mutation, mangled, ugly body, "
                        f"text, watermark, logo, signature."
                    )
                    
                    pose_image = create_pose_image(pose_info["reference_pose"], size=(512, 768))
                    
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=base_pil_image, # الصورة الكرتونية الأساسية كمدخل
                        control_image=[pose_image, canny_image],
                        controlnet_conditioning_scale=[0.9, 0.4],  # OpenPose أقوى، Canny للمحافظة على البنية
                        strength=0.7, # تقليل طفيف للسماح بتأثير أكبر للصورة الأصلية
                        guidance_scale=8.5, # توجيه جيد
                        num_inference_steps=45 # خطوات أكثر للجودة
                    )

                    pose_result_image = output.images[0]
                    
                    # --- تطبيق تحسين الوجه هنا ---
                    try:
                        print(f"Enhancing face for {pose_info['name']} pose...")
                        pose_result_image = enhance_face_with_gfpgan(pose_result_image)
                    except Exception as e:
                        print(f"Face enhancement failed for {pose_info['name']} pose: {e}")
                    
                    results.append(pose_result_image)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(output_dir, f"pose_{pose_info['name']}_{timestamp}.png")
                    pose_result_image.save(save_path)
                    all_generated_files.append(save_path)
                    print(f"Saved {pose_info['name']} pose to {save_path}")

                # Cleanup
                if 'pipe' in locals(): del pipe
                if 'controlnet_pose' in locals(): del controlnet_pose
                if 'controlnet_canny' in locals(): del controlnet_canny
                torch.cuda.empty_cache()
                print("Cleaned up GPU resources after body pose generation.")

                if results:
                    grid_img = create_image_grid(results, rows=1, cols=len(results)) # Make cols dynamic
                    if grid_img:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        grid_save_path = os.path.join(output_dir, f"poses_grid_{timestamp}.png")
                        grid_img.save(grid_save_path)
                        all_generated_files.append(grid_save_path)
                        generated_poses = results # Store the actual PIL images
                        return grid_img, f"تم توليد {len(results)} وضعيات بنجاح مع تحسين للوجه!"
                    else:
                        return None, "فشل إنشاء شبكة الصور للوضعيات."


            except Exception as e:
                print(f"An error occurred during pose generation: {e}")
                import traceback
                traceback.print_exc()
                return None, f"خطأ أثناء توليد الوضعيات: {e}"

    except Exception as e:
        print(f"Error in image processing for body poses: {e}")
        return None, f"خطأ في معالجة الصورة للوضعيات: {e}"

    return None, "فشل توليد الوضعيات."

def create_pose_image(keypoints, size=(512, 768)):
    """Create an OpenPose visualization with improved visibility"""
    image = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(image)
    
    # Draw connections between keypoints with thicker lines first
    connections = [
        # Torso
        (1, 2), (2, 3),  # Neck to chest to waist
        # Left arm
        (2, 4), (4, 5), (5, 6),  # Chest to left hand
        # Right arm
        (2, 7), (7, 8), (8, 9),  # Chest to right hand
        # Left leg
        (3, 10), (10, 11), (11, 12),  # Waist to left foot
        # Right leg
        (3, 13), (13, 14), (14, 15)  # Waist to right foot
    ]
    
    for conn in connections:
        if len(keypoints) > max(conn):  # Check if points exist
            start = keypoints[conn[0]]
            end = keypoints[conn[1]]
            start_pos = (start[0] * size[0], start[1] * size[1])
            end_pos = (end[0] * size[0], end[1] * size[1])
            draw.line([start_pos, end_pos], fill='white', width=3)  # ← زيادة سمك الخطوط
    
    # Draw keypoints as larger circles on top of connections
    for i, point in enumerate(keypoints):
        x, y = point[0] * size[0], point[1] * size[1]
        radius = 5  # ← زيادة حجم النقاط
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='white', outline='gray')
        
        # Add point labels for head and important joints
        if i == 0:  # Head
            draw.text((x+10, y), "Head", fill="white")
        elif i == 3:  # Waist
            draw.text((x+10, y), "Waist", fill="white")
            
    return image

def create_standing_pose():
    """Generate normalized keypoints for standing pose"""
    return [
        # Head
        (0.5, 0.1),  # Top of head
        (0.5, 0.15), # Neck
        # Body
        (0.5, 0.3),  # Chest
        (0.5, 0.45), # Waist
        # Arms
        (0.35, 0.3), # Left shoulder
        (0.3, 0.45), # Left elbow
        (0.25, 0.6), # Left hand
        (0.65, 0.3), # Right shoulder
        (0.7, 0.45), # Right elbow
        (0.75, 0.6), # Right hand
        # Legs
        (0.45, 0.6), # Left hip
        (0.45, 0.75), # Left knee
        (0.45, 0.9), # Left foot
        (0.55, 0.6), # Right hip
        (0.55, 0.75), # Right knee
        (0.55, 0.9), # Right foot
    ]

def create_sitting_pose():
    """Generate normalized keypoints for sitting pose"""
    return [
        # Head
        (0.5, 0.2),  # Top of head
        (0.5, 0.25), # Neck
        # Body
        (0.5, 0.4),  # Chest
        (0.5, 0.55), # Waist
        # Arms
        (0.35, 0.4), # Left shoulder
        (0.3, 0.5),  # Left elbow
        (0.25, 0.6), # Left hand
        (0.65, 0.4), # Right shoulder
        (0.7, 0.5),  # Right elbow
        (0.75, 0.6), # Right hand
        # Legs (sitting position)
        (0.45, 0.55), # Left hip
        (0.4, 0.7),   # Left knee
        (0.35, 0.85), # Left foot
        (0.55, 0.55), # Right hip
        (0.6, 0.7),   # Right knee
        (0.65, 0.85), # Right foot
    ]

def create_walking_pose():
    """Generate normalized keypoints for walking pose (improved)"""
    return [
        # Head
        (0.5, 0.1),  # Top of head
        (0.5, 0.15), # Neck
        # Body
        (0.5, 0.3),  # Chest
        (0.5, 0.45), # Waist
        # Arms (walking motion)
        (0.35, 0.3), # Left shoulder
        (0.25, 0.4), # Left elbow (forward)
        (0.2, 0.5),  # Left hand
        (0.65, 0.3), # Right shoulder
        (0.75, 0.4), # Right elbow (back)
        (0.82, 0.5),  # Right hand (more visible)
        # Legs (walking stance - more pronounced)
        (0.43, 0.6), # Left hip
        (0.35, 0.75), # Left knee (more forward)
        (0.3, 0.9), # Left foot (more forward)
        (0.57, 0.6), # Right hip
        (0.67, 0.75),# Right knee (more back)
        (0.72, 0.9),  # Right foot (more back)
    ]

def analyze_character(image):
    """
    تحليل خصائص الشخصية الكرتونية لإنشاء وصف أفضل للوضعيات
    """
    try:
        # For now, return a basic description from the existing analyze_image function
        # Later can be enhanced with more specific character analysis
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        basic_features = analyze_image(image_np)
        
        # Add more specific details for full body generation
        return f"{basic_features}, full figure, detailed body proportions"
    except Exception as e:
        print(f"Error analyzing character: {e}")
        return "character with natural proportions"

# --------------------------------------------------
# المرحلة 5: التعديل على الصورة (Inpainting)
# --------------------------------------------------
# Gradio's Image component with tool='sketch' can provide image+mask
def edit_image_inpainting(image_mask_dict, edit_prompt, api_choice="Local Stable Diffusion"):
    global edited_image, all_generated_files
    if image_mask_dict is None or edit_prompt is None or edit_prompt.strip() == "":
        return None, "Error: Please provide an image, draw a mask, and enter an edit description."

    try:
        # Convert input images correctly
        if isinstance(image_mask_dict["image"], np.ndarray):
            base_image = Image.fromarray(image_mask_dict["image"]).convert("RGB")
        else:
            base_image = image_mask_dict["image"].convert("RGB")

        if isinstance(image_mask_dict["mask"], np.ndarray):
            mask_image = Image.fromarray(image_mask_dict["mask"]).convert("RGB")
        else:
            mask_image = image_mask_dict["mask"].convert("RGB")

        # Inpainting models expect a white mask where changes should happen
        # Gradio might provide it differently, ensure mask is correct format (e.g., white on black)
        # Example conversion if mask is RGBA with transparency:
        # mask_image = mask_image.split()[-1].convert('L').point(lambda x: 255 if x > 100 else 0, '1')

        print(f"Inpainting using: {api_choice}")
        output_image = None

        if api_choice == "Local Stable Diffusion":
            if not torch.cuda.is_available():
                 return None, "Error: CUDA GPU required for local Stable Diffusion."
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                     SD_INPAINT_MODEL, torch_dtype=torch.float16
                ).to("cuda")

                # Ensure images are the size the model expects (often 512x512)
                base_image = base_image.resize((512, 512))
                mask_image = mask_image.resize((512, 512))

                output = pipe(
                    prompt=edit_prompt,
                    image=base_image,
                    mask_image=mask_image,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                )
                output_image = output.images[0]

                pipe = None # Release VRAM
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error during local SD inpainting: {e}")
                return None, f"Error: {e}"

        elif api_choice == "Other API (e.g., Replicate, Stability AI)":
            # Placeholder: API call for inpainting
            print("NOTE: API calls for inpainting not implemented. Using placeholder.")
            # Example (Conceptual):
            # api_key = os.getenv("STABILITY_API_KEY")
            # headers = { "Authorization": f"Bearer {api_key}", "Accept": "image/png" }
            # files = {
            #     'init_image': image_to_bytes(base_image),
            #     'mask_image': image_to_bytes(mask_image),
            #     'prompt': (None, edit_prompt),
            #     # Other params...
            # }
            # response = requests.post("https://api.stability.ai/v1/generation/stable-diffusion-v1-5/image-to-image/masking", headers=headers, files=files)
            # ... handle response ...
            output_image = base_image # Placeholder
            return None, "API inpainting not implemented. Placeholder used."

        if output_image:
            edited_image = output_image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f"edited_{timestamp}.png")
            output_image.save(save_path)
            all_generated_files.append(save_path)
            print(f"Edited image saved to {save_path}")
            return output_image, f"Image edited successfully! Saved to {save_path}"
        else:
            return None, f"Failed to edit image using {api_choice}."
    except Exception as e:
        print(f"Error in image processing: {e}")
        return None, f"Error: {e}"


# --------------------------------------------------
# المرحلة 6: جعل الشخصية تتكلم (TTS + Lip Sync)
# --------------------------------------------------
def generate_talking_video(image_to_animate, text_to_speak, animation_choice="Local XTTS"):
    global final_video_path, all_generated_files
    
    # Check inputs
    if image_to_animate is None or text_to_speak is None or text_to_speak.strip() == "":
        return None, "Error: Please provide a character image and text to speak."

    # Initialize variables
    source_image = None
    audio_path = None
    tts_error = None
    output_video_path = None
    status_message = ""

    # Get source image
    if 'edited_image' in globals() and edited_image is not None:
        source_image = edited_image
    elif 'current_cartoon_image' in globals() and current_cartoon_image is not None:
        source_image = current_cartoon_image
    else:
        source_image = Image.fromarray(image_to_animate).convert("RGB")

    print(f"Generating talking video using: {animation_choice}")

    try:
        # --- Step 6a: Text-to-Speech (TTS) ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"speech_{timestamp}.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        
        if animation_choice == "Local XTTS":
            try:
                print("Initializing TTS...")
                
                # Use gTTS instead of TTS for reliable Arabic support
                try:
                    from gtts import gTTS
                    
                    print(f"Using gTTS for Arabic text-to-speech")
                    # Create the audio directory if it doesn't exist
                    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                    
                    # Generate the audio with gTTS
                    tts_engine = gTTS(text=text_to_speak, lang='ar', slow=False)
                    tts_engine.save(audio_path)
                    print(f"Audio saved to {audio_path}")
                    all_generated_files.append(audio_path)
                except Exception as e:
                    tts_error = f"Error during TTS generation: {e}"
                    print(tts_error)
                    return None, tts_error
                
                # --- Step 6b: Lip Sync Animation ---
                # Save source image
                temp_image_path = os.path.join(output_dir, f"temp_source_{timestamp}.png")
                source_image.save(temp_image_path)
                all_generated_files.append(temp_image_path)

                video_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_video_filename = f"talking_video_{video_timestamp}.mp4"
                output_video_path = os.path.join(output_dir, output_video_filename)

                # Try MEMO if available
                memo_success = False
                if "Local XTTS" in animation_choice and 'MEMO_HELPER' in globals() and MEMO_HELPER:
                    try:
                        print("Attempting to use MEMO for lip-sync animation...")
                        # First ensure omegaconf is installed
                        try:
                            import omegaconf
                            print("omegaconf is already installed")
                        except ImportError:
                            print("Installing omegaconf...")
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "omegaconf"])
                            
                        output_video_path_memo = MEMO_HELPER.generate_video(temp_image_path, audio_path, output_dir)
                        if output_video_path_memo and os.path.exists(output_video_path_memo):
                            final_video_path = output_video_path_memo
                            all_generated_files.append(final_video_path)
                            memo_success = True
                            # Clean up temporary image file
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)
                            return final_video_path, "Talking video generated with MEMO lip-sync technology!"
                        else:
                            print("MEMO video generation failed, falling back to simple video.")
                    except Exception as e:
                        print(f"Error during MEMO generation: {e}")
                        print("Falling back to simple video.")

                # If MEMO failed or not available, try moviepy
                if not memo_success:
                    try:
                        # Check if moviepy is installed, if not, install it
                        try:
                            import moviepy.editor
                            print("moviepy is already installed")
                        except ImportError:
                            print("Installing moviepy...")
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
                            import moviepy.editor
                        
                        print("Creating simple video with moviepy...")
                        # Create a video clip from the static image
                        image_clip = moviepy.editor.ImageClip(temp_image_path, duration=5)  # 5 second default duration
                        
                        # Load the audio file
                        audio_clip = moviepy.editor.AudioFileClip(audio_path)
                        
                        # Set the duration of the image clip to match the audio
                        image_clip = image_clip.set_duration(audio_clip.duration)
                        
                        # Set the audio of the clip
                        video_clip = image_clip.set_audio(audio_clip)
                        
                        # Write the result to a file
                        video_clip.write_videofile(output_video_path, fps=24)
                        
                        print(f"Simple talking video created at {output_video_path}")
                        all_generated_files.append(output_video_path)
                        final_video_path = output_video_path
                        
                        # Clean up temporary image file
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            
                        return output_video_path, "Talking video generated successfully! (Simple animation without lip sync)"
                        
                    except Exception as e:
                        animation_error = f"Error creating simple video: {e}"
                        print(animation_error)
                        return None, f"Failed to generate video: {animation_error}"
                    
            except Exception as e:
                overall_error = f"Error in video generation process: {e}"
                print(overall_error)
                return None, f"Failed to generate video. {overall_error}"
        
        # Handle other animation choices here if needed
        else:
            return None, f"Animation choice '{animation_choice}' not implemented yet."
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, f"Failed to generate video due to an unexpected error: {e}"

    # This line should only be reached if no other return statement was hit
    return None, "Failed to generate video due to an unknown error."

# --------------------------------------------------
# المرحلة 7: حفظ النتائج وتصديرها
# --------------------------------------------------
def download_all_results():
    if not all_generated_files:
        print("No files generated yet.")
        return None # Return None for Gradio file download

    zip_filename = f"cartoon_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_filepath = os.path.join(output_dir, zip_filename)

    try:
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for file_path in all_generated_files:
                if os.path.exists(file_path):
                    # Add file to zip using its basename to avoid deep paths
                    zipf.write(file_path, os.path.basename(file_path))
        print(f"Created zip file: {zip_filepath}")
        # Return the path for Gradio's File component download
        return zip_filepath
    except Exception as e:
        print(f"Error creating zip file: {e}")
        return None

# --------------------------------------------------
# واجهة المستخدم (Gradio Example)
# --------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎨 مولد الشخصيات الكرتونية المتحركة 🗣️")
    gr.Markdown("الهدف: تحويل صورة واقعية إلى شخصية كرتونية 3D، مع تعابير ووضعيات مختلفة، وتعديلها، ثم جعلها تتكلم.")

    api_choice = gr.Radio(
        label="🧠 اختر محرك الذكاء الاصطناعي",
        choices=["Local Stable Diffusion", "D-ID API", "HeyGen API", "Other API (e.g., Replicate)"],
        value="Local Stable Diffusion", # Default
        info="Local يتطلب GPU قوي وإعداد مسبق. APIs تتطلب مفاتيح واشتراكات."
    )

    with gr.Tabs():
        with gr.TabItem("👤 المرحلة 1 و 2: الصورة الأساسية -> كرتون"):
            gr.Markdown("ارفع صورة شخص حقيقية لتحويلها إلى نمط كرتوني ثلاثي الأبعاد.")
            with gr.Row():
                input_image_s2 = gr.Image(label="الصورة الواقعية", type="numpy")
                output_image_s2 = gr.Image(label="النتيجة: شخصية كرتونية", type="pil")
            status_s2 = gr.Textbox(label="الحالة", interactive=False)
            btn_s2 = gr.Button("🚀 حوّل إلى كرتون!")

        with gr.TabItem("😊 المرحلة 3: تعابير الوجه"):
            gr.Markdown("إنشاء تعابير وجه مختلفة للشخصية الكرتونية.")
            with gr.Row():
                output_expressions_s3 = gr.Image(
                    label="شبكة التعابير",
                    type="pil"
                )
            status_s3 = gr.Textbox(
                label="الحالة",
                interactive=False,
                show_label=True
            )
            btn_s3 = gr.Button("🎭 ولّد التعابير!")

        with gr.TabItem("🧍 المرحلة 4: وضعيات الجسم"):
             gr.Markdown("إنشاء 3 وضعيات جسم كاملة للشخصية الكرتونية.")
             with gr.Row():
                  # Hidden state pass again
                  # base_cartoon_input_s4 = gr.Image(visible=False, type="numpy")
                  output_poses_s4 = gr.Image(label="شبكة الوضعيات (واقف، جالس، يمشي...)", type="pil")
             status_s4 = gr.Textbox(label="الحالة", interactive=False)
             btn_s4 = gr.Button("🏃‍♀️ ولّد الوضعيات!")

        with gr.TabItem("✏️ المرحلة 5: تعديل الصورة (Inpainting)"):
             gr.Markdown("حدد منطقة في الصورة الكرتونية (باستخدام أداة الرسم) واكتب وصفًا لتعديلها (مثل 'إضافة قبعة زرقاء').")
             with gr.Row():
                 # Input image allows drawing mask
                 input_image_s5 = gr.Image(
                     label="الصورة للتعديل (ارسم القناع)", 
                     type="pil", 
                     source="upload",
                     tool="sketch"  # Changed back to 'tool' for gradio 3.50.2
                 )
             edit_prompt_s5 = gr.Textbox(  # Added missing text input
                 label="وصف التعديل المطلوب",
                 placeholder="مثال: إضافة قبعة زرقاء"
             )
             output_image_s5 = gr.Image(label="النتيجة المعدلة", type="pil")
             status_s5 = gr.Textbox(label="الحالة", interactive=False)
             btn_s5 = gr.Button("✨ طبّق التعديل!")

        with gr.TabItem("💬 المرحلة 6: جعل الشخصية تتكلم"):
            gr.Markdown("أدخل النص الذي تريده أن تقوله الشخصية. سيتم توليد الصوت ومزامنته مع حركة الشفاه في فيديو.")
            with gr.Row():
                 # Allow selecting image? For now, uses last generated/edited one.
                 input_image_s6_display = gr.Image(label="الصورة للتحريك (آخر صورة تم إنشاؤها/تعديلها)", type="pil", interactive=False) # Display only
                 text_input_s6 = gr.Textbox(label="النص المراد قوله", lines=3)
            animation_choice_s6 = gr.Radio(
                label="🎬 اختر أداة التحريك",
                choices=[
                    "Local XTTS",
                    "Local XTTS + D-ID",
                    "Local XTTS + HeyGen",
                    "D-ID API",
                    "HeyGen API"
                ],
                value="Local XTTS",
                info="XTTS provides high-quality Arabic TTS. Choose '+API' options to use XTTS audio with online lip sync."
            )
            output_video_s6 = gr.Video(label="الفيديو النهائي")
            status_s6 = gr.Textbox(label="الحالة", interactive=False)
            btn_s6 = gr.Button("🎤 تكلم!")

        with gr.TabItem("💾 المرحلة 7: حفظ وتصدير الكل"):
            gr.Markdown("قم بتنزيل جميع الصور والفيديوهات التي تم إنشاؤها في ملف مضغوط واحد.")
            download_button_s7 = gr.Button("📥 تنزيل كل النتائج (.zip)")
            download_output_s7 = gr.File(label="ملف النتائج المضغوط")
            status_s7 = gr.Textbox(label="الحالة", interactive=False) # Optional status

    # --- Link Functions to Buttons ---
    # Stage 2
    btn_s2.click(
        fn=cartoonify_image,
        inputs=[input_image_s2, api_choice],
        outputs=[output_image_s2, status_s2]
    ).then(
        fn=lambda img: img,  # Simplified update function
        inputs=[output_image_s2],
        outputs=input_image_s6_display  # Remove list brackets
    )

    # Stage 3
    btn_s3.click(
        fn=generate_expressions_batch,
        inputs=[output_image_s2, api_choice],  # Make sure output_image_s2 is properly configured
        outputs=[output_expressions_s3, status_s3]
    )

    # Stage 4
    btn_s4.click(
        fn=generate_body_poses,
        inputs=[output_image_s2, api_choice], # Takes output from S2 as input
        outputs=[output_poses_s4, status_s4]
     )

    # Stage 5
    # Needs the image+mask dictionary from the sketch tool
    # input_image_s5 provides {'image': pil, 'mask': pil} when tool='sketch'
    btn_s5.click(
        fn=edit_image_inpainting,
        inputs=[input_image_s5, edit_prompt_s5, api_choice],
        outputs=[output_image_s5, status_s5]
    ).then(
        lambda img: img if img is not None else gr.update(), # Update S6 display if edited
        inputs=output_image_s5,
        outputs=[input_image_s6_display]
    )


    # Stage 6
    # The function 'generate_talking_video' needs to internally decide which image to use
    # (e.g., last edited or last cartoonized) based on global state 'edited_image' or 'current_cartoon_image'
    # The lambda updates above are simpler for direct passing.
    # Let's rely on the generate_talking_video function checking globals.
    # We just need *an* image displayed. The lambda already updates it.
    # This @gr.on might be redundant if the lambdas work.
    @gr.on(triggers=[output_image_s2.change, output_image_s5.change])
    def update_s6_image_display(s2_img, s5_img):
        # Function to update the display image in Stage 6 when S2 or S5 changes
        # Use the most recent one (simple logic: if s5 changed, use it, else use s2)
        # This requires capturing the changes via events, or the generate_talking_video
        # function just checks the global variables like `edited_image` and `current_cartoon_image`
        # The lambda updates above are simpler for direct passing.
        # Let's rely on the generate_talking_video function checking globals.
        # We just need *an* image displayed. The lambda already updates it.
        # This @gr.on might be redundant if the lambdas work.
        return s5_img if s5_img is not None else s2_img


    btn_s6.click(
        fn=generate_talking_video,
        # Pass the currently displayed image (updated by S2/S5) and the text
        inputs=[input_image_s6_display, text_input_s6, animation_choice_s6],
        outputs=[output_video_s6, status_s6]
     )

    # Stage 7
    download_button_s7.click(
        fn=download_all_results,
        inputs=[],
        outputs=[download_output_s7]
        # Can add a status update here too
    ).then(lambda: "Zip file ready for download." if download_output_s7 is not None else "No files to zip or error occurred.", outputs=status_s7)


# --- Launch the Gradio App ---
if __name__ == "__main__":
    app.queue()  # Add queue before launch
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Only set to True if you want public access
    )

class MEMOHelper:
    def __init__(self, memo_path: str = "memo"):
        self.memo_path = memo_path
        self.config_path = os.path.join(memo_path, "configs/inference.yaml")
        
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

def setup_xtts():
    """Ensures XTTS is properly set up with Arabic model"""
    try:
        from TTS.api import TTS
        
        # Check if Arabic model exists
        tts = TTS("tts_models/ar/cv/vits")
        print("XTTS Arabic model ready!")
        return True
        
    except Exception as e:
        print(f"Error setting up XTTS: {e}")
        return False

# Launch app if running as main script
if __name__ == "__main__":
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

# This can be used for debugging TTS models
try:
    from TTS.utils.manage import ModelManager
    manager = ModelManager()
    
    print("Available TTS models:")
    # Correctly access the available models
    for model_type in manager.models_dict:
        print(f"Model type: {model_type}")
        for lang in manager.models_dict[model_type]:
            print(f"  Language: {lang}")
            for dataset in manager.models_dict[model_type][lang]:
                print(f"    Dataset: {dataset}")
                for model in manager.models_dict[model_type][lang][dataset]:
                    print(f"      Model: {model}")
except Exception as e:
    print(f"Error listing TTS models: {e}")