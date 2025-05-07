# 🎨 مولّد الشخصيات الكرتونية ثلاثية الأبعاد المتحركة بالذكاء الاصطناعي 🗣️
<p align="center">
  <img src="assets/logo.png" alt="شعار KH" width="120"/> <!-- تأكد من وضع الشعار في مجلد assets -->
</p>

يهدف هذا المشروع إلى توفير أداة متكاملة باستخدام بايثون لتحويل الصور الواقعية إلى شخصيات كرتونية بأسلوب ثلاثي الأبعاد (شبيه بأسلوب بيكسار). يتيح التطبيق إنشاء تعابير وجه متنوعة، ووضعيات جسم مختلفة للشخصية، وإمكانية تعديل الصورة، وصولاً إلى توليد مقاطع فيديو ناطقة. يعتمد المشروع على نماذج ذكاء اصطناعي محلية (مثل Stable Diffusion و ControlNet) أو خدمات سحابية (APIs مثل D-ID و HeyGen)، ويقدم واجهة مستخدم تفاعلية مبنية باستخدام Gradio.

---

## 🌟 الميزات الرئيسية

*   **🖼️ تحويل الصورة إلى كرتون:** ارفع صورة شخص حقيقي لتحويلها إلى نمط كرتوني ثلاثي الأبعاد عالي الجودة.
*   **😊 توليد تعابير الوجه:** إنشاء مجموعة من 6 تعابير وجه مميزة (سعيد، حزين، غاضب، مندهش، طبيعي، مفكر) مع الحفاظ على هوية الشخصية.
*   **🧍 إنشاء وضعيات الجسم:** توليد 3 وضعيات جسم كاملة للشخصية الكرتونية (مثل واقف، جالس، يمشي) مع الحفاظ على تفاصيل الوجه.
*   **✍️ تعديل الصورة (Inpainting):** حدد منطقة في الصورة الكرتونية باستخدام أداة الرسم المدمجة واكتب وصفًا للتعديل المطلوب (مثال: "إضافة نظارات شمسية"، "تغيير لون الشعر إلى أزرق").
*   **🎤 جعل الشخصية تتكلم:** أدخل نصًا ليتم تحويله إلى كلام (TTS) باستخدام gTTS (يدعم العربية)، ثم محاولة مزامنة الصوت مع حركة الشفاه لإنشاء فيديو باستخدام MEMO (محليًا) أو MoviePy كبديل بسيط.
*   **💾 حفظ وتصدير النتائج:** يتم تتبع جميع الصور ومقاطع الفيديو التي تم إنشاؤها تلقائيًا، مع إمكانية تنزيلها كلها مجمعة في ملف مضغوط واحد (`.zip`).

---

## 🛠️ المتطلبات والإعداد

### 🐍 بيئة بايثون

يوصى بشدة باستخدام بيئة افتراضية (`venv` أو `conda`) لتنظيم تبعيات المشروع وتجنب التعارض بين المكتبات.


 1. إنشاء بيئة افتراضية (مثال باستخدام venv)
python -m venv venv_cartoon

 2. تفعيل البيئة
 على Windows
venv_cartoon\Scripts\activate
 على macOS/Linux
source venv_cartoon/bin/activate

 3. الآن يمكنك تثبيت المكتبات داخل هذه البيئة النشطة

### 🔧 تثبيت المكتبات الأساسية

تأكد من أن بيئتك الافتراضية مفعلة قبل تشغيل هذه الأوامر.

1. PyTorch (مع دعم CUDA إذا كان لديك GPU من NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

ملاحظة هامة:

استبدل cu118 بإصدار CUDA المتوافق مع تعريفات NVIDIA Driver المثبتة لديك (مثلاً cu117، cu121). يمكنك التحقق من إصدار CUDA الذي يدعمه تعريفك عن طريق الأمر nvidia-smi في الطرفية.

إذا لم يكن لديك GPU من NVIDIA، يمكنك تثبيت إصدار CPU من PyTorch (سيكون الأداء أبطأ بكثير للنماذج الكبيرة):
pip install torch torchvision torchaudio

2. مكتبات Hugging Face و Diffusers:
pip install diffusers transformers accelerate bitsandbytes sentencepiece
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

accelerate: لتحسين الأداء وتسريع تحميل النماذج الكبيرة.

bitsandbytes: (اختياري، لنظام Linux بشكل أساسي) لتمكين تحميل النماذج بدقة 8-bit لتوفير الذاكرة.

sentencepiece: قد تكون مطلوبة لبعض نماذج Transformers.

3. Gradio وأدوات مساعدة:
pip install gradio pillow opencv-python numpy gtts requests moviepy imageio-ffmpeg omegaconf
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

moviepy و imageio-ffmpeg: لإنشاء مقاطع فيديو بسيطة محليًا.

omegaconf: مكتبة مطلوبة لتشغيل أداة MEMO لمزامنة الشفاه.

4. مكتبات تحسين الوجه (اختياري ولكن موصى به):
pip install gfpgan
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

gfpgan: لتحسين جودة الوجوه في الصور المولدة.

### 🗝️ مفاتيح API (اختياري)

إذا كنت تخطط لاستخدام خدمات سحابية مثل D-ID أو HeyGen، ستحتاج للحصول على مفاتيح API الخاصة بها. أفضل ممارسة هي تعيين هذه المفاتيح كمتغيرات بيئة بدلاً من كتابتها مباشرة في الكود.

# مثال على تعيين متغيرات البيئة (Linux/macOS) في ملف .bashrc أو .zshrc
export D_ID_API_KEY="YOUR_D_ID_API_KEY"
export HEYGEN_API_KEY="YOUR_HEYGEN_API_KEY"

# مثال لـ Windows CMD (يجب تشغيله في كل جلسة طرفية جديدة أو تعيينها بشكل دائم عبر لوحة التحكم)
set D_ID_API_KEY=YOUR_D_ID_API_KEY
set HEYGEN_API_KEY=YOUR_HEYGEN_API_KEY

# مثال لـ Windows PowerShell
$env:D_ID_API_KEY="YOUR_D_ID_API_KEY"
$env:HEYGEN_API_KEY="YOUR_HEYGEN_API_KEY"
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
### 📥 تحميل النماذج المحلية (تلقائي)

عند تشغيل التطبيق لأول مرة واستخدام النماذج المحلية (مثل Stable Diffusion، ControlNet، VAE، Midas)، ستقوم مكتبة diffusers ومكتبات أخرى بتنزيل هذه النماذج تلقائيًا وتخزينها مؤقتًا (caching) في نظامك.

حجم النماذج: كن على علم بأن هذه النماذج يمكن أن تكون كبيرة جدًا (عدة جيجابايت لكل منها).

المتطلبات: تأكد من أن لديك مساحة كافية على القرص الصلب واتصال جيد بالإنترنت.

Clone MEMO Repository (لتفعيل مزامنة الشفاه المتقدمة محليًا)

إذا كنت ترغب في استخدام MEMO لمزامنة الشفاه (وهو أفضل من MoviePy البسيط):

git clone https://github.com/memoavatar/memo.git
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

تأكد من أن مجلد memo المستنسخ موجود في نفس الدليل الذي يوجد به ملف app.py. سيقوم MEMOHelper بمحاولة تثبيت متطلبات memo الإضافية إذا لزم الأمر.

### ✅ التحقق من التثبيت ودعم CUDA

يمكنك تشغيل السكريبت التالي للتحقق من تثبيت PyTorch ودعم CUDA:

# check_setup.py
import torch

print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    try:
        # قد يختلف هذا بناءً على كيفية تجميع PyTorch
        print(f"CUDA Version (PyTorch compiled with): {torch.version.cuda}")
    except AttributeError:
        print("Could not retrieve PyTorch CUDA compile version.")
else:
    print("CUDA not available. Operations will run on CPU (this will be very slow for image generation).")

try:
    import diffusers
    print(f"Diffusers version: {diffusers.__version__}")
    import gradio
    print(f"Gradio version: {gradio.__version__}")
except ImportError as e:
    print(f"A required library is missing: {e}")
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
### 🚀 طريقة التشغيل

تفعيل البيئة الافتراضية: تأكد من أنك قمت بتفعيل بيئتك الافتراضية (مثال: venv_cartoon\Scripts\activate).

تعيين مفاتيح API (إذا كنت ستستخدمها): تأكد من أن متغيرات البيئة لمفاتيح API الخاصة بك قد تم تعيينها.

تشغيل التطبيق: انتقل إلى مجلد المشروع في الطرفية وقم بتشغيل السكريبت الرئيسي:

python app.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

فتح الواجهة: سيقوم Gradio بتشغيل خادم محلي. ابحث عن عنوان URL في مخرجات الطرفية (عادةً ما يكون http://127.0.0.1:7860 أو http://0.0.0.0:7860) وافتحه في متصفح الويب الخاص بك.

اتبع المراحل في الواجهة:

ابدأ برفع صورة في "المرحلة 1 و 2" لتحويلها إلى شخصية كرتونية.

استخدم الصورة الناتجة في المراحل التالية لإنشاء تعابير الوجه، وضعيات الجسم، وإجراء التعديلات.

في "المرحلة 6"، أدخل نصًا ليتم تحويله إلى فيديو ناطق.

استخدم "المرحلة 7" لتنزيل جميع الملفات التي تم إنشاؤها في ملف مضغوط واحد.

### 💡 ملاحظات هامة

متطلبات GPU: لتشغيل النماذج المحلية بكفاءة (خاصة Stable Diffusion مع ControlNet)، يوصى بشدة باستخدام GPU من NVIDIA مع ذاكرة فيديو (VRAM) كافية.

8GB VRAM: الحد الأدنى المقبول، قد تحتاج إلى استخدام إعدادات أقل أو نماذج أخف.

12GB+ VRAM: موصى به للحصول على أداء أفضل وسرعات توليد أسرع.

التشغيل على CPU: ممكن نظريًا، ولكنه سيكون بطيئًا للغاية وغير عملي لمعظم مراحل توليد الصور.

ذاكرة النظام (RAM): تأكد من أن لديك 16GB RAM على الأقل، ويفضل 32GB إذا كنت تتعامل مع نماذج كبيرة وعمليات متعددة.

مشاكل المسارات مع الأحرف غير الإنجليزية: يفضل بشدة أن يكون مسار مجلد المشروع لا يحتوي على أي أحرف عربية أو رموز خاصة لتجنب مشاكل التوافق مع بعض المكتبات (مثل OpenCV).

تحديث المكتبات: حافظ على تحديث مكتباتك (خاصة diffusers, transformers, torch, gradio) للحصول على أحدث الميزات وإصلاحات الأخطاء.

### 🙏 شكر وتقدير

تم بناء هذا المشروع بالاعتماد على العديد من الأدوات والمكتبات الرائعة مفتوحة المصدر، بما في ذلك:

PyTorch

Hugging Face (خاصة مكتبات 🤗 diffusers و transformers)

Gradio

OpenCV

Pillow (PIL Fork)

GFPGAN (لتحسين الوجه)

MEMO (لتحريك الشخصيات)

خدمات مثل D-ID و HeyGen (كخيارات API)

### 📜 الترخيص

هذا المشروع مقدم بشكل أساسي كأداة تعليمية وعرض توضيحي لإمكانيات الذكاء الاصطناعي في توليد وتحريك الشخصيات.

الكود المصدري للمشروع: مرخص بموجب ترخيص MIT (أو الترخيص الذي تختاره).

النماذج والخدمات المستخدمة:

نماذج مثل Stable Diffusion لها تراخيصها الخاصة (غالبًا ما تكون OpenRAIL أو ما شابه، والتي قد تفرض قيودًا على بعض الاستخدامات).

خدمات API مثل D-ID و HeyGen لها شروط خدمة وتراخيص استخدام خاصة بها.

يرجى مراجعة التراخيص الفردية لكل مكون (نماذج، مكتبات، خدمات) قبل أي استخدام يتجاوز النطاق التعليمي أو الشخصي، خاصة للاستخدامات التجارية.

IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END