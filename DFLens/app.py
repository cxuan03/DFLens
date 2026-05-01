"""
DFLens v3 — Multimodal Deepfake Detection Flask Application

A web-based deepfake detection tool supporting image, video and audio analysis.
Vision backend: CLIP ViT-B/32 fine-tuned on FF++ C23 + CelebDF-v2
Audio backend: AASIST3 (primary) > AudioCNN (secondary) > Heuristic (fallback)

Features:
- MTCNN face detection with dual-path consensus mechanism
- Bayesian prior correction for class imbalance
- Image suitability pre-screening for OOD rejection
- Grad-CAM explainability heatmaps
- ELA (Error Level Analysis) image forgery detection
- Automated PDF report generation

Usage: python app.py
"""

# ── Standard library and core dependencies ──
import os,io,base64,tempfile,json,cv2,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from flask import Flask,render_template,request,jsonify,send_file

# ── Optional dependencies (graceful fallback if missing) ──
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_OK=True
except ImportError: GRADCAM_OK=False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image as RLImage,Table,TableStyle
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    PDF_OK=True
except ImportError: PDF_OK=False
try:
    from facenet_pytorch import MTCNN
    MTCNN_OK=True
except ImportError: MTCNN_OK=False

# ── Configuration ──
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_DIR=os.path.join(BASE_DIR,"models")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE=224; LABEL_REAL=0; LABEL_FAKE=1

# Model weight paths — can be overridden via environment variables
CLIP_PATH=os.environ.get("CLIP_MODEL_PATH",os.path.join(MODEL_DIR,"clip_detector.pth"))
EFFNET_PATH=os.environ.get("EFFNET_MODEL_PATH",os.path.join(MODEL_DIR,"best_model.pth"))
AASIST_PATH=os.environ.get("AASIST_MODEL_PATH",os.path.join(MODEL_DIR,"AASIST.pth"))
VOICE_PATH=os.environ.get("VOICE_MODEL_PATH",os.path.join(MODEL_DIR,"best_voice_model.pth"))

# Auto-detect available backends based on which model files exist
VISION_BACKEND="clip" if os.path.exists(CLIP_PATH) else "efficientnet"
AUDIO_BACKEND="aasist" if os.path.exists(AASIST_PATH) else ("audiocnn" if os.path.exists(VOICE_PATH) else "heuristic")
print(f"Config: device={DEVICE} vision={VISION_BACKEND} audio={AUDIO_BACKEND}")

# ══════════════════════════════════════════════════════════════
# VISION MODEL — CLIP ViT-B/32 Deepfake Detector
# ══════════════════════════════════════════════════════════════
if VISION_BACKEND=="clip":
    from transformers import CLIPModel
    class CLIPDeepfakeDetector(nn.Module):
        # CLIP ViT-B/32 with Linear(768, 2) head. Last 4 layers unfrozen.
        def __init__(self,name="openai/clip-vit-base-patch32",unfreeze_last_n=4):
            super().__init__()
            self.vision_encoder=CLIPModel.from_pretrained(name).vision_model
            self.feat_dim=self.vision_encoder.config.hidden_size
            for p in self.vision_encoder.parameters(): p.requires_grad=False
            num_layers=len(self.vision_encoder.encoder.layers)
            for i in range(num_layers-unfreeze_last_n,num_layers):
                for p in self.vision_encoder.encoder.layers[i].parameters(): p.requires_grad=True
            for p in self.vision_encoder.post_layernorm.parameters(): p.requires_grad=True
            self.classifier=nn.Linear(self.feat_dim,2)
        def forward(self,x):
            feat=F.normalize(self.vision_encoder(pixel_values=x).pooler_output,p=2,dim=-1)
            return self.classifier(feat),feat
    img_transform=transforms.Compose([transforms.Resize(IMG_SIZE),transforms.CenterCrop(IMG_SIZE),transforms.ToTensor(),transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711])])
    def load_vision():
        m=CLIPDeepfakeDetector()
        if os.path.exists(CLIP_PATH):
            s=torch.load(CLIP_PATH,map_location=DEVICE,weights_only=False)
            m.load_state_dict(s["model_state"] if isinstance(s,dict) and "model_state" in s else s)
            print(f"  CLIP loaded: {CLIP_PATH}")
        m.to(DEVICE).eval(); return m
    def gradcam_layers(m): return [m.vision_encoder.encoder.layers[-1].layer_norm1]
else:
    import timm
    img_transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def load_vision():
        m=timm.create_model("efficientnet_b0",pretrained=False,num_classes=2)
        if os.path.exists(EFFNET_PATH):
            m.load_state_dict(torch.load(EFFNET_PATH,map_location=DEVICE,weights_only=False)); print(f"  EfficientNet loaded: {EFFNET_PATH}")
        m.to(DEVICE).eval(); return m
    def gradcam_layers(m): return [m.blocks[-1][-1]]

model=load_vision()
if MTCNN_OK: mtcnn=MTCNN(keep_all=True,device=DEVICE,post_process=False); print("  MTCNN loaded")

# ══════════════════════════════════════════════════════════════
# AUDIO MODEL — AudioCNN for voice deepfake detection
# ══════════════════════════════════════════════════════════════
class AudioCNN(nn.Module):
    # Three-layer CNN for voice deepfake detection on MFCC spectrograms.
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(128*4*4,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,2))
    def forward(self,x): return self.fc(self.cnn(x))

# ── Load audio model with three-tier fallback ──
def load_audio():
    """Load audio detection model: AASIST3 → AudioCNN → heuristic fallback."""
    if AUDIO_BACKEND=="aasist":
        try:
            from model import aasist3
            m=aasist3.from_pretrained("MTUCI/AASIST3"); m.to(DEVICE).eval(); print("  AASIST3 loaded"); return m,"aasist3"
        except Exception as e:
            print(f"  AASIST failed: {e}")
            if os.path.exists(VOICE_PATH):
                m=AudioCNN(); m.load_state_dict(torch.load(VOICE_PATH,map_location=DEVICE,weights_only=False)); m.to(DEVICE).eval(); print("  AudioCNN loaded (fallback)"); return m,"audiocnn"
            return None,"heuristic"
    elif AUDIO_BACKEND=="audiocnn":
        try:
            m=AudioCNN(); m.load_state_dict(torch.load(VOICE_PATH,map_location=DEVICE,weights_only=False)); m.to(DEVICE).eval(); print(f"  AudioCNN loaded"); return m,"audiocnn"
        except: return None,"heuristic"
    return None,"heuristic"

voice_model,audio_model_name=load_audio()

 
# ══════════════════════════════════════════════════════════════
# IMAGE SUITABILITY PRE-SCREENING
# Rejects out-of-distribution inputs (screenshots, memes, flags)
# before running the deepfake model to prevent false positives.
# ══════════════════════════════════════════════════════════════
def assess_image_suitability(img_pil):
    #Check if image is suitable for deepfake detection. Rejects screenshots and memes
    arr = np.array(img_pil.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    img_area = h * w

    # --- 1. Detect faces and measure coverage ---
    face_ratio = 0.0
    face_box = None
    if MTCNN_OK:
        try:
            boxes, _ = mtcnn.detect(img_pil)
            if boxes is not None and len(boxes) > 0:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                best = int(np.argmax(areas))
                face_ratio = areas[best] / img_area
                face_box = boxes[best]
        except:
            pass

    # --- 2. Detect heavy text/graphics overlay via edge density ---
    # News images, flagged screenshots, memes all have sharp high-contrast edges
    # from text rendering. Natural photos have smoother edge distributions.
    gray = np.mean(arr, axis=2)
    # Sobel-like gradient magnitude
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad_mag = np.sqrt(gx**2 + gy**2)
    # Fraction of pixels with very sharp edges (>80 out of 255)
    sharp_edge_ratio = float(np.mean(grad_mag > 80))

    # --- 3. Detect flat/synthetic color regions (flags, graphics) ---
    # Real photos have continuous color variation; flags/graphics have large
    # uniform color blocks. Measure by std-dev of local color patches.
    patch_size = 16
    ph, pw = h // patch_size, w // patch_size
    flat_patches = 0
    total_patches = 0
    if ph > 0 and pw > 0:
        for i in range(ph):
            for j in range(pw):
                patch = arr[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                std = float(np.std(patch))
                total_patches += 1
                if std < 8.0:  # very uniform patch
                    flat_patches += 1
        flat_patch_ratio = flat_patches / max(total_patches, 1)
    else:
        flat_patch_ratio = 0.0

    # --- 4. Detect aspect-ratio letterboxing / screen captures ---
    # News screenshots often have black/white bars
    top_row_std = float(np.std(arr[:max(1, h//20), :]))
    bot_row_std = float(np.std(arr[min(h, h - h//20):, :]))
    has_letterbox = (top_row_std < 5.0 or bot_row_std < 5.0)

    # --- Decision logic ---
    # If there is a sufficiently large face, always proceed regardless of background
    if face_ratio >= 0.02:
        return True, "face_detected", face_ratio

    # Small/no face + heavy text overlay → likely news screenshot
    if sharp_edge_ratio > 0.20 and face_ratio < 0.02:
        return False, f"text_overlay_no_face (edge={sharp_edge_ratio:.2f})", face_ratio

    # Small/no face + mostly flat color blocks → likely flag or graphic
    if flat_patch_ratio > 0.60 and face_ratio < 0.02:
        return False, f"synthetic_graphic (flat={flat_patch_ratio:.2f})", face_ratio

    # Letterboxed + no meaningful face → screen capture / news clip
    if has_letterbox and face_ratio < 0.02:
        return False, "letterboxed_no_face", face_ratio

    # No face at all and image looks non-photographic → be very conservative
    if face_ratio == 0.0 and (sharp_edge_ratio > 0.15 or flat_patch_ratio > 0.50):
        return False, f"no_face_ood (edge={sharp_edge_ratio:.2f} flat={flat_patch_ratio:.2f})", face_ratio

    return True, "suitable", face_ratio


# ══════════════════════════════════════════════════════════════
# PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════
def strip_black_bars(img_pil, threshold=10):
    #Remove black letterbox bars from video frames
    arr = np.array(img_pil)
    h, w = arr.shape[:2]
    left = 0
    for col in range(w):
        if np.mean(arr[:, col, :]) < threshold: left = col + 1
        else: break
    right = w
    for col in range(w-1, -1, -1):
        if np.mean(arr[:, col, :]) < threshold: right = col
        else: break
    top = 0
    for row in range(h):
        if np.mean(arr[row, :, :]) < threshold: top = row + 1
        else: break
    bottom = h
    for row in range(h-1, -1, -1):
        if np.mean(arr[row, :, :]) < threshold: bottom = row
        else: break
    # Only strip if bars are meaningful (>1% of dimension) to avoid over-cropping
    if left > w*0.01 or right < w*0.99 or top > h*0.01 or bottom < h*0.99:
        return img_pil.crop((left, top, right, bottom))
    return img_pil

def correct_prior(fp, rp, prior_fake=0.55, prior_real=0.45):
    # Correct class imbalance bias using Bayesian prior (0.55:0.45).
    cf = (fp / prior_fake)
    cr = (rp / prior_real)
    total = cf + cr
    return cf / total, cr / total

def predict_image(img_pil):
    #Run CLIP inference on a single image with prior correction
    img_pil=strip_black_bars(img_pil)
    t=img_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits=model(t)[0] if VISION_BACKEND=="clip" else model(t)
        probs=torch.softmax(logits,dim=1)[0]
    rp,fp=probs[LABEL_REAL].item(),probs[LABEL_FAKE].item()
    fp,rp=correct_prior(fp,rp)
    return ("FAKE" if fp>rp else "REAL"),max(fp,rp),fp,rp

def predict_with_face(img_pil):
    #Dual-path prediction: whole image + face crop with dynamic thresholds
    FAKE_THRESHOLD = 0.62        # normal threshold when face present
    FAKE_THRESHOLD_SMALL = 0.70  # stricter when face is present but small
    FAKE_THRESHOLD_NO_FACE = 0.75  # very strict when no face detected

    # --- Pre-screen: is this image suitable for deepfake detection? ---
    suitable, reason, face_ratio = assess_image_suitability(img_pil)

    if not suitable:
        # Image is a graphic/news/flag — model is OOD here.
        # Return REAL with a note rather than running a misleading prediction.
        print(f"  [suitability] skipped deepfake model: {reason}")
        return ("REAL", 0.95, 0.05, 0.95), None

    # --- Run deepfake detection ---
    whole_result = predict_image(img_pil)
    whole_label, whole_conf, whole_fp, whole_rp = whole_result

    # Determine face box (already computed in assess_image_suitability, but re-detect here for crop)
    face = None
    if MTCNN_OK and face_ratio > 0.0:
        try:
            boxes, _ = mtcnn.detect(img_pil)
            if boxes is not None and len(boxes) > 0:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                best_idx = int(np.argmax(areas))
                box = boxes[best_idx]
                x1,y1,x2,y2 = [int(b) for b in box]; pad=40
                face = img_pil.crop((max(0,x1-pad), max(0,y1-pad),
                                     min(img_pil.width,x2+pad), min(img_pil.height,y2+pad)))
        except:
            pass

    if face is not None:
        face_result = predict_image(face)
        _, _, face_fp, face_rp = face_result
        # Weight face more heavily; reduce whole-image contribution to suppress overlay noise
        avg_fp = 0.20 * whole_fp + 0.80 * face_fp
        avg_rp = 1.0 - avg_fp
        # Smaller face → stricter threshold
        threshold = FAKE_THRESHOLD if face_ratio >= 0.08 else FAKE_THRESHOLD_SMALL
    else:
        # No face crop available — rely on whole image only with very strict threshold
        avg_fp = whole_fp
        avg_rp = whole_rp
        threshold = FAKE_THRESHOLD_NO_FACE

    if avg_fp > threshold:
        return ("FAKE", avg_fp, avg_fp, avg_rp), face
    return ("REAL", avg_rp, avg_fp, avg_rp), face

def gen_gradcam(img_pil):
    #Generate Grad-CAM heatmap. Reshapes ViT patch tokens to 2D grid.
    if not GRADCAM_OK: return None
    try:
        t=img_transform(img_pil).unsqueeze(0).to(DEVICE)
        rgb=np.array(img_pil.resize((IMG_SIZE,IMG_SIZE))).astype(np.float32)/255.0
        # Wrapper so GradCAM sees a single-tensor output instead of (logits, feat) tuple
        class _Wrap(nn.Module):
            def __init__(self,m): super().__init__(); self.m=m
            def forward(self,x): out=self.m(x); return out[0] if isinstance(out,tuple) else out
        wrap=_Wrap(model)
        tl = [wrap.m.vision_encoder.encoder.layers[-1].layer_norm1] if VISION_BACKEND=="clip" else gradcam_layers(wrap.m)
        # ViT outputs 3D (batch, seq_len, hidden) — reshape to 4D (batch, hidden, H, W) for GradCAM
        def vit_reshape(tensor):
            # Remove CLS token (first token), reshape remaining tokens to spatial grid
            # ViT-B/32 with 224px input: 7x7 = 49 patches + 1 CLS = 50 tokens
            result = tensor[:, 1:, :]  # remove CLS token
            h = w = int(result.shape[1] ** 0.5)  # sqrt(49) = 7
            result = result.reshape(result.shape[0], h, w, result.shape[2])
            result = result.permute(0, 3, 1, 2)  # (batch, hidden, H, W)
            return result
        rt = vit_reshape if VISION_BACKEND=="clip" else None
        with GradCAM(model=wrap, target_layers=tl, reshape_transform=rt) as cam:
            g=cam(input_tensor=t)[0]
        return Image.fromarray(show_cam_on_image(rgb,g,use_rgb=True))
    except Exception as e: print(f"GradCAM error: {e}"); return None

def pil_to_b64(img,fmt="JPEG"):
    buf=io.BytesIO(); img.save(buf,format=fmt); return base64.b64encode(buf.getvalue()).decode()

def predict_audio_file(path):
    #Analyse audio file for voice deepfake. Returns label, confidence and spectral features
    import librosa
    # MP3 files need ffmpeg via librosa; if that fails, convert with pydub as fallback
    try:
        y,sr=librosa.load(path,sr=16000,mono=True,duration=10.0)
    except Exception as load_err:
        # Try converting to wav first using pydub (which uses ffmpeg internally)
        try:
            from pydub import AudioSegment
            ext=os.path.splitext(path)[1].lower()
            audio=AudioSegment.from_file(path)
            wav_path=path+"_converted.wav"
            audio.set_frame_rate(16000).set_channels(1).export(wav_path,format="wav")
            y,sr=librosa.load(wav_path,sr=16000,mono=True,duration=10.0)
            os.unlink(wav_path)
        except Exception as conv_err:
            return {"error":f"Cannot read audio file. Install ffmpeg or try a WAV file. ({load_err})"}

    if len(y)<sr*0.5: return {"error":"Audio too short"}
    mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13); mfcc_var=float(np.mean(np.var(mfccs,axis=1)))
    flatness=float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr=float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    rolloff=float(np.mean(librosa.feature.spectral_rolloff(y=y,sr=sr)))
    rms=librosa.feature.rms(y=y)[0]; rms_var=float(np.var(rms))
    pitches,mags=librosa.piptrack(y=y,sr=sr); pv=pitches[mags>np.median(mags)]
    pitch_std=float(np.std(pv)) if len(pv)>0 else 0.0
    features=[{"name":"MFCC Variance","value":f"{mfcc_var:.1f}","suspicious":mfcc_var<15,"description":"Low variance may indicate AI-smoothed speech"},{"name":"Spectral Flatness","value":f"{flatness:.4f}","suspicious":flatness>0.05,"description":"High flatness suggests synthetic noise"},{"name":"Pitch Stability","value":f"{pitch_std:.1f} Hz","suspicious":pitch_std<30,"description":"Unnaturally stable pitch is a TTS marker"},{"name":"Energy Variance","value":f"{rms_var:.5f}","suspicious":rms_var<0.001,"description":"AI voices have overly consistent energy"},{"name":"Zero-Crossing Rate","value":f"{zcr:.4f}","suspicious":zcr<0.04,"description":"Low ZCR indicates waveform smoothing"},{"name":"Spectral Rolloff","value":f"{rolloff:.0f} Hz","suspicious":rolloff>6000,"description":"High rolloff may indicate upsampled audio"}]
    if audio_model_name=="aasist3" and voice_model:
        import torchaudio; at,osr=torchaudio.load(path)
        if osr!=16000: at=torchaudio.transforms.Resample(osr,16000)(at)
        if at.shape[0]>1: at=torch.mean(at,dim=0,keepdim=True)
        at=F.pad(at,(0,64600-at.shape[1])) if at.shape[1]<64600 else at[:,:64600]
        with torch.no_grad(): probs=torch.softmax(voice_model(at.to(DEVICE)),dim=1)[0]
        rp,fp=probs[0].item(),probs[1].item(); label="FAKE" if fp>rp else "REAL"; mu="AASIST3 (ASVspoof, pretrained)"
    elif audio_model_name=="audiocnn" and voice_model:
        yc=y[:48000] if len(y)>=48000 else np.pad(y,(0,48000-len(y)))
        m40=librosa.feature.mfcc(y=yc,sr=sr,n_mfcc=40); m40=(m40-m40.mean())/(m40.std()+1e-6)
        inp=torch.tensor(m40,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): probs=torch.softmax(voice_model(inp),dim=1)[0]
        rp,fp=probs[0].item(),probs[1].item(); label="FAKE" if fp>rp else "REAL"; mu="AudioCNN (FoR dataset)" 
    else:
        sc=0
        if mfcc_var<15:sc+=0.25
        if flatness>0.05:sc+=0.20
        if rms_var<0.001:sc+=0.20
        if pitch_std<30:sc+=0.20
        if zcr<0.04:sc+=0.15
        fp=min(sc,0.95); rp=1-fp; label="FAKE" if fp>0.5 else "REAL"; mu="Heuristic fallback"
    return {"label":label,"confidence":round(max(fp,rp)*100,1),"fake_prob":round(fp*100,1),"real_prob":round(rp*100,1),"features":features,"duration":round(len(y)/sr,1),"model_used":mu}

# ══════════════════════════════════════════════════════════════
# FLASK APPLICATION AND API ENDPOINTS
# ══════════════════════════════════════════════════════════════
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=500*1024*1024

@app.route("/")
def index(): return render_template("index.html")

@app.route("/predict/image",methods=["POST"])
def ri():
    #Image detection endpoint. Returns label, confidence and Grad-CAM
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    img=Image.open(request.files["file"]).convert("RGB"); result,bf=predict_with_face(img); label,conf,fp,rp=result
    cam=gen_gradcam(bf if bf else img)
    return jsonify({"label":label,"confidence":round(conf*100,1),"fake_prob":round(fp*100,1),"real_prob":round(rp*100,1),"image_b64":pil_to_b64(img),"gradcam_b64":pil_to_b64(cam) if cam else None,"model_backend":VISION_BACKEND})

@app.route("/predict/video",methods=["POST"])
def rv():
    #Video detection. Samples 16 frames and averages fake probabilities
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    f=request.files["file"]; suffix=os.path.splitext(f.filename)[1] or ".mp4"
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=suffix); f.save(tmp.name); tmp.close(); pp=tmp.name+"_preview.mp4"
    try:
        os.system(f'ffmpeg -i "{tmp.name}" -vcodec libx264 -acodec aac -y "{pp}" -loglevel quiet')
        rpath=pp if os.path.exists(pp) else tmp.name; cap=cv2.VideoCapture(rpath); tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot==0: return jsonify({"error":"Cannot read video"}),400
        n=16; idxs=[int(i*tot/n) for i in range(min(n,tot))]; fps_l=[]; fb64=[]
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ret,frame=cap.read()
            if not ret: continue
            pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); _,_,fp,_=predict_image(pil); fps_l.append(fp)
            if len(fb64)<8: fb64.append({"b64":pil_to_b64(pil.resize((160,120))),"label":"FAKE" if fp>0.5 else "REAL","fake_prob":round(fp*100,1)})
        cap.release(); af=float(np.mean(fps_l)) if fps_l else 0.5; label="FAKE" if af>0.5 else "REAL"
        vb64=None
        if os.path.exists(pp):
            with open(pp,"rb") as vf: vb64=base64.b64encode(vf.read()).decode()
        return jsonify({"label":label,"confidence":round((af if label=="FAKE" else 1-af)*100,1),"fake_prob":round(af*100,1),"real_prob":round((1-af)*100,1),"frames_analysed":len(fps_l),"fake_frames":sum(1 for p in fps_l if p>0.5),"real_frames":sum(1 for p in fps_l if p<=0.5),"sample_frames":fb64,"video_b64":vb64,"model_backend":VISION_BACKEND})
    finally:
        os.unlink(tmp.name)
        if os.path.exists(pp): os.unlink(pp)

@app.route("/predict/audio",methods=["POST"])
def ra():
    #Audio detection endpoint
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    f=request.files["file"]; tmp=tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(f.filename)[1] or ".wav"); f.save(tmp.name); tmp.close()
    try:
        r=predict_audio_file(tmp.name); return (jsonify(r),400) if "error" in r else jsonify(r)
    except Exception as e: return jsonify({"error":str(e)}),500
    finally: os.unlink(tmp.name)

@app.route("/predict/forgery",methods=["POST"])
def rf():
    #ELA forgery detection. Re-saves at JPEG quality 75 and scores the difference
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    try:
        img=Image.open(io.BytesIO(request.files["file"].read())).convert("RGB"); ia=np.array(img)
        buf=io.BytesIO(); img.save(buf,format="JPEG",quality=75); buf.seek(0); ca=np.array(Image.open(buf).convert("RGB"))
        ela=np.abs(ia.astype(np.float32)-ca.astype(np.float32)); em,ex,es=float(ela.mean()),float(ela.max()),float(ela.std())
        gray=np.array(img.convert("L")).astype(np.float32); ns=float((gray-np.roll(gray,1,axis=0)).std())
        r,g,b=ia[:,:,0],ia[:,:,1],ia[:,:,2]; rgc=float(np.corrcoef(r.flatten(),g.flatten())[0,1]); rbc=float(np.corrcoef(r.flatten(),b.flatten())[0,1])
        sc=0
        if em>8:sc+=30
        elif em>5:sc+=15
        if es>15:sc+=25
        elif es>10:sc+=12
        if ns<3:sc+=25
        elif ns<5:sc+=12
        if rgc>0.98:sc+=20
        elif rgc>0.95:sc+=10
        sc=min(sc,100); label="TAMPERED" if sc>=40 else "AUTHENTIC"
        return jsonify({"label":label,"confidence":round(sc if label=="TAMPERED" else 100-sc,1),"features":{"ELA Mean":round(em,2),"ELA Std":round(es,2),"ELA Max":round(ex,2),"Noise Std":round(ns,2),"RG Correlation":round(rgc,4),"RB Correlation":round(rbc,4)}})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/generate_pdf",methods=["POST"])
def rpdf():
    #Generate PDF detection report
    if not PDF_OK: return jsonify({"error":"reportlab not installed"}),500
    try:
        d=request.get_json()
        if not d: return jsonify({"error":"No JSON data received"}),400
        label=d.get("label","UNKNOWN"); conf=float(d.get("confidence",0)); fp=float(d.get("fake_prob",0)); rp=float(d.get("real_prob",0))
        ib,cb=d.get("image_b64","") or "",d.get("gradcam_b64","") or ""; fn=d.get("filename","upload")
        media_type=d.get("media_type","image")
        mi="CLIP ViT-B/32 + LN-tuning" if VISION_BACKEND=="clip" else "EfficientNet-B0"
        buf=io.BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
        ts=ParagraphStyle("t",fontSize=20,fontName="Helvetica-Bold",textColor=colors.HexColor("#0a1e50"),spaceAfter=20)
        ss=ParagraphStyle("s",fontSize=11,fontName="Helvetica",textColor=colors.HexColor("#4a6fa8"),spaceAfter=12)
        ds=ParagraphStyle("d",fontSize=8,fontName="Helvetica-Oblique",textColor=colors.HexColor("#64748b"),leading=12)
        ns=ParagraphStyle("n",fontSize=10,fontName="Helvetica",textColor=colors.HexColor("#334155"),spaceAfter=4)
        hs=ParagraphStyle("h2",fontSize=12,fontName="Helvetica-Bold",textColor=colors.HexColor("#0a1e50"),spaceAfter=8)

        type_label={"image":"Image","video":"Video","voice":"Voice/Audio"}.get(media_type,"Image")
        story=[Paragraph("DFLens v3 \u2014 Detection Report",ts),Paragraph(f"AI-Powered Deepfake Detection \u00b7 {type_label} Analysis \u00b7 {mi}",ss),Spacer(1,0.3*cm)]

        rc=colors.HexColor("#16a34a") if label=="REAL" else colors.HexColor("#dc2626")
        verdict="AUTHENTIC" if label=="REAL" else "DEEPFAKE DETECTED"

        # ── Analysis Result (verdict + conclusion) ──
        vs=ParagraphStyle("vs",fontSize=16,fontName="Helvetica-Bold",textColor=rc,alignment=1,spaceAfter=6)
        cs=ParagraphStyle("cs",fontSize=10,fontName="Helvetica",textColor=colors.HexColor("#334155"),alignment=1,spaceAfter=12,leading=14)
        story.append(Paragraph(verdict,vs))
        conclusion=d.get("conclusion","")
        if conclusion: story.append(Paragraph(conclusion,cs))
        story.append(Spacer(1,0.3*cm))

        # ── Probability bars as table ──
        prob_data=[["","Probability",""],["REAL",f"{rp:.1f}%",""],["FAKE",f"{fp:.1f}%",""]]
        pt=Table(prob_data,colWidths=[3*cm,4*cm,10*cm])
        real_bar_color=colors.HexColor("#16a34a"); fake_bar_color=colors.HexColor("#dc2626")
        pt.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),10),("TEXTCOLOR",(0,1),(0,1),real_bar_color),("TEXTCOLOR",(0,2),(0,2),fake_bar_color),("TEXTCOLOR",(1,1),(1,1),real_bar_color),("TEXTCOLOR",(1,2),(1,2),fake_bar_color),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f0f4ff")),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(pt); story.append(Spacer(1,0.4*cm))

        # ── Image: Face Pixels + Whole Image analysis ──
        if media_type=="image":
            fl=d.get("forgery_label",""); fc=float(d.get("forgery_confidence",0) or 0)
            face_status="FAKE" if label=="FAKE" else "REAL"
            forgery_status="TAMPERED" if fl=="TAMPERED" else "CLEAN"
            forgery_pct=fc if fl=="TAMPERED" else (100-fc)
            analysis_data=[
                ["Analysis","Result","Probability"],
                ["Face Pixels (AI face-swap?)",face_status,f"{fp:.1f}% Fake / {rp:.1f}% Real"],
                ["Whole Image (edited/tampered?)",forgery_status,f"{forgery_pct:.1f}% {forgery_status}"]
            ]
            at=Table(analysis_data,colWidths=[6*cm,3*cm,8*cm])
            face_color=colors.HexColor("#dc2626") if label=="FAKE" else colors.HexColor("#16a34a")
            forg_color=colors.HexColor("#dc2626") if fl=="TAMPERED" else colors.HexColor("#16a34a")
            at.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1B3A5C")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),("TEXTCOLOR",(1,1),(1,1),face_color),("TEXTCOLOR",(1,2),(1,2),forg_color),("FONTNAME",(1,1),(1,2),"Helvetica-Bold"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
            story.append(Paragraph("Analysis Breakdown",hs))
            story.append(at); story.append(Spacer(1,0.4*cm))

        # ── Detail info table ──
        story.append(Paragraph("Detection Details",hs))
        td=[["File",fn],["Media Type",type_label],["Model",mi],["Device",str(DEVICE).upper()]]

        # Add video-specific rows
        if media_type=="video":
            fa=d.get("frames_analysed",0); ff=d.get("fake_frames",0); rf=d.get("real_frames",0)
            td.append(["Frames Analysed",str(fa)])
            td.append(["Real Frames",str(rf)])
            td.append(["Fake Frames",str(ff)])

        # Add voice-specific rows
        if media_type=="voice":
            dur=d.get("duration",0); mu=d.get("model_used","")
            td.append(["Duration",f"{dur}s"])
            td.append(["Audio Model",mu])

        t=Table(td,colWidths=[5*cm,12*cm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#f0f4ff")),("TEXTCOLOR",(1,2),(1,2),rc),("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.append(t); story.append(Spacer(1,0.5*cm))

        # Image: add original + Grad-CAM
        def b64i(b,w=8*cm):
            if not b: return None
            try: return RLImage(io.BytesIO(base64.b64decode(b)),width=w,height=w*0.75)
            except: return None
        if media_type=="image":
            oi,ci=b64i(ib),b64i(cb)
            if oi or ci:
                story.append(Paragraph("Analysed Image",hs))
                rw,lr=[],[]
                if oi:rw.append(oi);lr.append(Paragraph("Original",ns))
                if ci:rw.append(ci);lr.append(Paragraph("Grad-CAM",ns))
                it=Table([rw,lr],colWidths=[8.5*cm]*len(rw)); it.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"TOP")]))
                story.append(it); story.append(Spacer(1,0.3*cm))

        # Video: add sample frames grid
        if media_type=="video" and d.get("sample_frames"):
            story.append(Paragraph("Sampled Frames Analysis",hs))
            frames = d["sample_frames"]
            # Build rows of 4 frames each
            row_imgs = []
            row_labels = []
            for i, sf in enumerate(frames):
                try:
                    fimg = RLImage(io.BytesIO(base64.b64decode(sf["b64"])), width=3.8*cm, height=2.85*cm)
                    row_imgs.append(fimg)
                    fl = sf.get("label","?")
                    fp_val = sf.get("fake_prob", 0)
                    rp_val = sf.get("real_prob", 100-fp_val) if fl=="REAL" else 100-fp_val
                    display_pct = rp_val if fl=="REAL" else fp_val
                    lbl_text = f"{fl} {display_pct}%"
                    lbl_color = "#16a34a" if fl=="REAL" else "#dc2626"
                    row_labels.append(Paragraph(f'<font color="{lbl_color}"><b>{lbl_text}</b></font>', ParagraphStyle("fl",fontSize=8,fontName="Helvetica",alignment=1)))
                except:
                    continue
                # Every 4 frames, flush a row
                if len(row_imgs) == 4 or i == len(frames)-1:
                    # Pad if less than 4
                    while len(row_imgs) < 4:
                        row_imgs.append(Paragraph("",ns))
                        row_labels.append(Paragraph("",ns))
                    ft = Table([row_imgs, row_labels], colWidths=[4.25*cm]*4)
                    ft.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
                    story.append(ft); story.append(Spacer(1,0.2*cm))
                    row_imgs = []
                    row_labels = []

        # Voice: add feature analysis table
        if media_type=="voice" and d.get("features"):
            story.append(Paragraph("Audio Feature Analysis",hs))
            feat_data=[["Feature","Value","Status"]]
            for f in d["features"]:
                status="Suspicious" if f.get("suspicious") else "Normal"
                feat_data.append([f.get("name",""),str(f.get("value","")),status])
            ft=Table(feat_data,colWidths=[5.5*cm,4*cm,3.5*cm])
            ft_style=[("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1B3A5C")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]
            for i,f in enumerate(d["features"],1):
                if f.get("suspicious"):
                    ft_style.append(("TEXTCOLOR",(2,i),(2,i),colors.HexColor("#dc2626")))
                else:
                    ft_style.append(("TEXTCOLOR",(2,i),(2,i),colors.HexColor("#16a34a")))
            ft.setStyle(TableStyle(ft_style))
            story.append(ft); story.append(Spacer(1,0.3*cm))

        story.append(Spacer(1,0.5*cm)); story.append(Paragraph("DISCLAIMER: AI-generated report for educational purposes only. DFLens v3 \u2014 INT4203E.",ds))
        doc.build(story); buf.seek(0)
        return send_file(buf,mimetype="application/pdf",as_attachment=True,download_name=f"dflens_report_{media_type}_{label.lower()}.pdf")
    except Exception as e:
        print(f"PDF generation error: {e}")
        return jsonify({"error":str(e)}),500

# ── Video upload and streaming endpoints ──
import uuid, threading

# In-memory store for uploaded preview videos {token: (path, timer)}
_video_store = {}
_video_lock  = threading.Lock()

def _expire_video(token):
    #Auto-delete uploaded video after 5 minutes to free disk space.
    with _video_lock:
        entry = _video_store.pop(token, None)
    if entry:
        try: os.unlink(entry["path"])
        except: pass

@app.route("/upload/video", methods=["POST"])
def ruv():
    #Upload and transcode video for browser playback
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    f = request.files["file"]
    suffix = os.path.splitext(f.filename)[1] or ".mp4"
    # Save original
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name); tmp.close()

    # Extract a poster frame from the original video before transcoding
    poster_b64 = None
    try:
        cap = cv2.VideoCapture(tmp.name)
        if cap.isOpened():
            # Try a few positions to avoid a black frame
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for pos in [0, min(5, total-1), min(15, total-1)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret and float(np.mean(frame)) > 10:  # not a black frame
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    poster_b64 = base64.b64encode(buf.tobytes()).decode()
                    break
        cap.release()
    except:
        pass

    # Transcode to H.264 MP4 for guaranteed Chrome compatibility
    out_path = tmp.name + "_preview.mp4"
    ret = os.system(f'ffmpeg -i "{tmp.name}" -vcodec libx264 -acodec aac -movflags faststart -preset ultrafast -crf 28 -y "{out_path}" -loglevel quiet')
    if ret != 0 or not os.path.exists(out_path):
        # ffmpeg failed — serve original file as-is
        out_path = tmp.name
    else:
        # Transcode succeeded — remove original
        try: os.unlink(tmp.name)
        except: pass
    token = str(uuid.uuid4())
    timer = threading.Timer(300, _expire_video, args=[token])
    timer.daemon = True; timer.start()
    with _video_lock:
        _video_store[token] = {"path": out_path, "timer": timer}
    return jsonify({"token": token, "poster_b64": poster_b64})

@app.route("/stream/video/<token>")
def rsv(token):
    #Stream uploaded video with Range support
    from flask import Response, stream_with_context
    with _video_lock:
        entry = _video_store.get(token)
    if not entry: return jsonify({"error":"Not found"}),404
    path = entry["path"]
    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range")
    suffix = os.path.splitext(path)[1].lower()
    # Transcoded files always end with _preview.mp4, force correct mime
    mime = "video/mp4" if "_preview.mp4" in path or suffix in (".mp4",".m4v") else "video/x-msvideo" if suffix==".avi" else "video/quicktime" if suffix==".mov" else "video/mp4"
    if range_header:
        # Partial content for seeking support
        byte_start, byte_end = 0, file_size - 1
        m = __import__("re").search(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            byte_start = int(m.group(1))
            byte_end   = int(m.group(2)) if m.group(2) else file_size - 1
        length = byte_end - byte_start + 1
        def generate():
            with open(path, "rb") as fv:
                fv.seek(byte_start)
                remaining = length
                while remaining > 0:
                    chunk = fv.read(min(65536, remaining))
                    if not chunk: break
                    remaining -= len(chunk)
                    yield chunk
        resp = Response(stream_with_context(generate()), 206, mimetype=mime,
                        direct_passthrough=True)
        resp.headers["Content-Range"]  = f"bytes {byte_start}-{byte_end}/{file_size}"
        resp.headers["Accept-Ranges"]  = "bytes"
        resp.headers["Content-Length"] = str(length)
        return resp
    # Full file
    return send_file(path, mimetype=mime, conditional=True)

@app.route("/preview/video",methods=["POST"])
def rpv():
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    f=request.files["file"]; tmp=tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(f.filename)[1] or ".mp4"); f.save(tmp.name); tmp.close(); pp=tmp.name+"_preview.mp4"
    try:
        os.system(f'ffmpeg -i "{tmp.name}" -vcodec libx264 -acodec aac -movflags faststart -y "{pp}" -loglevel quiet')
        if os.path.exists(pp):
            with open(pp,"rb") as vf: return jsonify({"video_b64":base64.b64encode(vf.read()).decode()})
        return jsonify({"error":"Transcode failed"}),500
    finally:
        os.unlink(tmp.name)
        if os.path.exists(pp): os.unlink(pp)

@app.route("/model_info")
def rmi():
    #Return backend diagnostics 
    return jsonify({"vision_backend":VISION_BACKEND,"vision_model":"CLIP ViT-B/32 + LN-tuning" if VISION_BACKEND=="clip" else "EfficientNet-B0","audio_model":audio_model_name,"device":str(DEVICE),"gradcam":GRADCAM_OK,"face_detection":MTCNN_OK})

if __name__=="__main__":
    os.makedirs("templates",exist_ok=True); print(f"\nDFLens v3 | vision={VISION_BACKEND} audio={audio_model_name} device={DEVICE}\nhttp://localhost:5000\n"); app.run(debug=False,port=5000)