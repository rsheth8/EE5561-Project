"""Read-only scene IO and display helpers for frozen evaluation, AI-assisted starter."""
from pathlib import Path
import csv,hashlib,json
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
PROTOCOL=json.loads((HERE/'test_protocol_frozen.json').read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def read_scene(folder):
    folder=Path(folder)
    files=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.tif','.tiff'))
    e=np.loadtxt(folder/'exposure.txt').reshape(-1)
    if len(files)!=3 or len(e)!=3 or not np.isfinite(e).all() or not np.all(np.diff(e)>0):raise ValueError(f'Bad exposures: {folder}')
    raw=[]
    for p in files:
        x=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
        if x is None or x.dtype!=np.uint16 or x.ndim!=3 or x.shape[-1]!=3:raise ValueError(f'Bad uint16 RGB input: {p}')
        raw.append(np.ascontiguousarray(x[...,::-1]))
    if len({x.shape for x in raw})!=1:raise ValueError('Input shapes disagree')
    ldr=np.stack(raw).astype(np.float32)/np.float32(65535)
    times=np.exp2(e).astype(np.float32)
    if not np.all(np.isfinite(times)&(times>0)):raise ValueError('Invalid exposure times')
    ok=np.all((ldr>=.05)&(ldr<=.95),axis=(0,3))
    means=[float(x[ok].mean()) if ok.any() else None for x in ldr]
    if ok.any() and not np.all(np.diff(means)>0):raise ValueError(f'Exposure ordering check failed: {folder}')
    return {'ldr':ldr,'t':times,'files':files,'exposure_file':folder/'exposure.txt','shape':ldr.shape[1:],'pairing_pixels':int(ok.sum()),'pairing_means':means}

def read_target(folder,shape):
    p=Path(folder)/'HDRImg.hdr';x=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
    if x is None:raise ValueError(f'Missing target: {p}')
    x=np.ascontiguousarray(x[...,::-1]).astype(np.float32)
    if x.shape!=shape or not np.isfinite(x).all() or x.min()<0 or x.max()>1.000001:raise ValueError(f'Target shape/range differs from frozen convention: {p}')
    return x

def thumbnail(ldr):
    g=cv2.cvtColor(ldr.astype(np.float32),cv2.COLOR_RGB2GRAY)
    g=cv2.resize(g,(72,48),interpolation=cv2.INTER_AREA).astype(np.float64)
    return ((g-g.mean())/(g.std()+1e-12)).ravel()

def choose_crop(ldr,t,size=256):
    """Inputs only: disagreement plus texture, not output error or a true motion mask."""
    h,w=ldr.shape[1:3];size=min(size,h,w)
    rad=ldr**2.2/t[:,None,None,None]
    usable=np.all((ldr[1]>.05)&(ldr[1]<.95),axis=-1)
    diff=np.max(np.abs(rad[[0,2]]-rad[1])/(rad[[0,2]]+rad[1]+1e-6),axis=(0,3))
    gray=cv2.cvtColor(ldr[1],cv2.COLOR_RGB2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    texture=np.hypot(gx,gy)
    scores=diff*np.minimum(texture,.2)*usable
    candidates=[]
    for y in sorted(set(range(0,h-size+1,32))|{h-size}):
        for x in sorted(set(range(0,w-size+1,32))|{w-size}):
            sl=np.s_[y:y+size,x:x+size]
            candidates.append((float(scores[sl].mean()),x,y,float(texture[sl].mean()),float(usable[sl].mean())))
    best=max(candidates,key=lambda a:a[0])
    return {'xywh':[best[1],best[2],size,size],'input_proxy_score':best[0],'mean_texture':best[3],'usable_fraction':best[4],
            'rule':'Highest mean input radiance disagreement * capped reference gradient * reference usable mask, 32px grid; ties top-left. Not ground-truth motion.'}

def display_exposure(reference_radiance):
    lum=reference_radiance.astype(np.float64)@np.array([.2126,.7152,.0722])
    return float(np.exp(np.log(np.maximum(lum,1e-6)).mean()))

def display(h,log_average):
    h=np.array(h,dtype=np.float64,copy=True);scaled=.18*h/log_average
    lum=scaled@np.array([.2126,.7152,.0722])
    return np.clip(scaled/(1+lum[...,None]),0,1)**(1/2.2)


def verify_code():
    manifest=json.loads((HERE/'helper_sha256.json').read_text())
    for name,digest in manifest.items():
        if sha(HERE/name)!=digest:raise RuntimeError(f'Helper file changed after package verification: {name}')
