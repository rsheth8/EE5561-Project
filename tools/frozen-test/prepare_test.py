"""Safely extract official test ZIP; audit overlap and freeze INPUT-selected crops."""
from pathlib import Path,PurePosixPath
import argparse,json,os,stat,zipfile,zlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from common import verify_code,PROTOCOL,HERE,read_scene,sha,thumbnail,choose_crop,write_json,write_csv

def extract_archive(path,root):
    if path.stat().st_size!=PROTOCOL['test_archive_bytes']:raise ValueError('Archive size differs from verified official index; download may be incomplete')
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'ZIP checksum failure: {bad}')
        for entry in z.infolist():
            rel=PurePosixPath(entry.filename)
            if rel.is_absolute() or '..' in rel.parts or rel.parts[0]!='Test' or stat.S_ISLNK(entry.external_attr>>16):raise ValueError(f'Unsafe ZIP member: {entry.filename}')
            target=root/str(rel)
            if entry.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                crc=0
                with target.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):crc=zlib.crc32(chunk,crc)
                if target.stat().st_size!=entry.file_size or crc!=entry.CRC:raise FileExistsError(f'Existing extracted file differs; no overwrite: {target}')
            else:
                with z.open(entry) as source,target.open('xb') as out:
                    while chunk:=source.read(1024*1024):out.write(chunk)

def main():
    verify_code()
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True,type=Path);ap.add_argument('--raw',required=True,type=Path);ap.add_argument('--train',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(f'Use a fresh output directory: {a.out}')
    a.out.mkdir(parents=True);write_json(a.out/'status.json',{'status':'preparing, not evaluated'})
    extract_archive(a.archive,a.raw)
    manual=json.loads((HERE/'input_illustration_crops.json').read_text())
    root=a.raw/'Test';actual=sorted(str(p.parent.relative_to(root)) for p in root.rglob('exposure.txt'))
    if actual!=PROTOCOL['test_scene_keys']:raise ValueError(f'Test set differs: {actual}')
    split=json.loads((HERE/'training_split_frozen.json').read_text());ids=sorted(split['train']+split['val'])
    train_hashes={};train_thumbs=[]
    for sid in ids:
        folder=a.train/sid;files=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.tif','.tiff'))
        if len(files)!=3:raise ValueError(f'Missing training scene {sid}')
        import cv2
        ref=cv2.imread(str(files[1]),cv2.IMREAD_UNCHANGED)
        train_thumbs.append(thumbnail(ref[...,::-1].astype(np.float32)/65535))
        for p in files:train_hashes.setdefault(sha(p),[]).append(f'Training/{sid}/{p.name}')
    train_thumbs=np.stack(train_thumbs)
    rows=[];scenes=[];duplicates=[];similar=[]
    fig,axes=plt.subplots(5,3,figsize=(12,13))
    for ax,key in zip(axes.flat,actual):
        folder=root/key;s=read_scene(folder);crop=choose_crop(s['ldr'],s['t'])
        if key in manual:
            auto=crop;crop={'xywh':manual[key]['xywh'],'rule':manual[key]['reason'],'automatic_alternative':auto}
            x,y,w,h=crop['xywh']
            if min(x,y)<0 or min(w,h)<=0 or x+w>s['shape'][1] or y+h>s['shape'][0]:raise ValueError('Illustration crop outside image')
        scores=train_thumbs@thumbnail(s['ldr'][1])/train_thumbs.shape[1]
        k=int(np.argmax(scores));ncc=float(scores[k])
        if ncc>=.9:similar.append({'test_scene':key,'training_scene':ids[k],'ncc':ncc})
        hashes={str(p.relative_to(root)):sha(p) for p in s['files']+[s['exposure_file']]}
        for p in s['files']:
            if sha(p) in train_hashes:duplicates.append({'test_file':str(p),'training_files':train_hashes[sha(p)]})
        scenes.append({'scene_key':key,'shape':list(s['shape']),'times':s['t'].tolist(),'input_hashes':hashes,'crop':crop})
        rows.append({'scene':key,'height':s['shape'][0],'width':s['shape'][1],'max_training_ncc':ncc,'nearest_training_scene':ids[k],'similarity_review_needed':ncc>=.9,'pairing_pixels':s['pairing_pixels']})
        ax.imshow(s['ldr'][1]);x,y,w,h=crop['xywh'];ax.add_patch(Rectangle((x,y),w,h,fill=False,color='red',linewidth=1.5));ax.set_title(key,fontsize=10);ax.axis('off')
        print(f'Prepared {key}: crop {crop["xywh"]}, nearest Training/{ids[k]} NCC={ncc:.3f}',flush=True)
    fig.suptitle('All 15 test reference inputs, before model evaluation\nRed boxes: input-only illustration patches; not ground-truth motion',fontsize=13)
    fig.tight_layout(rect=(0,.01,1,.95));fig.savefig(a.out/'all_test_inputs.png',dpi=140);plt.close(fig)
    manifest={'status':'ready' if not duplicates and not similar else 'review_required','protocol_sha256':sha(HERE/'test_protocol_frozen.json'),'archive_sha256':sha(a.archive),'archive':str(a.archive.resolve()),'test_root':str(root.resolve()),'train_root':str(a.train.resolve()),'all_training_ids_checked':ids,'exact_input_overlaps':duplicates,'similarity_flags':similar,'scenes':scenes,'figure_selection':'Fixed from input images before evaluator sees predictions or target previews. PeopleTalking uses the documented input-only human-motion illustration; others use the automatic texture/disagreement proxy.','illustration_crops_sha256':sha(HERE/'input_illustration_crops.json')}
    write_json(a.out/'input_manifest.json',manifest);write_csv(a.out/'overlap_audit.csv',rows);write_json(a.out/'status.json',{'status':manifest['status'],'exact_overlaps':len(duplicates),'similarity_flags':len(similar),'scenes':len(scenes)})
    if duplicates:raise RuntimeError('Exact train/test input overlap detected. See input_manifest.json before evaluating.')
    print(f'Input preparation complete: {a.out}',flush=True)
    if similar:print('Similarity flags require review. Do not claim leakage-free test results until reviewed.',flush=True)
if __name__=='__main__':main()
