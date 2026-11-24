"""Evaluate every official test scene with frozen checkpoints and preprocessing.

No training, checkpoint selection, target fitting, resizing or method tuning.
AI-assisted evaluation starter; students must review code and results.
"""
from pathlib import Path
import argparse,csv,json,os,platform,sys,tarfile,time,traceback
import cv2
import numpy as np
import torch
import skimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import verify_code,HERE,PROTOCOL,sha,read_scene,read_target,write_json,write_csv,display,display_exposure
from frozen_hdr.alignment import align_scene
from frozen_hdr.model import build_model,model_forward
from frozen_hdr.metrics import all_metrics,mu_law
from classical_rules import merge,radiance

METHODS=['C0','C1','C2','C3','M0','M1','M2','M3']

def load_model(variant,root,device):
    entry=PROTOCOL['checkpoints'][variant];path=Path(root)/entry['path']
    if sha(path)!=entry['sha256']:raise ValueError(f'Checkpoint differs from frozen validation-selected weights: {path}')
    from numpy.core.multiarray import _reconstruct
    allowed=[_reconstruct,(_reconstruct,'numpy._core.multiarray._reconstruct'),np.ndarray,np.dtype,type(np.dtype(np.uint32))]
    with torch.serialization.safe_globals(allowed):ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['step']!=50000 or ck['config']['variant']!=variant:raise ValueError('Checkpoint variant/step mismatch')
    model=build_model(variant).eval();model.load_state_dict(ck['model'],strict=True)
    if not all(torch.isfinite(x).all() for x in model.state_dict().values()):raise ValueError('Nonfinite weights')
    return model.to(device)

def build_batch(s,alignment,device):
    # Same CHW layout and FP32 feature construction as hdr.data.to_batch.
    ldr=torch.from_numpy(np.ascontiguousarray(s['ldr'].transpose(0,3,1,2))[None]).to(device)
    t=torch.from_numpy(s['t'][None]).to(device).view(1,3,1,1,1)
    feat=torch.cat([ldr,ldr.clamp(0,1).pow(2.2)/t],dim=2)
    lw=torch.from_numpy(np.ascontiguousarray(alignment['Lw'].transpose(0,3,1,2))[None]).to(device)
    ldrw=torch.stack([lw[:,0],ldr[:,1],lw[:,1]],dim=1)
    featw=torch.cat([ldrw,ldrw.clamp(0,1).pow(2.2)/t],dim=2)
    cues=np.stack([np.stack([alignment['q_ref'],alignment['q_w'][k],alignment['v'][k].astype(np.float32),alignment['r'][k]]) for k in range(2)]).astype(np.float32)
    cues=torch.from_numpy(np.ascontiguousarray(cues)[None]).to(device)
    return {'feat':feat,'featw':featw,'cues':cues,'valid':cues[:,:,2:3]}

def classical(s,a):
    L=s['ldr'];t=[float(x) for x in s['t']];ones=np.ones(L.shape[1:3],np.float32);cfg=PROTOCOL['classical_constants']
    return {'C0':radiance(L[1],t[1]).astype(np.float32),
      'C1':merge(list(L),t,[ones,ones,ones],cfg,False)[0],
      'C2':merge(list(L),t,[ones,ones,ones],cfg,True)[0],
      'C3':merge([a['Lw'][0],L[1],a['Lw'][1]],t,[a['v'][0],ones,a['v'][1]],cfg,True)[0]}

def draw_case(folder,key,s,pred,target,crop):
    folder.mkdir(parents=True,exist_ok=True);x,y,w,h=crop;sl=np.s_[y:y+h,x:x+w]
    avg=display_exposure(pred['C0'])
    fig,axes=plt.subplots(2,4,figsize=(12,7))
    axes[0,0].imshow(s['ldr'][1]);axes[0,0].add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='red',linewidth=1.5));axes[0,0].set_title('Full reference input + crop box')
    for ax,name,label in zip(axes[0,1:],['C1','C3','TARGET'],['Weighted merge','Aligned rejection','HDR target']):
        image=target if name=='TARGET' else pred[name]
        ax.imshow(display(image[sl],avg));ax.set_title(label)
    for ax,name in zip(axes[1],['M0','M1','M2','M3']):
        ax.imshow(display(pred[name][sl],avg));ax.set_title({'M0':'M0: original inputs','M1':'M1: aligned inputs','M2':'M2: constant cues','M3':'M3: measured cues'}[name])
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(key+' | same input-selected crop across all methods',fontsize=14)
    fig.text(.5,.015,'Outputs and target share ONE reference-derived display. Locator uses stored input encoding. No per-method enhancement.\nScores use full float images; a crop is an illustration, not the test score.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.02,right=.98,top=.87,bottom=.13,wspace=.12,hspace=.4);fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    gt=mu_law(target);e2=np.abs(mu_law(pred['M2'])-gt).mean(-1);e3=np.abs(mu_law(pred['M3'])-gt).mean(-1)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,error,label in zip(axes[:2],[e2,e3],['M2 absolute mu error','M3 absolute mu error']):
        im=ax.imshow(error[sl],vmin=0,vmax=.05,cmap='inferno');fig.colorbar(im,ax=ax,fraction=.046);ax.set_title(label)
    im=axes[2].imshow((e3-e2)[sl],vmin=-.02,vmax=.02,cmap='coolwarm');fig.colorbar(im,ax=axes[2],fraction=.046);axes[2].set_title('M3 minus M2 error\nBlue: M3 lower; red: higher')
    for ax in axes:ax.axis('off')
    fig.suptitle(key);fig.text(.5,.015,'Fixed shared scales. Values beyond the color limits are display-saturated; all errors still enter numeric scores.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.92));fig.savefig(folder/'errors.png',dpi=150);plt.close(fig)

def summarize(rows):
    result=[]
    for group,selected in [('ALL15',rows),('PAPER',[r for r in rows if r['scene'].startswith('PAPER/')]),('EXTRA',[r for r in rows if r['scene'].startswith('EXTRA/')])]:
        for method in METHODS:
            rs=[r for r in selected if r['method']==method]
            result.append({'group':group,'method':method,'n_scenes':len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in ['psnr_mu','psnr_l','ssim_mu','ssim_l','mae_mu']}})
    return result

def main():
    verify_code()
    ap=argparse.ArgumentParser();ap.add_argument('--input-manifest',type=Path,required=True);ap.add_argument('--checkpoints',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(f'Use a fresh output directory: {args.out}')
    manifest=json.loads(args.input_manifest.read_text())
    if manifest['protocol_sha256']!=sha(HERE/'test_protocol_frozen.json'):raise ValueError('Protocol changed after input/crop preparation')
    if manifest.get('illustration_crops_sha256')!=sha(HERE/'input_illustration_crops.json'):raise ValueError('Illustration crops changed after preparation')
    if manifest['exact_input_overlaps']:raise RuntimeError('Exact training/test input overlap needs resolution')
    if manifest['similarity_flags']:raise RuntimeError('Input similarity flags need human review. Share input_manifest.json before proceeding; do not remove test scenes silently.')
    if [s['scene_key'] for s in manifest['scenes']]!=PROTOCOL['test_scene_keys']:raise ValueError('Incomplete test manifest')
    if not torch.cuda.is_available():raise RuntimeError('Start an MSI GPU allocation, then run again. No training is needed.')
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run inside an allocated MSI Slurm GPU session, not a login node.')
    args.out.mkdir(parents=True);write_json(args.out/'status.json',{'status':'incomplete','completed_scenes':0})
    torch.manual_seed(5561);torch.cuda.manual_seed_all(5561);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;cv2.setNumThreads(1)
    device=torch.device('cuda:0');start=time.time();rows=[];completed=[]
    runtime={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'opencv':cv2.__version__,'numpy':np.__version__,'skimage':skimage.__version__,'gpu':torch.cuda.get_device_name(0),'slurm_job_id':os.environ['SLURM_JOB_ID'],'protocol_sha256':sha(HERE/'test_protocol_frozen.json'),'input_manifest_sha256':sha(args.input_manifest),'model_selection':'Frozen before test evaluation; no optimizer constructed','flow_version_limit':'Alignment implementation is frozen. OpenCV version is recorded; compare it with training environment before claiming exact cross-environment replication.'}
    write_json(args.out/'runtime.json',runtime)
    try:
        models={v:load_model(v,args.checkpoints,device) for v in ['M0','M1','M2','M3']}
        for entry in manifest['scenes']:
            key=entry['scene_key'];folder=Path(manifest['test_root'])/key
            for rel,digest in entry['input_hashes'].items():
                if sha(Path(manifest['test_root'])/rel)!=digest:raise ValueError(f'Input changed: {rel}')
            s=read_scene(folder);target=read_target(folder,s['shape']);target_before=sha(folder/'HDRImg.hdr')
            a=align_scene(s['ldr'],s['t']);pred=classical(s,a);batch=build_batch(s,a,device)
            with torch.inference_mode():
                for name,model in models.items():
                    output,_=model_forward(model,name,batch);pred[name]=output[0].permute(1,2,0).float().cpu().numpy().copy()
                    del output
            del batch
            record={'scene':key,'target_sha256':target_before,'crop_xywh':entry['crop']['xywh'],'prediction_sha256':{}}
            for method in METHODS:
                p=pred[method]
                if p.shape!=target.shape or not np.isfinite(p).all() or p.min()<0:raise ValueError(f'Invalid prediction: {key}/{method}')
                m=all_metrics(p,target);m['mae_mu']=float(np.abs(mu_law(p)-mu_law(target)).mean())
                if not all(np.isfinite(x) for x in m.values()):raise ValueError(f'Nonfinite metrics: {key}/{method}')
                rows.append({'scene':key,'method':method,**m});dest=args.out/'predictions'/key/method;dest.parent.mkdir(parents=True,exist_ok=True);np.save(str(dest)+'.npy',p)
                record['prediction_sha256'][method]=sha(Path(str(dest)+'.npy'))
            draw_case(args.out/'figures'/key,key,s,pred,target,entry['crop']['xywh'])
            if sha(folder/'HDRImg.hdr')!=target_before:raise RuntimeError('Source target changed')
            completed.append(record);write_csv(args.out/'per_scene_metrics.csv',rows);write_json(args.out/'completed_scenes.json',completed)
            write_json(args.out/'status.json',{'status':'incomplete','completed_scenes':len(completed),'expected_scenes':15})
            print(f'Finished {key}: M2 {rows[-2]["psnr_mu"]:.4f}, M3 {rows[-1]["psnr_mu"]:.4f} dB; {len(completed)}/15',flush=True)
            del s,a,pred,target
        if len(rows)!=120 or len(completed)!=15:raise RuntimeError('Incomplete method/scene grid')
        summary=summarize(rows);write_csv(args.out/'summary_metrics.csv',summary)
        write_json(args.out/'status.json',{'status':'complete','completed_scenes':15,'method_scene_pairs':120,'elapsed_seconds':time.time()-start,'claim_scope':'frozen within-dataset holdout; one seed; no cross-dataset robustness claim'})
        (args.out/'READ_RESULTS.txt').write_text('All 15 official test scenes were scored with fixed checkpoints and preprocessing. Primary summary: ALL15 mean PSNR-mu. PAPER and EXTRA are separate descriptive summaries. No test-target tuning. All figures use input-selected crops with locator boxes and one shared display. No per-method tone mapping. One seed, within-dataset holdout only. Review runtime version differences. Original predictions are under predictions/ and are excluded from the small report archive.\n')
        archive=args.out/'test_report_evidence.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for p in sorted(args.out.rglob('*')):
                if p.is_file() and p!=archive and 'predictions' not in p.relative_to(args.out).parts:tar.add(p,arcname=str(p.relative_to(args.out)))
            tar.add(args.input_manifest,arcname='input_manifest.json');tar.add(HERE/'test_protocol_frozen.json',arcname='test_protocol_frozen.json')
        print('\nCOMPLETE. Download:',archive,flush=True)
        for r in summary:
            if r['group']=='ALL15':print(r['method'],round(r['psnr_mu'],4),round(r['ssim_mu'],6),flush=True)
    except Exception:
        write_json(args.out/'status.json',{'status':'failed/incomplete','completed_scenes':len(completed),'elapsed_seconds':time.time()-start,'error':traceback.format_exc()});raise
if __name__=='__main__':main()
