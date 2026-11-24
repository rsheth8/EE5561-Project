"""Exploratory HDR failure analysis on the team's frozen validation scenes.

AI-supplied starting analysis code. Review the masks, metrics and interpretation.
Requires classical_baselines.py and motion_rejection.py beside this file.
Reads source data/predictions only. Does not train, align, tune or overwrite runs.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tarfile
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import skimage

import classical_baselines as base
import motion_rejection as rejection

FROZEN_VAL = ['025', '045', '051', '056', '062']
REGIONS = ['all', 'reference_usable', 'reference_bright_or_clipped', 'reference_dark_only']
LEARNED = ['M0', 'M1', 'M2', 'M3']
LOW, HIGH, THRESHOLD, MU = 0.05, 0.95, 0.20, 5000.0


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_scene(folder):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    e = np.loadtxt(folder / 'exposure.txt').reshape(-1)
    if len(files) != 3 or e.size != 3 or not np.isfinite(e).all():
        raise ValueError(f'{folder}: expected three TIFFs and three finite exponents.')
    times = np.exp2(e)
    if not np.all(np.isfinite(times) & (times > 0)) or not np.all(np.diff(times) > 0):
        raise ValueError(f'{folder}: require increasing finite positive exposure times.')
    raw = [base.read_rgb(p) for p in files]
    if any(x.dtype != np.uint16 for x in raw) or len({x.shape for x in raw}) != 1:
        raise ValueError(f'{folder}: expected equal-shape uint16 TIFFs.')
    ldr = np.stack([x.astype(np.float32) / 65535 for x in raw])
    times = times.astype(np.float32)
    rad = ldr ** 2.2 / times[:, None, None, None]
    target_path = folder / 'HDRImg.hdr'
    target = base.read_rgb(target_path).astype(np.float32)
    if target.shape != ldr.shape[1:] or not np.all((target >= 0) & (target <= 1)):
        raise ValueError(f'{folder}: target shape/range does not match the declared convention.')
    if min(target.shape[:2]) < 7:
        raise ValueError('SSIM requires image dimensions >= 7.')
    source_paths = files + [folder / 'exposure.txt', target_path]
    return ldr, rad, target, times, {str(p): sha(p) for p in source_paths}


def region_masks(reference):
    bright = np.any(reference >= HIGH, axis=-1)
    dark = (~bright) & np.any(reference <= LOW, axis=-1)
    usable = ~(bright | dark)
    assert np.all(usable.astype(np.uint8) + bright + dark == 1)
    return {'all': np.ones(reference.shape[:2], bool), 'reference_usable': usable,
            'reference_bright_or_clipped': bright, 'reference_dark_only': dark}


def region_scores(mapped_pred, mapped_target, mask):
    count = int(mask.sum())
    if count == 0:
        return count, None, None, None
    error = mapped_pred[mask] - mapped_target[mask]
    mae = float(np.abs(error).mean())
    mse = float(np.square(error).mean())
    return count, mae, mse, float(-10 * np.log10(mse)) if mse else float('inf')


def crop_from_brightness(mask, requested):
    """Largest bright-flag fraction on a fixed grid, selected from INPUT only."""
    height, width = mask.shape
    size = min(requested, height, width)
    ys = sorted(set(range(0, height - size + 1, 32)) | {height - size})
    xs = sorted(set(range(0, width - size + 1, 32)) | {width - size})
    integral = cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_64F)
    best, coord = -1, (0, 0)
    for y in ys:
        for x in xs:
            count = integral[y+size, x+size] - integral[y, x+size] - integral[y+size, x] + integral[y, x]
            if count > best:
                best, coord = count, (y, x)
    y, x = coord
    return x, y, size, size


def show_rgb(ax, rgb, title, crop=None):
    if crop is not None:
        x, y, w, h = crop
        rgb = rgb[y:y+h, x:x+w]
    ax.imshow(np.clip(rgb, 0, 1))
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def scene_figures(out, sid, ldr, target, methods, masks, crop, scope):
    od = out / 'figures'
    x, y, w, h = crop
    ordered = [('reference_only', 'Reference only'),
               ('weighted_merge_no_alignment', 'Weighted merge'),
               ('reference_fallback_rejection', 'Whole-pixel fallback')]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for k, (name, label) in enumerate(ordered):
        show_rgb(axes[0, k], base.mu_map(methods[name]), label)
        show_rgb(axes[1, k], base.mu_map(methods[name]), 'Same input-selected crop', crop)
    show_rgb(axes[0, 3], base.mu_map(target), 'HDR target')
    show_rgb(axes[1, 3], base.mu_map(target), 'Same input-selected crop', crop)
    fig.suptitle(f'Scene {sid}: classical controls, fixed threshold 0.20', fontsize=11)
    fig.text(.5, .015, f'All panels: fixed mu-law display, mu=5000. Crop x={x}, y={y}, w={w}, h={h}; selected by bright-reference fraction, not output error.', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .05, 1, .94)); fig.subplots_adjust(hspace=.4); fig.savefig(od / f'{sid}_classical.png', dpi=130); plt.close(fig)

    if 'M2' not in methods:
        return
    p2, p3, gt = (base.mu_map(methods['M2']), base.mu_map(methods['M3']), base.mu_map(target))
    e2, e3 = np.abs(p2-gt).mean(-1), np.abs(p3-gt).mean(-1)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for k, (im, title) in enumerate([(p2, 'M2: constant cues'), (p3, 'M3: measured cues'), (gt, 'HDR target')]):
        show_rgb(axes[0, k], im, title, crop)
    axes[0, 3].imshow(masks['reference_bright_or_clipped'][y:y+h, x:x+w], cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title('Reference bright flag (input)', fontsize=9); axes[0, 3].axis('off')
    for k, (err, title) in enumerate([(e2, 'M2 absolute mu error'), (e3, 'M3 absolute mu error')]):
        m = axes[1, k].imshow(err[y:y+h, x:x+w], cmap='inferno', vmin=0, vmax=.05)
        axes[1, k].set_title(title, fontsize=9); axes[1, k].axis('off')
        fig.colorbar(m, ax=axes[1, k], fraction=.046, pad=.02)
    m=axes[1, 2].imshow((e3-e2)[y:y+h, x:x+w], cmap='coolwarm', vmin=-.02, vmax=.02)
    axes[1, 2].set_title('M3 minus M2 error\nblue: M3 better; red: worse', fontsize=9); axes[1, 2].axis('off')
    fig.colorbar(m, ax=axes[1, 2], fraction=.046, pad=.02)
    show_rgb(axes[1, 3], ldr[1], 'Reference photo (stored encoding)', crop)
    fig.suptitle(f'Scene {sid}: explicit-cue comparison on the same input-selected crop', fontsize=11)
    fig.text(.5, .015, f'{scope}. Crop x={x}, y={y}, w={w}, h={h}. Error scales are fixed; display saturation does not affect numeric scores.', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .05, 1, .94)); fig.subplots_adjust(hspace=.4); fig.savefig(od / f'{sid}_cue_comparison.png', dpi=130); plt.close(fig)


def summaries(rows):
    output=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        for region in REGIONS:
            group=[r for r in rows if r['method']==method and r['region']==region]
            nonempty=[r for r in group if r['pixel_count']]
            output.append({'method':method, 'region':region, 'scenes_total':len(group),
                           'scenes_nonempty':len(nonempty), 'pixel_count_total':sum(r['pixel_count'] for r in group),
                           'mean_region_fraction':float(np.mean([r['region_fraction'] for r in group])),
                           'mean_mae_mu_per_scene':float(np.mean([r['mae_mu'] for r in nonempty])) if nonempty else None,
                           'mean_psnr_mu_per_scene':float(np.mean([r['psnr_mu'] for r in nonempty])) if nonempty else None,
                           'mean_ssim_mu_full_image':float(np.mean([r['ssim_mu_full_image'] for r in group])) if region=='all' else None})
    return output


def differences(rows, has_learned):
    index={(r['scene_id'],r['method'],r['region']):r for r in rows}
    pairs=[('reference_fallback_rejection','weighted_merge_no_alignment'),
           ('reference_fallback_rejection','reference_only')]
    if has_learned:
        pairs += [('M1','M0'),('M2','M1'),('M3','M2')]
    output=[]
    for sid in dict.fromkeys(r['scene_id'] for r in rows):
        for a,b in pairs:
            for region in REGIONS:
                ra,rb=index[(sid,a,region)],index[(sid,b,region)]
                output.append({'scene_id':sid,'comparison':f'{a}_minus_{b}','region':region,
                               'pixel_count':ra['pixel_count'], 'region_fraction':ra['region_fraction'],
                               'delta_mae_mu':ra['mae_mu']-rb['mae_mu'] if ra['pixel_count'] else None,
                               'delta_psnr_mu':ra['psnr_mu']-rb['psnr_mu'] if ra['pixel_count'] else None})
    return output


def overview_figures(out, rows, diffs, scenes):
    first={r['scene_id']:{x['region']:x['region_fraction'] for x in rows if x['scene_id']==r['scene_id'] and x['method']=='reference_only'} for r in rows}
    fig,ax=plt.subplots(figsize=(7,3.4))
    bottom=np.zeros(len(scenes))
    for region,label,color in [('reference_usable','Usable reference','#6fa88c'),('reference_bright_or_clipped','Bright / near-clipped','#e1ac46'),('reference_dark_only','Dark, not bright','#8c9eaf')]:
        values=np.array([first[s][region]*100 for s in scenes])
        ax.bar(scenes,values,bottom=bottom,label=label,color=color);bottom+=values
    ax.set_ylabel('Reference pixels (%)');ax.set_ylim(0,105);ax.legend(fontsize=8,loc='lower right')
    ax.set_title('Input-defined reference regions, fixed bounds 0.05 / 0.95')
    fig.tight_layout();fig.savefig(out/'figures/region_fractions.png',dpi=150);plt.close(fig)
    if not any(d['comparison']=='M3_minus_M2' for d in diffs):
        return
    fig,ax=plt.subplots(figsize=(8,3.8));positions=np.arange(len(scenes))
    for k,(region,label,color) in enumerate([('all','Full image','#497fa1'),('reference_usable','Usable reference','#3d9977'),('reference_bright_or_clipped','Bright / near-clipped','#d38e32')]):
        vals=[]
        for sid in scenes:
            d=next(r for r in diffs if r['scene_id']==sid and r['comparison']=='M3_minus_M2' and r['region']==region)
            vals.append(d['delta_mae_mu']*1000 if d['delta_mae_mu'] is not None else np.nan)
        ax.bar(positions+(k-1)*.24,vals,width=.24,label=label,color=color)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(positions,scenes)
    ax.set_ylabel('M3 - M2 mean absolute mu error (units of 0.001)')
    ax.set_title('Below zero: M3 helps; above zero: M3 hurts')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'figures/cue_gain_by_scene.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='Kalantari Training directory, containing scene folders.')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--learned-dir',type=Path,help='Existing float predictions: M0/025.npy, ..., M3/062.npy. Never use display PNGs here.')
    parser.add_argument('--expected-metrics',type=Path,help='Saved team validation CSV, used only to check recomputed full-image model scores.')
    parser.add_argument('--scenes',nargs='+',default=FROZEN_VAL,help='Default frozen five validation IDs. Other IDs are explicitly diagnostic only.')
    parser.add_argument('--crop-size',type=int,default=256,help='Display crop only; metrics always use whole images or full input-defined regions.')
    args=parser.parse_args()
    if args.out.exists():parser.error('Output already exists. Use a new run name to preserve it.')
    if len(set(args.scenes))!=len(args.scenes) or args.crop_size<7:parser.error('Require unique scene IDs and crop size >=7.')
    if args.expected_metrics and not args.learned_dir:parser.error('--expected-metrics requires --learned-dir.')
    for sid in args.scenes:
        if not sid.isdigit() or len(sid)!=3:parser.error('Use three-digit numeric scene IDs.')
        if not (args.data/sid).is_dir():parser.error(f'Scene folder missing: {args.data/sid}')
        if args.learned_dir:
            for model in LEARNED:
                path=args.learned_dir/model/f'{sid}.npy'
                if not path.is_file() or not os.access(path,os.R_OK):parser.error(f'Missing/unreadable prediction: {path}')
    expected={}
    if args.expected_metrics:
        with args.expected_metrics.open() as f:
            expected={(r['scene'],r['model']):r for r in csv.DictReader(line for line in f if not line.startswith('#')) if r['scene']!='mean'}
        for sid in args.scenes:
            for model in LEARNED:
                if (sid,model) not in expected:parser.error(f'No expected score for {sid}/{model}.')
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'figures').mkdir()
    frozen=args.scenes==FROZEN_VAL
    config={'status':'running','analysis_status':'Exploratory validation analysis, not independent test evidence' if frozen else 'Nonstandard-scene diagnostic only; not frozen validation results',
            'scenes':args.scenes,'frozen_validation_ids':FROZEN_VAL,'gamma':2.2,'mu':MU,'fallback_threshold':THRESHOLD,
            'reference_brightness_bounds':[LOW,HIGH], 'regions':{'reference_usable':'every RGB channel strictly between .05 and .95','reference_bright_or_clipped':'any RGB channel >= .95; a brightness proxy, not verified scene-radiance clipping','reference_dark_only':'not bright, but any RGB channel <= .05'},
            'summary_rule':'Equal scene means among nonempty regions; empty regions are blank and omitted, never treated as zero error.',
            'region_metrics':'RGB mu-law MAE and PSNR; SSIM only on full images to avoid cross-region window ambiguity.',
            'no_change_regions':'Whole-pixel fallback equals the original weighted merge wherever the reference is unusable BY CONSTRUCTION. This is not a new empirical finding.',
            'fixed_settings':'No threshold tuning, output scaling, training or alignment. Prediction/target used for error metrics; input masks and crop selection use reference input only.',
            'crop_selection':'256x256 default, largest bright-reference fraction over fixed 32-pixel grid, ties resolved top-left. Display only.',
            'predictions_dir':str(args.learned_dir.resolve()) if args.learned_dir else None,
            'expected_metrics_file':str(args.expected_metrics.resolve()) if args.expected_metrics else None,
            'expected_score_tolerance':{'psnr_mu_db':0.001,'ssim_mu':0.00001},
            'python':sys.version.split()[0],'platform':platform.platform(),'numpy':np.__version__,'opencv':cv2.__version__,'skimage':skimage.__version__,
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'code_sha256':{'analysis':sha(__file__),'baseline':sha(base.__file__),'fallback':sha(rejection.__file__)},
            'attribution':'AI-supplied starting analysis code; students must inspect, verify, interpret and describe their actual work accurately.'}
    if args.expected_metrics:config['expected_metrics_sha256']=sha(args.expected_metrics)
    cfgpath=args.out/'config.json'
    def save_config():cfgpath.write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    save_config();start=time.perf_counter();rows=[];audit=[];scene_info=[];hashes={}
    cv2.setNumThreads(1)
    try:
        for sid in args.scenes:
            print(f'Loading scene {sid}...',flush=True)
            ldr,rad,target,t,source_hashes=read_scene(args.data/sid);hashes.update(source_hashes)
            fallback,mask,_,_,ref_usable=rejection.reject_motion(ldr,rad,THRESHOLD,LOW,HIGH)
            methods={'reference_only':base.reference_only(rad),'weighted_merge_no_alignment':base.weighted_merge(ldr,rad),'reference_fallback_rejection':fallback}
            if args.learned_dir:
                for model in LEARNED:
                    path=args.learned_dir/model/f'{sid}.npy'
                    pred=np.load(path,allow_pickle=False)
                    if pred.shape!=target.shape or not np.issubdtype(pred.dtype,np.floating):raise ValueError(f'Prediction must be HWC floating point, matching target: {path}')
                    methods[model]=pred;hashes[str(path)]=sha(path)
            masks=region_masks(ldr[1]);assert np.array_equal(masks['reference_usable'],ref_usable)
            assert np.array_equal(fallback[~ref_usable],methods['weighted_merge_no_alignment'][~ref_usable])
            mapped_target=base.mu_map(target)
            for name,pred in methods.items():
                full_psnr,full_ssim=base.evaluate(pred,target)
                if name in LEARNED and expected:
                    old=expected[(sid,name)];dp=full_psnr-float(old['psnr_mu']);ds=full_ssim-float(old['ssim_mu'])
                    passed=abs(dp)<=.001 and abs(ds)<=.00001
                    audit.append({'scene_id':sid,'model':name,'psnr_mu_recomputed':full_psnr,'psnr_mu_saved':old['psnr_mu'],'delta_psnr_mu':dp,'ssim_mu_recomputed':full_ssim,'ssim_mu_saved':old['ssim_mu'],'delta_ssim_mu':ds,'within_tolerance':passed})
                    if not passed:raise ValueError(f'{sid}/{name}: full-image scores disagree with saved record (dPSNR={dp:.6g}, dSSIM={ds:.6g}). Stop and investigate data/checkpoint pairing.')
                mapped=base.mu_map(pred)
                for region,region_mask in masks.items():
                    count,mae,mse,psnr=region_scores(mapped,mapped_target,region_mask)
                    rows.append({'scene_id':sid,'method':name,'region':region,'pixel_count':count,'region_fraction':count/region_mask.size,'mae_mu':mae,'mse_mu':mse,'psnr_mu':psnr,'ssim_mu_full_image':full_ssim if region=='all' else None})
                print(f'  {name}: PSNR-mu {full_psnr:.4f}; SSIM-mu {full_ssim:.6f}',flush=True)
            crop=crop_from_brightness(masks['reference_bright_or_clipped'],args.crop_size)
            scene_figures(args.out,sid,ldr,target,methods,masks,crop,'Exploratory validation analysis' if frozen else 'Nonstandard-scene diagnostic')
            scene_info.append({'scene_id':sid,'height':target.shape[0],'width':target.shape[1],'relative_times':t.tolist(),'crop_xywh':list(crop),'fallback_fraction':float(mask.mean()),'exact_full_scale_reference_fraction':float(np.any(ldr[1]>=1,axis=-1).mean())})
            write_csv(args.out/'per_scene_regions.csv',rows);write_csv(args.out/'full_image_score_audit.csv',audit)
        summary=summaries(rows);diffs=differences(rows,bool(args.learned_dir))
        write_csv(args.out/'summary_regions.csv',summary);write_csv(args.out/'paired_differences.csv',diffs)
        overview_figures(args.out,rows,diffs,args.scenes)
        config.update(status='completed',elapsed_seconds=time.perf_counter()-start,scenes_completed=scene_info,input_sha256=hashes,full_image_model_score_audit_passed=all(r['within_tolerance'] for r in audit) if expected else None)
        save_config()
        notes='''EXPLORATORY FAILURE ANALYSIS\n\nRead summary_regions.csv for equal-scene averages and per_scene_regions.csv for all scene/region counts and scores. Smaller MAE is better; larger PSNR is better. For paired_differences.csv, negative delta_mae_mu means the first method helps.\n\nRegions come from the reference input. Bright/near-clipped is a proxy; it is not a ground-truth motion mask. Empty regions are blank. Regional SSIM is deliberately not calculated.\n\nThe fallback keeps the weighted merge in unusable reference regions by construction. Their equality there is an implementation check, not a discovery.\n\nM3-M2 is the explicit-cue comparison at equal parameter count. M2-M1 also changes architecture/capacity. This is post-hoc validation analysis, not a new independent test or statistical proof. Preserve all scenes and unfavorable outcomes.\n\nThe figures select the crop using bright-reference fraction only. Inspect every scene, then write what you observe in your own words. The error-map scales are fixed. Full-image scores always use all pixels, not the illustrated crop.\n\nDo not count use of a supplied script as independent authorship of its code. Record your review, checks, experiments, interpretation and any changes honestly.\n'''
        (args.out/'READ_RESULTS.txt').write_text(notes)
        archive=args.out/'report_evidence.tar.gz'
        with tarfile.open(archive,'x:gz') as tf:
            for p in sorted(args.out.rglob('*')):
                if p.is_file() and p!=archive:tf.add(p,arcname=str(p.relative_to(args.out)))
        print(f'\nCOMPLETE: {args.out}',flush=True)
        print('Primary outputs: summary_regions.csv, paired_differences.csv, full_image_score_audit.csv, figures/.',flush=True)
        print(f'Download and share: {archive}',flush=True)
    except Exception as exc:
        config.update(status='failed',error=str(exc),elapsed_seconds=time.perf_counter()-start)
        save_config();write_csv(args.out/'per_scene_regions.csv',rows);write_csv(args.out/'full_image_score_audit.csv',audit)
        raise


if __name__=='__main__':main()
