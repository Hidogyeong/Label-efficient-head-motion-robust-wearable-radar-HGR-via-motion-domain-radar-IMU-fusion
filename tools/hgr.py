#!/usr/bin/env python3
"""Safe, standard-library launcher around the archived experiment engines.
The experiment algorithms and original manifests are not rewritten by this tool.
Python 3.8+; Linux recommended for the per-result-root advisory lock.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
STAGES = json.loads((ROOT / 'configs/stages.json').read_text(encoding='utf-8'))

def stamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tasks(stage):
    s = STAGES[stage]
    p = ROOT / 'experiments' / s['project'] / s['manifest']
    m = json.loads(p.read_text(encoding='utf-8'))
    out = []
    for e in m['experiments']:
        c = dict(m.get('defaults', {})); c.update(e)
        folds = c.get('folds', m.get('folds', [0, 1, 2, 3, 4]))
        if isinstance(folds, int): folds = list(range(folds))
        for f in folds:
            for seed in c.get('seeds', m.get('seeds', [42, 43, 44])):
                out.append((c, int(f), int(seed)))
    return p, out

def config(args):
    p = Path(args.config).expanduser().resolve()
    if not p.is_file():
        raise ValueError('Local paths are missing. Run: python tools/hgr.py init; then edit configs/paths.local.json')
    cfg = json.loads(p.read_text(encoding='utf-8'))
    for key in ['data_root', 'result_root', 'paper_results_root']:
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError('Missing path: ' + key)
        cfg[key] = Path(os.path.expandvars(cfg[key])).expanduser().resolve()
    return cfg

def summary_status(root, c, f, seed):
    run = root / 'runs' / c['id'] / ('fold_%02d' % f) / ('seed_%d' % seed)
    p = run / 'run_summary.json'
    if not p.exists(): return 'pending'
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
        if d.get('exp_id') != c['id'] or int(d['fold']) != f or int(d['seed']) != seed:
            return 'invalid_summary_identity'
        if 'test_macro_f1' not in d: return 'missing_metrics'
    except Exception:
        return 'unreadable_summary'
    # Legacy engine only tests existence of summary; report absent model separately.
    return 'summary_and_model_present' if (run / 'model.pt').is_file() else 'summary_only_model_missing'

def live(cmd, cwd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[COMMAND]', ' '.join(map(str, cmd)), flush=True)
    print('[LOG]', log, flush=True)
    env = dict(os.environ); env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    with log.open('x', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            assert p.stdout is not None
            for line in p.stdout:
                print(line, end='', flush=True); f.write(line); f.flush()
            rc = p.wait()
        except KeyboardInterrupt:
            p.terminate()
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
            print('\n[STOPPED] Completed runs remain; interrupted run may restart from epoch 1.')
            return 130
        finally:
            if p.stdout is not None: p.stdout.close()
    print('[EXIT]', rc, flush=True)
    return rc

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=str(ROOT/'configs/paths.local.json'))
    sp = ap.add_subparsers(dest='command', required=True)
    sp.add_parser('init', help='Create a local JSON paths file; never overwrite it.')
    sp.add_parser('list', help='List stages and exact task counts without importing PyTorch.')
    st = sp.add_parser('status', help='Inspect summaries/checkpoints; does not launch training.')
    st.add_argument('--source', choices=['new','paper'], default='new')
    rn = sp.add_parser('run', help='Print plan; requires --execute to train.')
    rn.add_argument('stage', choices=sorted(STAGES))
    rn.add_argument('--execute', action='store_true')
    rn.add_argument('--cpu', action='store_true')
    rn.add_argument('--allow-existing-results', action='store_true', help='Acknowledge that pre-existing runs cannot be tied to this source automatically.')
    an = sp.add_parser('analyze', help='Run archived revision analyzer into a new timestamped snapshot.')
    an.add_argument('--source', choices=['new','paper'], default='new')
    sp.add_parser('verify', help='Check archived byte hashes and source syntax.')
    args = ap.parse_args()
    if args.command == 'init':
        p = Path(args.config).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists(): print('[UNCHANGED]', p)
        else:
            with p.open('x', encoding='utf-8') as f: f.write((ROOT/'configs/paths.example.json').read_text(encoding='utf-8'))
            print('[CREATED]', p)
        print('Edit paths before running training. Default result_root is NEW, not your paper results.')
        return 0
    if args.command == 'verify':
        return subprocess.call([sys.executable, str(ROOT/'tools/verify_source.py')])
    if args.command == 'list':
        for stage in STAGES:
            _, t = tasks(stage)
            print('%-12s %3d runs  %s' % (stage, len(t), STAGES[stage]['project']))
        print('Training is opt-in: run <stage> --execute. Analysis is a separate command.')
        return 0
    cfg = config(args)
    if args.command == 'status':
        from collections import Counter
        root = cfg['paper_results_root'] if args.source=='paper' else cfg['result_root']
        print('[RESULT ROOT]', root)
        for stage in STAGES:
            _, ts = tasks(stage)
            counts = Counter(summary_status(root,c,f,s) for c,f,s in ts)
            print(stage, dict(counts))
        print('Presence checks do not verify historical data/source identity or training quality.')
        return 0
    if args.command == 'run':
        manifest, ts = tasks(args.stage)
        proj = ROOT/'experiments'/STAGES[args.stage]['project']
        engine = proj/'run_fusion_protocol_suite.py'
        cmd = [sys.executable, '-u', str(engine), '--data_root', str(cfg['data_root']),
               '--result_root', str(cfg['result_root']), '--manifest', str(manifest), '--resume']
        if args.cpu: cmd.append('--cpu')
        print('[PLAN]', args.stage, len(ts), 'runs')
        print('[DATA]', cfg['data_root']);print('[RESULTS]', cfg['result_root'])
        if not args.execute:
            print('[DRY RUN]', ' '.join(cmd));print('Add --execute to run. No experiments were started.')
            return 0
        if not cfg['data_root'].is_dir(): raise ValueError('Dataset is not mounted: '+str(cfg['data_root']))
        if cfg['result_root']==cfg['data_root']: raise ValueError('data_root and result_root must differ.')
        cfg['result_root'].mkdir(parents=True,exist_ok=True)
        try:
            import fcntl
        except ImportError:
            raise ValueError('This safe training launcher requires Linux/POSIX fcntl. Use the archived scripts manually on other platforms.')
        lock = (cfg['result_root']/'.hgr_archive_runner.lock').open('a+')
        try:
            try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise ValueError('Another archive launcher is using this result root. Run stages sequentially.')
            finger = {'engine_sha256':sha(engine),'manifest_sha256':sha(manifest),'data_root':str(cfg['data_root'])}
            fp = cfg['result_root']/'.archive_run_fingerprints'/(args.stage+'.json')
            if fp.exists():
                prior=json.loads(fp.read_text())
                if prior['fingerprint']!=finger:
                    raise ValueError('Code, manifest, or data-root path changed. Use a NEW result_root; no overwriting.')
            else:
                present = [summary_status(cfg['result_root'],c,f,s) for c,f,s in ts]
                if any(x!='pending' for x in present) and not args.allow_existing_results:
                    raise ValueError('Pre-existing summaries found without an archive fingerprint. Analyze them with --source paper, or use a NEW result_root. Explicit --allow-existing-results accepts the provenance risk.')
                fp.parent.mkdir(parents=True,exist_ok=True)
                fp.write_text(json.dumps({'fingerprint':finger,'note':'Dataset contents are NOT hashed by this launcher. Use a new result root after data changes.'},indent=2)+'\n')
            invalid = [summary_status(cfg['result_root'],c,f,s) for c,f,s in ts]
            if any(x in ['invalid_summary_identity','missing_metrics','unreadable_summary'] for x in invalid):
                raise ValueError('Invalid existing summary found. Preserve original results and inspect before resuming.')
            tag=stamp();log=ROOT/'artifacts/logs'/(tag+'_'+args.stage+'.log')
            meta={'stage':args.stage,'command':cmd,'cwd':str(proj),'fingerprint':finger,'synthetic_or_real':'specified by caller; not inferred'}
            log.parent.mkdir(parents=True,exist_ok=True)
            log.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
            return live(cmd,proj,log)
        finally:
            lock.close()
    if args.command == 'analyze':
        source = cfg['paper_results_root'] if args.source=='paper' else cfg['result_root']
        files=sorted((source/'runs').glob('*/fold_*/seed_*/run_summary.json'))
        if not files: raise ValueError('No run_summary.json files at '+str(source/'runs'))
        # Analyzers read untrusted summaries as JSON, never load model pickle files here.
        for p in files: json.loads(p.read_text(encoding='utf-8'))
        out=ROOT/'artifacts/analysis'/stamp();out.mkdir(parents=True,exist_ok=False)
        with (out/'source_summaries_used.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['path','sha256'])
            for p in files:w.writerow([str(p),sha(p)])
        analyzer=ROOT/'experiments/revision/analyze_revision_lr_stats.py'
        (out/'analysis_provenance.json').write_text(json.dumps({'source_root':str(source),'analyzer_sha256':sha(analyzer),'warning':'Uses legacy fold/seed paired tests without independence or multiple-testing correction. Review inference separately.'},indent=2)+'\n')
        cmd=[sys.executable,'-u',str(analyzer),'--result_root',str(source),'--analysis_dir',str(out)]
        rc=live(cmd,analyzer.parent,out/'analysis.log')
        print('[SNAPSHOT]',out)
        if rc==0:
            zp=out/'REVISION_CHATGPT_UPLOAD_PACKAGE.zip'
            if not zp.is_file(): raise ValueError('Analyzer exited without expected ZIP. See '+str(out/'analysis.log'))
            import zipfile
            with zipfile.ZipFile(zp,'a',zipfile.ZIP_DEFLATED) as z:
                for name in ['source_summaries_used.csv','analysis_provenance.json']:z.write(out/name,arcname=name)
            print('[UPLOAD AFTER REVIEW]',zp)
        return rc
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,OSError,json.JSONDecodeError) as e:
        print('[ERROR]',e,file=sys.stderr);raise SystemExit(2)
