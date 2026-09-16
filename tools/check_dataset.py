#!/usr/bin/env python3
"""Read-only CSV inventory/schema audit. Not a substitute for a full model run."""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,uuid
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LABELS=['01RtoL','02LtoR','03UtoD','04DtoU','05None']
HEADS={'none':'None','down':'Down','left':'Left','random':'Random','right':'Right','up':'UP'}
RADAR=['range','doppler','horizontal_angle','vertical_angle']
QUAT=['q.w','q.i','q.j','q.k']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--scan-rows',action='store_true');p.add_argument('--hash-files',action='store_true')
    a=p.parse_args();root=Path(a.root).expanduser().resolve()
    if not root.is_dir():raise SystemExit('Dataset root not found: '+str(root))
    rows=[];counts=Counter()
    for person in sorted(root.iterdir()):
        if not person.is_dir():continue
        for label in LABELS:
            folder=person/label
            if not folder.is_dir():continue
            for head in sorted(folder.iterdir()):
                if not head.is_dir():continue
                h=HEADS.get(head.name.lower(),head.name)
                for f in sorted(head.rglob('*.csv')):
                    d={'file':str(f.relative_to(root)),'person':person.name,'gesture':label,'head':h,
                       'readable':False,'has_raw_backups':False,'corrected_radar_columns':0,
                       'has_quaternion':False,'has_world_xyz':False,'row_count':'','sha256':'','error':''}
                    try:
                        with f.open(encoding='utf-8-sig',newline='') as stream:
                            rd=csv.reader(stream); cols=next(rd)
                            d.update(readable=True,has_raw_backups=all(c+'_raw' in cols for c in RADAR),
                                     corrected_radar_columns=sum(c+'_corr' in cols for c in RADAR),
                                     has_quaternion=all(c in cols for c in QUAT),
                                     has_world_xyz=all(c in cols for c in ['radar_x_world','radar_y_world','radar_z_world']))
                            if a.scan_rows:d['row_count']=sum(1 for _ in rd)
                        if a.hash_files:
                            hasher=hashlib.sha256()
                            with f.open('rb') as s:
                                for block in iter(lambda:s.read(1024*1024),b''):hasher.update(block)
                            d['sha256']=hasher.hexdigest()
                    except Exception as e:d['error']=str(e)
                    rows.append(d);counts[(label,h)]+=1
    if not rows:raise SystemExit('No candidate CSV trials found under person/gesture/head/.')
    out=ROOT/'artifacts/dataset_audits'/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    with (out/'file_schema_inventory.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'sample_counts_long.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['gesture','head','candidate_csv_count'])
        for (l,h),n in sorted(counts.items()):w.writerow([l,h,n])
    summary={'data_root':str(root),'candidate_trials':len(rows),'subjects':sorted(set(r['person'] for r in rows)),
             'four_class_candidate_trials':sum(r['gesture']!='05None' for r in rows),
             'files_with_all_4_corrected_radar_columns':sum(r['corrected_radar_columns']==4 for r in rows),
             'files_with_quaternion':sum(r['has_quaternion'] for r in rows),
             'files_with_world_xyz':sum(r['has_world_xyz'] for r in rows),
             'read_failures':sum(not r['readable'] for r in rows),
             'warning':'Counts are candidate files, not validated training samples. Legacy radar_only prefers *_corr columns. Column presence does not establish how they were produced.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('[OUTPUT]',out)
if __name__=='__main__':main()
