#!/usr/bin/env python3
"""Export the user's ACTIVE environment, without installing or modifying packages.
Outputs stay under ignored environment/exports until manually reviewed.
"""
from __future__ import annotations
import datetime,json,os,platform,re,shutil,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def redact(s):
    s=re.sub(r'(?im)^prefix:.*$', 'prefix: <LOCAL_PREFIX_REMOVED>', s)
    s=re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@',r'\1<REDACTED>@',s)
    s=re.sub(r'(/t/)[^/\s]+',r'\1<REDACTED>',s)
    s=re.sub(r'(?i)([?&](?:token|key|api_key|access_token)=)[^\s&]+',r'\1<REDACTED>',s)
    return s

def main():
    tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=ROOT/'environment/exports'/tag;out.mkdir(parents=True,exist_ok=False)
    cmds=[('pip_freeze.txt',[sys.executable,'-m','pip','freeze']),('pip_check.txt',[sys.executable,'-m','pip','check']),
          ('conda_full_no_builds.yml',['conda','env','export','--prefix',sys.prefix,'--no-builds']),
          ('conda_from_history.yml',['conda','env','export','--prefix',sys.prefix,'--from-history']),
          ('conda_explicit.txt',['conda','list','--prefix',sys.prefix,'--explicit']),
          ('gpu_driver.txt',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'])]
    status=[]
    for name,cmd in cmds:
        if not Path(cmd[0]).is_file() and shutil.which(cmd[0]) is None:
            status.append({'file':name,'status':'command_missing','command':cmd});continue
        try:
            r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
            # Do not create an invalid YAML pretending to be a successful export.
            target=name if r.returncode==0 else name+'.failed.txt'
            txt=r.stdout if r.returncode==0 else r.stdout+'\nSTDERR\n'+r.stderr
            (out/target).write_text(redact(txt),encoding='utf-8')
            status.append({'file':target,'returncode':r.returncode,'command':cmd})
        except Exception as e:status.append({'file':name,'status':'failed','error':str(e)})
    runtime={'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'platform':platform.platform(),
             'conda_env_name':os.environ.get('CONDA_DEFAULT_ENV'),'torch':None}
    try:
        import torch
        runtime['torch']={'version':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'cudnn':torch.backends.cudnn.version()}
    except Exception as e:runtime['torch_error']=str(e)
    (out/'runtime.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n')
    (out/'export_status.json').write_text(json.dumps(status,indent=2)+'\n')
    (out/'REVIEW_BEFORE_GIT.txt').write_text('Inspect package URLs, editable/local paths, credentials, user/host identifiers before copying exports into a tracked location. Redaction is best-effort, not a security guarantee. The export reflects the environment active NOW, not proven historical versions. Remove the prefix line (do not restore <LOCAL_PREFIX_REMOVED>) before conda env create.\n')
    print('[EXPORTED]',out);print('Review files locally before publishing. No packages were installed.')
if __name__=='__main__':main()
