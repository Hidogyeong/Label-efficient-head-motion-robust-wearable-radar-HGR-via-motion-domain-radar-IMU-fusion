#!/usr/bin/env python3
"""Best-effort local pre-publish scan; prints locations, never credential values."""
from pathlib import Path
import re,sys,json
ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'private_archive','artifacts','workspaces','.git','__pycache__','exports'}
PATTERNS=[('private key',re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
 ('GitHub token',re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b')),
 ('AWS access ID',re.compile(r'\bAKIA[0-9A-Z]{16}\b'))]

def main():
    hits=[];warnings=[];n=0
    for p in ROOT.rglob('*'):
        if not p.is_file() or set(p.relative_to(ROOT).parts)&EXCLUDE:continue
        rel=p.relative_to(ROOT).as_posix()
        if rel=='tools/scan_before_publish.py':continue
        n+=1
        if p.stat().st_size>100*1024*1024:hits.append((rel,'file exceeds 100 MiB'))
        if p.name=='paths.env' or p.name=='.env' or p.name=='paths.local.json':
            warnings.append((rel,'local configuration: must be ignored by Git'));continue
        try:s=p.read_text(encoding='utf-8')
        except UnicodeError:continue
        for label,pattern in PATTERNS:
            for m in pattern.finditer(s):hits.append((rel,label+' at line '+str(s[:m.start()].count('\n')+1)))
        if p.suffix=='.ipynb':
            try:
                nb=json.loads(s)
                if any(c.get('outputs') for c in nb.get('cells',[])):warnings.append((rel,'notebook contains saved outputs'))
            except ValueError:warnings.append((rel,'notebook JSON unreadable'))
    print('[CANDIDATE FILES]',n)
    for x in warnings:print('[REVIEW]',*x)
    for x in hits:print('[BLOCK]',*x)
    print('This is NOT a complete credential, personal-data, or license review. Inspect git diff --cached and external rights before a public release.')
    return 1 if hits else 0
if __name__=='__main__':raise SystemExit(main())
