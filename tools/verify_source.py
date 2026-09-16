#!/usr/bin/env python3
"""Verify archive preservation without executing user research scripts."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    errors=[];warnings=[];checked=0
    with (ROOT/'provenance/source_manifest.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    seen=set()
    for r in rows:
        rel=r['repository_path']
        if not rel or rel in seen:continue
        seen.add(rel);p=ROOT/rel
        if not p.is_file():
            if rel.startswith('private_archive/'):
                warnings.append('Optional local-only file absent (normal after Git clone): '+rel);continue
            errors.append('Missing: '+rel);continue
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['stored_sha256']:
            errors.append('Changed from organized snapshot: '+rel)
        checked+=1
    py_count=0
    for p in sorted(ROOT.rglob('*.py')):
        if set(p.relative_to(ROOT).parts)&{'private_archive','artifacts','workspaces','.git','__pycache__'}:continue
        try:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
        except Exception as e:errors.append('Syntax/decode error '+str(p.relative_to(ROOT))+': '+str(e))
        py_count+=1
    for p in sorted(ROOT.rglob('*.json')):
        if set(p.relative_to(ROOT).parts)&{'private_archive','artifacts','workspaces','.git'}:continue
        try:json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:errors.append('JSON error '+str(p.relative_to(ROOT))+': '+str(e))
    print('[SOURCE FILES HASH CHECKED]',checked)
    print('[PYTHON FILES PARSED]',py_count)
    for w in warnings:print('[NOTE]',w)
    for e in errors:print('[FAIL]',e)
    print('[RESULT]', 'FAIL' if errors else 'PASS (integrity/syntax only; not scientific reproduction)')
    return 1 if errors else 0
if __name__=='__main__':raise SystemExit(main())
