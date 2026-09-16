#!/usr/bin/env python3
"""Copy archived preprocessing sources to a disposable workspace; do not execute them."""
from __future__ import annotations
from pathlib import Path
import argparse,datetime,json,shutil,uuid,csv
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--include-recovered-merge',action='store_true')
    a=ap.parse_args()
    tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=ROOT/'workspaces'/('preprocessing_'+tag)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/'legacy/BGT60TR13C',out/'BGT60TR13C')
    shutil.copytree(ROOT/'legacy/geometry',out/'geometry')
    if a.include_recovered_merge:shutil.copytree(ROOT/'recovered_prior_uploads',out/'recovered_prior_uploads')
    shutil.copy2(ROOT/'provenance/path_literals_to_review.csv',out/'PATHS_TO_EDIT.csv')
    (out/'READ_FIRST.txt').write_text('This is a WORKING COPY, not the immutable archive.\nEdit only required input/output paths in the selected script.\nInspect docs/PIPELINE_KO.md or docs/PIPELINE_EN.md first.\nDo NOT run every variant in order. Do not overwrite archived datasets.\nNo acquisition or preprocessing has been executed by this command.\nMissing acquisition/export and intermediate resampling steps still require confirmation.\n',encoding='utf-8')
    print('[WORKSPACE]',out);print('[NEXT] Read READ_FIRST.txt and PATHS_TO_EDIT.csv; no scripts have run.')
if __name__=='__main__':main()
