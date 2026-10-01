import argparse, json
from pathlib import Path

def load(root, name):
    with open(Path(root)/name, encoding='utf-8') as f:
        return json.load(f)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args=ap.parse_args(); root=args.root
    files=['category.json','scene.json','sample.json','sample_data.json','sample_annotation.json','sensor.json']
    data={}
    for f in files:
        p=Path(root)/f
        if p.exists(): data[f]=load(root,f)
    print('nuScenes metadata summary')
    for f,v in data.items(): print(f'{f:24s}: {len(v):,} records')
    if 'category.json' in data:
        print('\nCategories:')
        for x in data['category.json']:
            print(' -', x.get('name'))
    if 'scene.json' in data:
        print('\nFirst scenes:')
        for x in data['scene.json'][:5]: print(' -', x.get('name'), ':', x.get('description',''))

if __name__=='__main__': main()
