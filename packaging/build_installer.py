"""Compile an already audited standalone payload; never read live user data."""
import argparse, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--payload',required=True);p.add_argument('--output',required=True);p.add_argument('--makensis',required=True)
a=p.parse_args();stage=Path(a.payload).resolve();output=Path(a.output).resolve();name=Path(__file__).resolve().parents[1].name
if name not in {'WhoChat','VideoCollector'}:raise ValueError('Unknown application folder')
if output.is_relative_to(stage):raise ValueError('Output must be outside payload')
files=sorted(f for f in stage.rglob('*') if f.is_file() and f.name!='payload-manifest.json')
for f in files:
    if f.relative_to(stage).parts[0] in {'data','browser','screenshots'}:raise ValueError('User data directory in payload')
    if f.is_symlink() or f.name in {'config.json','Cookies','Login Data','secrets.json','credentials.json'} or f.suffix in {'.db','.pyc','.key'} or any(x in f.relative_to(stage).parts for x in ('.runtime','.whochat-data','__pycache__')):
        raise ValueError('Private or unreviewed file in payload: '+str(f.relative_to(stage)))
manifest={f.relative_to(stage).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
manifest_path=stage/'payload-manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8');files.append(manifest_path)
delete='\n'.join('  Delete "$INSTDIR\\'+str(f.relative_to(stage)).replace('/','\\')+'"' for f in files)
dirs=set()
for f in files:
 d=f.parent
 while d!=stage:dirs.add(d);d=d.parent
remove='\n'.join('  RMDir "$INSTDIR\\'+str(d.relative_to(stage)).replace('/','\\')+'"' for d in sorted(dirs,key=lambda d:len(d.parts),reverse=True))
template=Path(__file__).with_name('installer-template.nsi').read_text(encoding='utf-8-sig')
script=template.replace('@NAME@',name).replace('@OUTPUT@',str(output)).replace('@STAGE@',str(stage)).replace('@DELETE@',delete).replace('@REMOVE@',remove)
output.parent.mkdir(parents=True,exist_ok=True)
script_path=output.parent/(name+'.build.nsi');script_path.write_text(script,encoding='utf-8-sig')
subprocess.run([a.makensis,'/V2',str(script_path)],check=True)
print(json.dumps({'installer':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'files':len(files)}))
