"""Build a clean relocatable CPython runtime from distribution RECORD allowlists."""
import sys, shutil, json, hashlib, base64
from pathlib import Path
import importlib.metadata as md
try: from packaging.requirements import Requirement
except ImportError: from pip._vendor.packaging.requirements import Requirement
out=Path(sys.argv[1]); roots=sys.argv[2:]; base=Path(sys.base_prefix)
out.mkdir(parents=True,exist_ok=True)
for pattern in ('python*.exe','python*.dll','vcruntime*.dll','LICENSE.txt'):
 for p in base.glob(pattern):shutil.copy2(p,out/p.name)
def ignored(directory,names):
 return [n for n in names if n in ('site-packages','__pycache__','test','tests','idlelib','ensurepip') or n.endswith('.pyc')]
for folder in ('Lib','DLLs','tcl'):
 shutil.copytree(base/folder,out/folder,ignore=ignored,dirs_exist_ok=True)
site=out/'Lib/site-packages';site.mkdir(exist_ok=True)
seen={};verified=0

def visit(name,extras=()):
 global verified
 d=md.distribution(name);key=d.metadata['Name']
 if key in seen:return
 seen[key]=d.version
 source=Path(d.locate_file('')).resolve()
 for entry in d.files or []:
  relative=Path(str(entry))
  if '..' in relative.parts or relative.is_absolute() or '__pycache__' in relative.parts or relative.suffix=='.pyc':continue
  p=Path(d.locate_file(entry)).resolve()
  if not p.is_relative_to(source):raise RuntimeError('Dependency path escapes package')
  if not p.is_file():raise RuntimeError('Missing dependency file: '+str(relative))
  content=p.read_bytes()
  if entry.hash:
   actual=base64.urlsafe_b64encode(hashlib.new(entry.hash.mode,content).digest()).rstrip(b'=').decode()
   if actual!=entry.hash.value:raise RuntimeError('Installed dependency hash changed: '+key+' '+str(relative))
   verified+=1
  dest=site/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
 for raw in d.requires or []:
  r=Requirement(raw)
  if r.marker and not any(r.marker.evaluate({'extra':x}) for x in ('',*extras)):continue
  visit(r.name,r.extras)
for name in roots:visit(name)
(out/'DEPENDENCIES.json').write_text(json.dumps({'python':sys.version.split()[0],'packages':seen,'recordHashesVerified':verified},indent=2),encoding='utf-8')
print(json.dumps({'runtime':str(out),'packages':seen,'recordHashesVerified':verified},ensure_ascii=False))
