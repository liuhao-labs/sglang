import hashlib
import json
from pathlib import Path
root=Path('/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj')
rows=[]
for file in sorted((root/'wheels').glob('*.whl')):
    rows.append({'file':file.name,'size':file.stat().st_size,
                 'sha256':hashlib.file_digest(file.open('rb'),'sha256').hexdigest()})
(root/'runtime-049/wheel-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print('wheel manifest',len(rows),'files')
