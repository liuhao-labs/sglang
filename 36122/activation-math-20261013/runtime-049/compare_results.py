import hashlib
import json
from pathlib import Path
import subprocess

root=Path('/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj/runtime-049')
arms={role:json.loads((root/f'results-{role}/client.json').read_text()) for role in ('main','integrated')}
for x in arms.values():
    assert x['meta_info']['prompt_tokens']==8
    assert x['meta_info']['completion_tokens']==16
    assert x['meta_info']['cached_tokens']==0
assert arms['main']['text']==arms['integrated']['text']
assert arms['main']['output_ids']==arms['integrated']['output_ids']
normalized={'text':arms['main']['text'],'output_ids':arms['main']['output_ids'],
            'prompt_tokens':8,'completion_tokens':16,'cached_tokens':0}
digest=hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest()
provenance={role:json.loads((root/f'import-{role}.json').read_text()) for role in arms}
for role,p in provenance.items():
    assert p['capability'] is False
    assert p['kernel_version']=='0.4.9'
    assert p['packages']['torch']=='2.14.1'
    assert p['packages']['flashinfer-python']=='0.7.1'
    assert p['packages']['triton']=='3.8.0'
    assert all(v is None for v in p['version_check_disable_env'].values())
assert provenance['main']['packages']==provenance['integrated']['packages']
summary={'status':'PASS','paired_source_published_wheel':True,
         'token_ids_equal':True,'text_equal':True,'canonical_output_sha256':digest,
         'canonical_output':normalized,'packages':provenance['main']['packages'],
         'common_ops_sha256':provenance['main']['common_ops_sha256']}
(root/'paired-model-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
