"""Stage the unmodified full AOT source and verified CMake FetchContent pins."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
target = root / 'full-wheel'
target.mkdir(exist_ok=True)
source = target / 'source'
if not source.exists():
    shutil.copytree(root / 'integrated/python/sglang/kernels/aot', source,
                    ignore=shutil.ignore_patterns('__pycache__'))
archives = target / 'archives'
archives.mkdir(exist_ok=True)
deps = target / 'deps'
deps.mkdir(exist_ok=True)
text = (source / 'CMakeLists.txt').read_text() + (source / 'cmake/flashmla.cmake').read_text()
pins = re.findall(r'FetchContent_Declare\(\s*([^\s]+)\s+URL\s+https://\$\{GITHUB_ARTIFACTORY\}/([^\s]+)\s+URL_HASH SHA256=([0-9a-f]{64})', text)
assert len(pins) == 7, pins
manifest = []
for name, urlpath, expected in pins:
    archive = archives / (name+'.tar.gz')
    url = 'https://github.com/'+urlpath
    if not archive.exists():
        print('Downloading pinned dependency', name, url, flush=True)
        subprocess.run(['curl', '--proxy', 'http://127.0.0.1:17890', '-fL',
                        '--connect-timeout', '20', '--max-time', '600',
                        '--retry', '2', '-o', str(archive), url], check=True)
    actual = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
    assert actual == expected, (name, actual, expected)
    location = (deps / 'repo-flashmla/csrc/cutlass' if name == 'repo-flashmla-cutlass'
                else deps / name)
    marker = location / '.probe-extraction-sha256'
    if not marker.exists() or marker.read_text().strip() != expected:
        location.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '-xzf', str(archive), '--strip-components=1',
                        '-C', str(location)], check=True)
        marker.write_text(expected+'\n')
    manifest.append({'name': name, 'url': url, 'sha256': actual,
                     'size': archive.stat().st_size,
                     'container_path': '/probe/full-wheel/'+str(location.relative_to(target))})
payload = {'integration_sha': '15d569117ce2d059ed2a3203ed53701063ace606',
           'integration_tree': '36909c6658f0ce1ed346db5b85a3568325b9357d',
           'aot_tree': subprocess.check_output(['git','-C',str(root/'integrated'),
                          'rev-parse','HEAD:python/sglang/kernels/aot'],text=True).strip(),
           'dependencies': manifest}
(target / 'source-manifest.json').write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps(payload), flush=True)
