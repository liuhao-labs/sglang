"""Copy selected experiment evidence, never source trees/caches/private state."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
target = root/'publish/36122/activation-math-20261013/production-bindings-followup'
target.mkdir(parents=True, exist_ok=True)
files = ['audit_latest.py', 'latest-drift.json', 'resource-samples.jsonl',
         'record_resources.py', 'prepare_full_wheel.py', 'build_full_wheel.py',
         'run_full_wheel.sh', 'package_followup.py']
files += [str(p.relative_to(root)) for p in (root/'runtime-049').glob('published-*')
          if p.is_file()]
files += ['runtime-049/published_moe_probe.py', 'runtime-049/run_published_moe.sh',
          'runtime-049/compare_published.py']
files += [str(p.relative_to(root)) for p in (root/'full-wheel').glob('*')
          if p.is_file() and p.suffix in ('.log', '.json', '.jsonl', '.py', '.sh', '.cmake')]
files += [str(p.relative_to(root)) for p in (root/'full-wheel/dist').glob('*.whl')]
for relative in sorted(set(files)):
    source = root/relative
    if not source.is_file():
        continue
    # Wheel binaries remain local; record a manifest, not a large repository blob.
    if source.suffix == '.whl':
        payload = {'file': source.name, 'bytes': source.stat().st_size,
                   'sha256': hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()}
        target.joinpath('wheel-manifest.json').write_text(json.dumps(payload, indent=2)+'\n')
        continue
    destination = target/relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
build_dir = root/'full-wheel/build-gb10'
if not build_dir.exists():
    build_dir = root/'full-wheel/build-clean'
compile_commands = build_dir/'compile_commands.json'
if compile_commands.exists():
    shutil.copy2(compile_commands, target/'full-wheel/compile_commands.json')
cache = build_dir/'CMakeCache.txt'
if cache.exists():
    shutil.copy2(cache, target/'full-wheel/CMakeCache.txt')
sums = []
for file in sorted(target.rglob('*')):
    if file.is_file() and file.name != 'SHA256SUMS':
        sums.append(hashlib.file_digest(file.open('rb'), 'sha256').hexdigest()
                    +'  '+str(file.relative_to(target)))
target.joinpath('SHA256SUMS').write_text('\n'.join(sums)+'\n')
print(json.dumps({'artifact_dir': str(target), 'hash_entries': len(sums)}), flush=True)
