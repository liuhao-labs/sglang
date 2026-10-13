import importlib.metadata as metadata
import json
import os
import subprocess
from pathlib import Path

import torch

root = Path('/probe/full-wheel')
assert torch.__version__.split('+')[0] == '2.14.1', torch.__version__
pins = json.loads(root.joinpath('source-manifest.json').read_text())
args = ['-DSGL_KERNEL_COMPILE_THREADS=1', '-DENABLE_CCACHE=OFF',
        '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DTorch_DIR=/probe/runtime-049/probe-site/torch/share/cmake/Torch',
        '-DCaffe2_DIR=/probe/runtime-049/probe-site/torch/share/cmake/Caffe2']
args += [f'-DFETCHCONTENT_SOURCE_DIR_{p["name"].upper()}={p["container_path"]}'
         for p in pins['dependencies']]
gb10_only = os.environ.get('PROBE_GB10_ONLY') == '1'
build_dir = '/probe/full-wheel/build-gb10' if gb10_only else '/probe/full-wheel/build-clean'
if gb10_only:
    args.append('-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=/probe/full-wheel/gb10-arch.cmake')
os.environ['CMAKE_ARGS'] = ' '.join(args)
command = ['python3', '-m', 'pip', 'wheel', '--no-index', '--no-deps',
           '--no-build-isolation', '-v', '-w', str(root/'dist'),
           '--config-settings', 'build-dir='+build_dir, str(root/'source')]
payload = {'torch': torch.__version__, 'cuda': torch.version.cuda,
           'torch_path': torch.__file__, 'torch_cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
           'packages': {p: metadata.version(p) for p in ('scikit-build-core', 'cmake', 'ninja', 'wheel')},
           'command': command, 'cmake_args': args, 'source': pins,
           'scope': 'full AOT wheel, all default targets and sources, no test namespace',
           'common_ops_arch': 'SM121a only' if gb10_only else 'default multi-architecture'}
root.joinpath('build-provenance.json').write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps(payload), flush=True)
subprocess.run(command, check=True)
