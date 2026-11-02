"""Reject prediction caches produced with different inputs or analysis code."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform


def analysis_signature(root, inputs, code, parameters):
    root = Path(root)
    sources = list(code) + [p.relative_to(root).as_posix() for p in
                             sorted((root / 'src/measured_vs').rglob('*.py'))]
    return {
        'inputs': {label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for label, path in inputs.items()},
        # Normalize newlines so Git's Windows checkout convention is immaterial.
        'code': {name: hashlib.sha256((root / name).read_text(encoding='utf-8-sig')
                                      .encode('utf8')).hexdigest()
                 for name in sorted(set(sources))},
        'parameters': parameters,
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scikit-learn', 'scipy']},
    }


def protect_cache(output, signature):
    output = Path(output)
    stamp = output / 'cache_signature.json'
    if stamp.exists():
        if json.loads(stamp.read_text(encoding='utf8')) != signature:
            raise ValueError(f'Cache inputs, code, parameters or environment changed: {output}. '
                             'Use --output with a new directory for a fresh run.')
        return
    if output.exists() and any(output.rglob('*predictions.csv')):
        raise ValueError(f'Existing predictions have no cache signature: {output}. '
                         'Use --output with a new directory for a fresh run.')
    output.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(signature, indent=2, sort_keys=True), encoding='utf8')
