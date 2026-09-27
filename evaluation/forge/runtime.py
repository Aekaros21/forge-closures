"""OpenFOAM execution confined to this FORGE checkout's writable namespace."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import threading
import shlex

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASHRC = Path('/usr/lib/openfoam/openfoam2312/etc/bashrc')


_ACTIVE_GROUPS = set()
_GROUP_LOCK = threading.RLock()
_HANDLERS_INSTALLED = False


def _stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # Surviving descendants must be killed even if the group leader exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def install_signal_handlers():
    """Install controller cleanup once, from the main thread before workers."""
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED or threading.current_thread() is not threading.main_thread():
        return
    def stop(signum, frame):
        with _GROUP_LOCK:
            active = list(_ACTIVE_GROUPS)
        for proc in active:
            _stop_group(proc)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    _HANDLERS_INSTALLED = True


def _root(root: Path | str) -> Path:
    root = Path(root).resolve()
    if root != PROJECT_ROOT:
        raise ValueError(f'Runtime root must be this isolated checkout: {PROJECT_ROOT}')
    return root


def _within(path: Path | str, root: Path | None = None) -> Path:
    root = PROJECT_ROOT if root is None else root
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Refusing writable path outside FORGE checkout: {path}')
    return path


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def foam_env(root: Path | str) -> dict[str, str]:
    root = _root(root)
    install_signal_handlers()
    # Do not inherit a previously sourced OpenFOAM or library search path.
    clean = {key: os.environ[key] for key in ('HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL',
             'PBS_JOBID', 'PBS_NODEFILE', 'PBS_NP', 'PBS_O_WORKDIR', 'PBS_ENVIRONMENT') if key in os.environ}
    clean['PATH'] = '/usr/local/bin:/usr/bin:/bin'
    profile_path = root/'runtime/platform.json'
    script = 'source /usr/lib/openfoam/openfoam2312/etc/bashrc >/dev/null 2>&1 && env -0'
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        if profile.get('kind') != 'hx1_native' or profile.get('version') != '2312':
            raise RuntimeError('Unsupported explicit runtime platform profile')
        script = 'set -e\nsource '+shlex.quote(profile['module_init'])+' >/dev/null 2>&1 || exit 1\nmodule purge >/dev/null 2>&1\n'
        script += 'module load '+' '.join(shlex.quote(x) for x in profile['modules'])+' >/dev/null 2>&1\n'
        script += 'export FOAM_CONFIG_MODE=o\nsource '+shlex.quote(profile['bashrc'])+' WM_MPLIB=SYSTEMOPENMPI FOAM_MODULE_PREFIX=none >/dev/null 2>&1 || exit 1\nenv -0'
    result = subprocess.run(['bash', '--noprofile', '--norc', '-c', script],
                            env=clean, capture_output=True, check=True, timeout=30)
    env = dict(item.decode().split('=', 1) for item in result.stdout.split(b'\0') if b'=' in item)
    expected_foam = Path(profile['bashrc']).parent.parent if profile_path.exists() else BASHRC.parent.parent
    if env.get('WM_PROJECT_DIR') != str(expected_foam) or env.get('WM_DIR') != str(expected_foam/'wmake'):
        raise RuntimeError('OpenFOAM environment does not match the explicitly selected installation')
    if env.get('WM_PROJECT_VERSION', '').lstrip('v') != '2312' or env.get('WM_OPTIONS') != 'linux64GccDPInt32Opt':
        raise RuntimeError('Unexpected OpenFOAM build configuration')
    for key, relative in {
        'WM_PROJECT_USER_DIR': 'runtime/openfoam', 'FOAM_USER_APPBIN': 'runtime/bin',
        'FOAM_USER_LIBBIN': 'runtime/lib', 'FOAM_RUN': 'runs',
        'FOAM_JOB_DIR': 'runtime/jobControl', 'TMPDIR': 'runtime/tmp',
    }.items():
        target = _within(root / relative, root)
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    # Drop historical user search paths which the bashrc constructed.
    user_base = str(Path(env.get('HOME', '/nonexistent')) / 'OpenFOAM')
    for key in ('PATH', 'LD_LIBRARY_PATH'):
        env[key] = ':'.join(p for p in env.get(key, '').split(':') if p and not p.startswith(user_base))
    env['PATH'] = str(root / 'runtime/bin') + ':' + env['PATH']
    env['LD_LIBRARY_PATH'] = str(root / 'runtime/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def platform_identity(root: Path | str) -> dict | None:
    """Bind non-default environment configuration and native dependencies."""
    root = _root(root)
    path = root/'runtime/platform.json'
    if not path.exists():
        return None
    profile = json.loads(path.read_text())
    files = [profile['bashrc'], profile['module_init'], *profile.get('identity_files', [])]
    return {'profile':profile, 'profile_sha256':sha256(path),
            'files_sha256':{name:sha256(name) for name in files}}


def run_command(args, cwd, log, timeout, env) -> dict:
    install_signal_handlers()
    cwd = _within(cwd)
    log = _within(log)
    if not cwd.is_dir():
        raise ValueError(f'Command cwd does not exist: {cwd}')
    if not args or isinstance(args, str):
        raise ValueError('Commands must be nonempty argument lists; no shell interpolation')
    if timeout <= 0:
        raise ValueError('Timeout must be positive')
    for key in ('FOAM_USER_APPBIN', 'FOAM_USER_LIBBIN', 'FOAM_JOB_DIR', 'FOAM_RUN', 'TMPDIR'):
        if key in env:
            _within(env[key])
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    timed_out = False
    with log.open('w') as handle:
        proc = subprocess.Popen([str(a) for a in args], cwd=cwd, env=env,
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        with _GROUP_LOCK:
            _ACTIVE_GROUPS.add(proc)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_group(proc)
            code = proc.returncode
        except BaseException:
            _stop_group(proc)
            raise
        finally:
            with _GROUP_LOCK:
                _ACTIVE_GROUPS.discard(proc)
    return {'args': [str(a) for a in args], 'cwd': str(cwd), 'log': str(log),
            'returncode': code, 'timed_out': timed_out, 'wall_seconds': time.time() - started,
            'wall_s': time.time() - started, 'started_unix': started, 'finished_unix': time.time(), 'log_sha256': sha256(log)}


def _source_files(source):
    return [p for p in sorted(source.rglob('*')) if p.is_file()
            and not any(q in ('lnInclude', 'linux64GccDPInt32Opt') for q in p.parts)
            and p.suffix not in ('.o', '.dep') and not p.name.startswith('log.')]


def build_solver(root: Path | str) -> dict:
    root = _root(root)
    source = root / 'src/kOmegaSSTBasis'
    env = foam_env(root)
    sources = {str(p.relative_to(source)): sha256(p) for p in _source_files(source)}
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    build = root / 'runtime/builds' / (source_hash[:16] + '-' + str(time.time_ns()))
    copied = build / 'src'
    for rel in sources:
        target = copied / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, target)
    files = copied / 'Make/files'
    text = files.read_text()
    if 'libkOmegaSSTBasisValidated' not in text:
        raise RuntimeError('Unexpected source library identity')
    files.write_text(text.replace('libkOmegaSSTBasisValidated', 'libFORGEv2Closure'))
    output = build / 'output'
    output.mkdir()
    env['FOAM_USER_LIBBIN'] = str(output)
    run = run_command(['wmake', 'libso'], copied, build / 'build.log', 1800, env)
    if run['returncode'] != 0 or run['timed_out']:
        raise RuntimeError(f'Solver build failed; see {run["log"]}')
    lib = output / 'libFORGEv2Closure.so'
    binary_hash = sha256(lib)
    frozen = root / 'runtime/lib' / f'libFORGEv2Closure-{binary_hash[:16]}.so'
    if frozen.exists() and sha256(frozen) != binary_hash:
        raise RuntimeError('Immutable library hash collision')
    if not frozen.exists():
        shutil.copy2(lib, frozen)
        frozen.chmod(0o444)
    report = {'schema_version': 1, 'qualified_build': True, 'library': str(frozen),
              'library_sha256': binary_hash, 'source_sha256': source_hash,
              'sources': sources, 'openfoam_version': env['WM_PROJECT_VERSION'],
              'wm_options': env['WM_OPTIONS'], 'build': run,
              'source_tree': str(copied), 'isolation_root': str(root)}
    path = root / 'runtime/build_manifest.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(path)
    return report


def library_path(root: Path | str) -> Path:
    root = _root(root)
    report = json.loads((root / 'runtime/build_manifest.json').read_text())
    path = _within(report['library'], root)
    if not report.get('qualified_build') or sha256(path) != report['library_sha256']:
        raise RuntimeError('Compiled library is missing or does not match its immutable manifest')
    return path
