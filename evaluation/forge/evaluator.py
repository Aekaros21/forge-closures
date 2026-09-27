"""Real local OpenFOAM evaluation, immutable experiments and matched SST.

All CFD paths, library snapshots, caches and accounting belong to FORGE v2.
Historical endpoints are never imported as newly qualified baselines.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

import numpy as np

from tedp.spec import CandidateSpec, to_json, canonical
from .core import ROOT, BudgetExhausted, Ledger, atomic_json, file_lock, fingerprint, sha256, source_identity, utc_now, within
from .metrics import assess


def set_dictionary_value(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^(\s*){re.escape(key)}\s+[^;\n]+;"
    # Function objects have their own writeControl/writeInterval entries.
    # Count only top-level dictionary entries, preserving nested settings.
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), text)
    matches = [m for m in re.finditer(pattern, text)
               if masked[:m.start()].count('{') == masked[:m.start()].count('}')]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous repeated dictionary key: {key}")
    if matches:
        m = matches[0]
        return text[:m.start()] + f"{m[1]}{key} {value};" + text[m.end():]
    return text + f"\n{key} {value};\n"


def instrument_case(work: Path, protocol: dict, ranks: int) -> None:
    """Retain a long-span comparison and adjacent endpoints; never purge them."""
    end = int(protocol["end_time"])
    adjacent = int(protocol.get("adjacent_window", 5))
    span = int(protocol.get("long_span_steps", max(20, end//4)))
    if end <= max(span, adjacent):
        raise ValueError("Endpoint is too short for the declared qualification windows")
    path = work / "system/controlDict"
    text = path.read_text()
    for key, value in {"startFrom": "startTime", "startTime": "0", "stopAt": "endTime",
                       "endTime": str(end), "deltaT": "1", "writeControl": "timeStep",
                       "writeInterval": str(math.gcd(end, end-span)), "purgeWrite": "0",
                       "writePrecision": "14", "writeFormat": "ascii", "runTimeModifiable": "false"}.items():
        text = set_dictionary_value(text, key, value)
    block = f'''
    forgeQualificationWindow
    {{
        type writeObjects;
        libs (utilityFunctionObjects);
        timeStart {end-adjacent};
        writeControl timeStep;
        writeInterval 1;
        writeOption anyWrite;
        objects ("(U|p|k|omega|nut|nonlinearStress|Rcorr)");
    }}
'''
    if "forgeQualificationWindow" in text:
        raise ValueError("Case already instrumented")
    if re.search(r"\bfunctions\s*\{", text):
        text = re.sub(r"(\bfunctions\s*\{)", lambda m: m[0]+block, text, count=1)
    else:
        text += "\nfunctions\n{\n"+block+"}\n"
    path.write_text(text)
    # An early SIMPLE stop would remove the prescribed comparison window.
    solution = work / "system/fvSolution"
    contents = solution.read_text()
    contents = re.sub(r"\bresidualControl\s*\{[^{}]*\}", "", contents)
    if 'linear_tolerance' in protocol:
        tolerance = float(protocol['linear_tolerance'])
        relative = float(protocol.get('linear_rel_tol', 0.))
        if not math.isfinite(tolerance) or not 0 < tolerance <= 1e-6 or not 0 <= relative < 1:
            raise ValueError('Invalid declared linear-solver tolerances')
        for key, value in (('tolerance',tolerance), ('relTol',relative)):
            contents, count = re.subn(r'\b'+key+r'\s+[+\-\d.eE]+\s*;', f'{key} {value:.16g};', contents)
            if not count:
                raise ValueError(f'Missing linear solver setting {key}; cannot apply matched policy')
    if 'pressure_linear_tolerance' in protocol:
        tolerance = float(protocol['pressure_linear_tolerance'])
        relative = float(protocol.get('pressure_linear_rel_tol',0.))
        if not math.isfinite(tolerance) or not 0 < tolerance <= 1e-6 or not 0 <= relative < 1:
            raise ValueError('Invalid pressure linear-solver policy')
        def pressure_settings(match):
            block=match[0]
            for key,value in (('tolerance',tolerance),('relTol',relative)):
                block,count=re.subn(r'\b'+key+r'\s+[+\-\d.eE]+\s*;',f'{key} {value:.16g};',block)
                if count!=1:raise ValueError(f'Ambiguous pressure setting {key}')
            return block
        contents,count=re.subn(r'(?m)^\s*p\s*\{[^{}]*\}',pressure_settings,contents)
        if count!=1:raise ValueError('One explicit p solver block is required for the pressure policy')
    solution.write_text(contents)
    decomposition = protocol.get("decomposition")
    if decomposition is None:
        method = "method scotch;\n"
    else:
        # Decomposition-independence studies: an explicit simple (n_x n_y n_z) split.
        n = [int(v) for v in decomposition.get("n", [])]
        if decomposition.get("method") != "simple" or len(n) != 3 or min(n) < 1 or math.prod(n) != ranks:
            raise ValueError("Only a simple decomposition whose split matches the MPI ranks is supported")
        method = f"method simple;\nsimpleCoeffs {{ n ({n[0]} {n[1]} {n[2]}); }}\n"
    (work / "system/decomposeParDict").write_text(
        'FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n'
        f'numberOfSubdomains {ranks};\n' + method + 'distributed no;\nroots ();\n')


def _last_time(work: Path) -> str:
    times = [(float(p.name), p.name) for p in work.iterdir()
             if p.is_dir() and re.fullmatch(r"\d+(?:\.\d+)?", p.name)]
    if not times:
        raise ValueError("No solution time directories")
    return max(times)[1]


class Evaluator:
    backend_kind = 'local'

    def _runtime_context(self):
        from . import runtime
        env = runtime.foam_env(self.root)
        library = runtime.library_path(self.root)
        solver = shutil.which('simpleFoam', path=env['PATH'])
        if not solver:
            raise RuntimeError('simpleFoam unavailable')
        return env, library, Path(solver).resolve(), runtime.platform_identity(self.root)

    def __init__(self, root: Path, manifest: dict, contract: dict, config: dict):
        from . import runtime
        self.root = Path(root).resolve()
        self.manifest, self.contract, self.config = copy.deepcopy(manifest), copy.deepcopy(contract), copy.deepcopy(config)
        entries = manifest["cases"]
        self.cases = {c["id"]: c for c in entries}
        if len(self.cases) != len(entries):
            raise ValueError("Duplicate case IDs")
        self.env, self.library, self.solver, self.runtime_identity = self._runtime_context()
        # Scotch's threaded mapper can vary even after randomReset(), changing
        # the partition and GAMG cost between otherwise identical experiments.
        self.env['SCOTCH_PTHREAD_NUMBER'] = '1'
        self.library_sha = sha256(self.library)
        self.code = fingerprint(source_identity(self.root, physical_only=True))
        self.ranks = int(config.get("backend", {}).get("ranks", 4))
        if self.ranks < 1 or (self.backend_kind == 'local' and self.ranks > (os.cpu_count() or 1)):
            raise ValueError("Invalid local MPI rank count")
        if config.get("backend", {}).get("kind", "local") != self.backend_kind:
            raise ValueError('Evaluator and configured backend do not match')
        self.experiments = within(self.root, self.root / "runs/experiments")
        self.campaign = within(self.root, self.root / "results" / config["campaign"])
        self.ledger = Ledger(self.campaign / "resources.json", config["budget"]["core_hours"])
        self.identity = {"schema_version": 2, "manifest": manifest, "contract": contract,
                         "config": config, "source_hash": self.code,
                         "library_sha256": self.library_sha, "simpleFoam_sha256": sha256(self.solver),
                         "wm_options": self.env.get("WM_OPTIONS"), "ranks": self.ranks}
        if self.runtime_identity is not None:
            self.identity['runtime_identity'] = self.runtime_identity
        self.protocol_hash = fingerprint(self.identity)
        self.campaign.mkdir(parents=True, exist_ok=True)
        protocol_path = self.campaign / "protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text())["fingerprint"] != self.protocol_hash:
                raise ValueError("Campaign protocol changed; use a new campaign name")
        else:
            atomic_json(protocol_path, {"fingerprint": self.protocol_hash, "definition": self.identity})
        self.numerics_path = self.campaign / "qualified_numerics.json"
        self.qualified_numerics = json.loads(self.numerics_path.read_text()) if self.numerics_path.exists() else {}

    def effective_protocol(self, case: dict, stage: str) -> dict:
        protocol = {**case["protocol"], **self.config.get("numerics", {})}
        protocol.update(self.config.get("case_protocol_overrides", {}).get(case["id"], {}))
        protocol.update(self.qualified_numerics.get(case["id"], {}).get("protocol", {}))
        stage_protocol = self.config.get("stages", {}).get(stage, {}).get("protocol", {})
        protocol.update(stage_protocol)
        if "end_time" in stage_protocol and "long_span_steps" not in stage_protocol:
            protocol["long_span_steps"] = max(20, int(protocol["end_time"])//4)
        if "iterations" in protocol and "end_time" not in protocol:
            protocol["end_time"] = protocol["iterations"]
        protocol["end_time"] = int(protocol["end_time"])
        protocol.setdefault("long_span_steps", max(20, protocol["end_time"]//4))
        protocol.setdefault("adjacent_window", 5)
        protocol.setdefault("timeout_s", 7200)
        return protocol

    def final_access(self, spec) -> bool:
        """Whether this evaluator may run final-test cases for ``spec`` (None is stock SST).

        Only a verification campaign may, and only for the models of its freeze record. The
        record is named with its SHA-256 in the configuration, which is part of the campaign's
        protocol identity, so the models allowed to see final cases are fixed before any runs."""
        if self.config.get("purpose") != "verification":
            return False
        freeze = self.config.get("verification", {}).get("freeze")
        if not freeze:
            return False
        path = within(self.root, self.root / freeze["path"])
        if sha256(path) != freeze["sha256"]:
            raise RuntimeError("Verification freeze record changed")
        record = json.loads(path.read_text())
        if record["manifest_fingerprint"] != fingerprint(self.manifest) or \
                record["contract_fingerprint"] != fingerprint(self.contract):
            raise RuntimeError("Verification freeze record names a different manifest or contract")
        if spec is None:
            return True
        model = json.loads(to_json(canonical(spec)))
        model.pop("name", None)
        return fingerprint(model) in {m["model_fingerprint"] for m in record["models"]}

    def case_identity(self, spec, case, protocol, ranks):
        pairing = {'case':case, 'protocol':protocol, 'library':self.library_sha,
                   'solver':self.identity['simpleFoam_sha256'], 'source':self.code,
                   'ranks':ranks, 'initialization':'case-declared common initial fields',
                   'environment':{k:self.env.get(k) for k in
                                  ('WM_OPTIONS','WM_PROJECT_VERSION','SCOTCH_PTHREAD_NUMBER')}}
        if self.runtime_identity is not None:
            pairing['runtime_identity'] = self.runtime_identity
        model = json.loads(to_json(canonical(spec))) if spec else None
        if model is not None:
            model.pop('name', None)
        identity = {**pairing, 'spec':model}
        return pairing, identity, fingerprint(identity)

    def _cached_valid(self, result: dict, identity: dict) -> bool:
        if result.get("identity") != identity:
            return False
        for path, digest in result.get("artifacts", {}).items():
            p = Path(path)
            if not p.is_file() or sha256(p) != digest:
                return False
        return bool(result.get("artifacts"))

    def run_case(self, spec: CandidateSpec | None, cid: str, stage: str = "development",
                 *, protocol_override: dict | None = None) -> dict:
        from . import cases as adapters, qualification, runtime
        if cid not in self.cases:
            raise ValueError(f"Unknown case {cid}")
        case = self.cases[cid]
        # On a compute node the one-case worker carries the controller's recorded grant.
        if case["role"] in ("final", "locked_final", "test") and not (
                self.config.get("final_access_granted") or self.final_access(spec)):
            raise PermissionError("Discovery cannot evaluate final-test cases")
        protocol = self.effective_protocol(case, stage)
        if protocol_override:
            if spec is not None:
                raise ValueError("Numerical protocol selection must use SST alone")
            protocol.update(protocol_override)
        ranks = min(self.ranks, int(protocol.get("nprocs", self.ranks)))
        if ranks < 1:
            raise ValueError("Case MPI rank count must be positive")
        pairing, identity, key = self.case_identity(spec, case, protocol, ranks)
        pairing_hash = fingerprint(pairing)
        folder = within(self.root, self.experiments / key)
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        with file_lock(folder / ".lock"):
            if sha256(self.library) != self.library_sha:
                raise RuntimeError("Immutable solver library changed")
            if result_path.exists():
                checksum = folder/"result.sha256"
                if not checksum.exists() or checksum.read_text().strip() != sha256(result_path):
                    raise RuntimeError(f"Cached result checksum failure: {folder}")
                cached = json.loads(result_path.read_text())
                if not self._cached_valid(cached, identity):
                    raise RuntimeError(f"Cached experiment integrity failure: {folder}")
                return {**cached, "cache_hit": True, "charged_core_hours": 0.0}
            attempts = sorted(folder.glob("attempt-*"))
            # An interrupted attempt is never reused as if complete. Preserve
            # all its files and the reserved cost; start from identical inputs.
            for old in attempts:
                accounting_id = key+":"+old.name
                row = self.ledger.snapshot()["jobs"].get(accounting_id)
                if row and row["status"] == "running":
                    self.ledger.settle(accounting_id, row["reservation_core_hours"], "interrupted",
                                       {"charge": "full reservation; interrupted cost unresolved"})
            attempt = folder / f"attempt-{len(attempts)+1:03d}"
            attempt.mkdir()
            work = attempt / "case"
            # One aggregate allocation covers solver, utilities and extraction.
            total_timeout = float(protocol.get("job_wall_timeout_s", float(protocol["timeout_s"])+4200))
            reservation = total_timeout*ranks/3600
            job_id = key+":"+attempt.name
            self.ledger.reserve(job_id, reservation, {"case": cid, "experiment": key},
                                protected_reserve=float(self.config["budget"].get("confirmation_reserve_core_hours", 0)))
            commands, start = [], time.monotonic()
            result = {"schema_version": 2, "case_id": cid, "identity": identity,
                      "pairing_hash": pairing_hash, "experiment_hash": key, "workdir": str(work),
                      "stage": stage, "started_utc": utc_now(), "errors": {}, "observables": {},
                      "uncertainty": {}, "artifacts": {}, "cache_hit": False}
            def command(args, name, timeout):
                remaining = total_timeout - (time.monotonic()-start)
                if remaining <= 0:
                    raise TimeoutError("Aggregate case time allowance exhausted")
                outcome = runtime.run_command(args, cwd=work, log=work/name,
                                              timeout=min(timeout, remaining), env=self.env)
                commands.append(outcome)
                if outcome["returncode"]:
                    raise RuntimeError(f"{name} exited {outcome['returncode']}")
                return outcome
            try:
                effective_case = {**case, "protocol": {**protocol, "nprocs": ranks}}
                metadata = adapters.prepare_case(effective_case, spec, work, end_time=protocol["end_time"], library=self.library)
                result["preparation"] = metadata
                # All mutable files, including meshes, must resolve inside v2.
                for p in work.rglob("*"):
                    if p.is_symlink() and not p.resolve().is_relative_to(self.root):
                        raise ValueError(f"Generated case links outside v2: {p}")
                instrument_case(work, protocol, ranks)
                control = (work/"system/controlDict").read_text()
                if spec and str(self.library) not in control:
                    raise ValueError("Candidate does not load the immutable v2 library")
                if not (work/"constant/polyMesh/points").exists():
                    command(["blockMesh"], "log.blockMesh", 300)
                    moved = adapters.post_mesh(case, work)
                    if moved is not None:
                        result["post_mesh"] = moved
                end = protocol["end_time"]
                selected_times = [str(t) for t in sorted({end-protocol["long_span_steps"],
                                  *range(end-int(protocol["adjacent_window"]), end+1)})]
                # Large 3D verification cases keep only their final state, on the cluster.
                retention = protocol.get("field_retention", "all")
                if retention not in ("all", "final_state_remote"):
                    raise ValueError(f"Unknown field retention policy {retention}")
                slim = retention == "final_state_remote"
                if ranks > 1:
                    command(["decomposePar", "-force"], "log.decomposePar", 300)
                    execution = command(["mpirun", "--use-hwthread-cpus", "-np", str(ranks),
                                         str(self.solver), "-parallel"], "log.simpleFoam", protocol["timeout_s"])
                    if slim:
                        command(["reconstructPar", "-time", ",".join(selected_times)], "log.reconstructPar", 1800)
                        for p in work.glob("processor*"):
                            shutil.rmtree(p)
                    else:
                        command(["reconstructPar"], "log.reconstructPar", 600)
                else:
                    execution = command([str(self.solver)], "log.simpleFoam", protocol["timeout_s"])
                atomic_json(work/"execution.json", execution)
                for function in metadata.get("required_postprocess", ["grad(U)"]):
                    label = re.sub(r"[^A-Za-z0-9]", "", function)
                    command([str(self.solver), "-postProcess", "-func", function], "log."+label,
                            1800 if slim else 600)
                scores = []
                for t in selected_times:
                    if time.monotonic()-start >= total_timeout:
                        raise TimeoutError("Aggregate case time allowance exhausted during extraction")
                    scored = adapters.score_case(case, work, time=t)
                    scores.append({"time": float(t), **scored})
                result.update(scores[-1])
                result["history"] = scores
                # Temporal error/observable drift is retained with an honest
                # scope label. It is not called mesh or reference uncertainty.
                for field in ("errors", "observables"):
                    for name, value in result[field].items():
                        if isinstance(value, (float, int)) and math.isfinite(value):
                            history = [s[field].get(name) for s in scores]
                            if all(isinstance(v, (float, int)) and math.isfinite(v) for v in history):
                                result["uncertainty"][name] = max(abs(v-value) for v in history)
                        elif isinstance(value, list):
                            try:
                                arrays = [np.asarray(s[field].get(name), dtype=float) for s in scores]
                            except (ValueError, TypeError):
                                continue
                            if all(a.shape == arrays[-1].shape and np.isfinite(a).all() for a in arrays):
                                result["uncertainty"][name] = max(float(np.max(np.abs(a-arrays[-1]))) for a in arrays)
                records = []
                for s in scores:
                    values = {}
                    for name,value in s["observables"].items():
                        if isinstance(value,(int,float,list)):
                            try:
                                a = np.asarray(value,dtype=float)
                            except (ValueError, TypeError):
                                continue
                            if np.isfinite(a).all(): values[name] = value
                    records.append({"time":s["time"],"values":values})
                atomic_json(work/"observables.json", {"records": records})
                result["qualification"] = qualification.qualify_run(work, {**protocol, "execution": execution})
                if case["adapter"] in ("plate", "bump", "naca"):
                    parity = all(s["observables"].get("force_history_parity_passed") is True for s in scores)
                    result["qualification"]["checks"]["independent_surface_force_parity"] = parity
                    if not parity:
                        result["qualification"].update(qualified=False,status="unqualified")
                        result["qualification"]["reasons"].append("Surface integration does not match retained force-coefficient history")
                result["status"] = "complete" if result["qualification"]["qualified"] else "unqualified"
                for p in [work/"log.simpleFoam", work/"execution.json", work/"system/controlDict",
                          work/"system/fvSchemes",work/"system/fvSolution",work/"system/decomposeParDict",
                          work/"constant/turbulenceProperties", work/"observables.json"]:
                    result["artifacts"][str(p)] = sha256(p)
                if slim:
                    # The qualification window and every score are recorded above; the cluster
                    # keeps the final state (without the derivable velocity gradient) and the
                    # transfer brings back only the small evidence files.
                    for p in work.iterdir():
                        if p.is_dir() and re.fullmatch(r"\d+(?:\.\d+)?", p.name) and p.name not in ("0", selected_times[-1]):
                            shutil.rmtree(p)
                    (work/selected_times[-1]/"grad(U)").unlink(missing_ok=True)
                    result["remote_artifacts"] = {}
                for t in (selected_times[-1:] if slim else selected_times):
                    for name in ("U", "p", "k", "omega", "nut", "grad(U)", "wallShearStress", "nonlinearStress", "Rcorr"):
                        p = work / t / name
                        if p.exists():
                            result["remote_artifacts" if slim else "artifacts"][str(p)] = sha256(p)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                result["status"] = "failed"
                result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                result["qualification"] = {"qualified": False, "status": "failed", "reasons": [str(exc)]}
                for p in work.glob("log.*") if work.exists() else []:
                    result["artifacts"][str(p)] = sha256(p)
            finally:
                elapsed = time.monotonic()-start
                charged = elapsed*ranks/3600
                # Python postprocessing uses capped threads in the launcher;
                # charging the allocation throughout is conservative.
                self.ledger.settle(job_id, charged, result.get("status", "interrupted"))
            result.update(commands=commands, wall_s=elapsed, core_hours=charged,
                          charged_core_hours=charged, finished_utc=utc_now())
            if not result["artifacts"]:
                atomic_json(attempt/"failure.json", result.get("failure", {}))
                result["artifacts"][str(attempt/"failure.json")] = sha256(attempt/"failure.json")
            atomic_json(result_path, result)
            (folder/"result.sha256").write_text(sha256(result_path)+"\n")
            return result

    def qualify_baselines(self, case_ids: list[str], factors=(1, 2, 4)) -> dict:
        """Predeclared endpoint ladder selected on SST numerical evidence only."""
        from .admission import admission, baseline_agreement, is_airfoil_control
        outcomes = {}
        for cid in case_ids:
            if cid in self.qualified_numerics:
                outcomes[cid] = self.run_case(None, cid, "development")
                continue
            base = self.effective_protocol(self.cases[cid], "development")
            previous = None
            for factor in factors:
                if factor < 1:
                    raise ValueError("Baseline endpoint factors must be >= 1")
                selected = {"end_time": int(base["end_time"]*factor),
                            "long_span_steps": int(base["long_span_steps"]*factor)}
                print(f"SST qualification: {cid}, {selected['end_time']} iterations", flush=True)
                record = self.run_case(None, cid, "development", protocol_override=selected)
                outcomes[cid] = record
                verdict = admission(record, self.cases[cid])
                agreement = None
                ready = verdict['admitted']
                if is_airfoil_control(self.cases[cid]):
                    agreement = baseline_agreement(previous, record, self.cases[cid]) if previous else None
                    ready = ready and bool(agreement and agreement['passed'])
                if ready:
                    self.qualified_numerics[cid] = {"protocol": selected,
                        "baseline_experiment": record["experiment_hash"], "selection": "SST numerical evidence only",
                        "declared_ladder": list(factors), 'admission': verdict,
                        'control_baseline_agreement': agreement}
                    atomic_json(self.numerics_path, self.qualified_numerics)
                    break
                # A failed intermediate endpoint breaks the control sequence;
                # two apparently stable endpoints cannot straddle a failure.
                previous = record if verdict['admitted'] else None
                if record["status"] == "failed":
                    break  # An integration/solver failure is not fixed by more iterations.
            if cid not in self.qualified_numerics:
                print(f"SST qualification unresolved: {cid}", flush=True)
        return {"qualified": all(r["qualification"]["qualified"] for r in outcomes.values()),
                'admitted': all(cid in self.qualified_numerics and admission(r,self.cases[cid])['admitted']
                                for cid,r in outcomes.items()),
                "cases": outcomes, "numerics": self.qualified_numerics}

    def __call__(self, spec: CandidateSpec | None, case_ids: list[str], stage: str) -> dict:
        from .admission import admission, is_airfoil_control
        if len(set(case_ids)) != len(case_ids) or not case_ids:
            raise ValueError("Evaluation needs a nonempty unique case list")
        results, baselines = {}, {}
        for cid in case_ids:
            baselines[cid] = self.run_case(None, cid, stage)
            if spec is None:
                results[cid] = baselines[cid]
            elif (admission(baselines[cid], self.cases[cid])['admitted'] and
                  (not is_airfoil_control(self.cases[cid]) or
                   self.qualified_numerics.get(cid,{}).get('control_baseline_agreement',{}).get('passed'))):
                results[cid] = self.run_case(spec, cid, stage)
            else:
                results[cid] = {"qualification": {"qualified": False},
                                "pairing_hash": baselines[cid]["pairing_hash"],
                                "status": "unqualified", "reason": "SST baseline lacks required admission/endpoint agreement"}
            for group in (results,baselines):
                group[cid] = {**group[cid], 'development_admission': admission(group[cid],self.cases[cid])}
        assessment = assess([self.cases[c] for c in case_ids], results, baselines, self.contract)
        status = ("failed" if any(r["status"] == "failed" for r in results.values()) else
                  "complete" if all(r['development_admission']['admitted'] for r in results.values()) else "unqualified")
        charged = sum(r.get("charged_core_hours", 0) for r in results.values())
        if spec is not None:
            charged += sum(r.get("charged_core_hours", 0) for r in baselines.values())
        return {"status": status, "case_results": results, "baselines": baselines,
                "assessment": assessment, "cost": {"core_hours": charged},
                "case_evaluations": sum(not r.get("cache_hit", True) for r in results.values())}
