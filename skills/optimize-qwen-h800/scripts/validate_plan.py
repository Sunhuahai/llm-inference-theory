#!/usr/bin/env python3
"""Check an experiment record, without running code or trusting its evidence.

Exit 0: the supplied record is structurally consistent.
Exit 2: missing fields, archived task, or a conflicting invariant.
No network, GPU calls, file writes, or source-code execution are performed.
"""
import argparse
import json
from pathlib import Path
import re
import sys

FIELDS = (
    'model_id', 'model_revision', 'tokenizer_revision', 'chat_template_sha256',
    'gpu_model', 'gpu_count', 'topology_id', 'parallel_layout', 'engine_base_sha',
    'weight_format', 'weight_scale_contract', 'activation_dtype', 'compute_dtype',
    'kv_cache_dtype', 'indexer_cache_dtype', 'ssm_state_dtype', 'ple_embedding_dtype',
    'matmul_math_mode', 'accumulation_contract', 'workload_id', 'workload_sha256',
    'sampling_contract', 'output_stop_contract', 'cache_mode', 'graph_mode',
    'mtp_mode', 'thinking_mode', 'feature_contract', 'load_contract',
)
PROTOTYPE_GATES = (
    'card_preconditions_met', 'code_path_confirmed', 'baseline_reused',
    'precision_contract_pass', 'rollback_available',
)
BENCHMARK_GATES = ('correctness_pass', 'candidate_observed', 'request_semantics_pass')
PROTOTYPE_EVIDENCE = ('environment', 'baseline', 'profile', 'source_audit', 'rollback')
BENCHMARK_EVIDENCE = ('correctness', 'candidate_environment')
UNKNOWN = {'', 'unknown', 'todo', 'tbd', 'null', 'none', '待填写', '待补', '未知'}


def missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in UNKNOWN
    if isinstance(value, dict):
        return not value or any(missing(v) for v in value.values())
    if isinstance(value, list):
        return not value or any(missing(v) for v in value)
    return False


def validate(plan, root, phase):
    errors = []
    if not isinstance(plan, dict):
        return ['计划必须是一个 JSON object。']
    if type(plan.get('schema_version')) is not int or plan.get('schema_version') != 1:
        errors.append('schema_version 必须为 1。')
    tid = plan.get('task_id', '')
    if not isinstance(tid, str) or not re.fullmatch(r'T\d{3}', tid):
        errors.append('task_id 必须是一个具体 T 编号；R 工作流先选择单个阶段和对应 T 编号。')
    elif int(tid[1:]) <= 13:
        errors.append('T001–T013 已归档；重访历史实验须另列经用户明确要求的新计划。')
    card_value = plan.get('card_path')
    if not isinstance(card_value, str) or missing(card_value):
        errors.append('card_path 缺失。')
    else:
        card = (root / card_value).resolve()
        try:
            rel = card.relative_to(root)
        except ValueError:
            errors.append('card_path 必须位于研究仓库内。')
        else:
            if 'archive' in rel.parts or not card.is_file():
                errors.append('card_path 不存在或指向归档目录。')
            elif not card.name.startswith(str(tid) + '-'):
                errors.append('card_path 与 task_id 不匹配。')
    for key in ('implementation_repo', 'baseline_run_id'):
        if not isinstance(plan.get(key), str) or missing(plan.get(key)):
            errors.append(f'{key} 缺失或不是文本。')
    change = plan.get('single_variable')
    if not isinstance(change, list) or len(change) != 1 or not isinstance(change[0], dict):
        errors.append('single_variable 必须恰好有一个变更。')
    else:
        for key in ('name', 'before', 'after'):
            if missing(change[0].get(key)):
                errors.append(f'single_variable[0].{key} 缺失。')
        if not missing(change[0].get('before')) and change[0].get('before') == change[0].get('after'):
            errors.append('唯一变量的 before 与 after 不能相同。')
    invariants = plan.get('invariants', {})
    if not isinstance(invariants, dict):
        invariants = {}
    baseline = invariants.get('baseline', {})
    candidate = invariants.get('candidate', {})
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        errors.append('invariants.baseline 与 candidate 必须为 object。')
        baseline, candidate = {}, {}
    if set(baseline) != set(candidate):
        errors.append('baseline 与 candidate 的 invariant 字段集合不同。')
    for key in set(FIELDS) | set(baseline) | set(candidate):
        for label, values in (('baseline', baseline), ('candidate', candidate)):
            if missing(values.get(key)):
                errors.append(f'invariants.{label}.{key} 缺失。')
            elif key in FIELDS and key != 'gpu_count' and not isinstance(values.get(key), str):
                errors.append(f'invariants.{label}.{key} 必须为明确的文本合同。')
        if baseline.get(key) != candidate.get(key):
            errors.append(f'invariant 改变：{key}。')
    for label, values in (('baseline', baseline), ('candidate', candidate)):
        gpu_count = values.get('gpu_count')
        if not missing(gpu_count) and (type(gpu_count) is not int or gpu_count <= 0):
            errors.append(f'{label}.gpu_count 必须为实际正整数。')
        if not missing(values.get('gpu_model')) and 'H800' not in str(values.get('gpu_model', '')):
            errors.append(f'{label}.gpu_model 未确认 H800。')
        for key in ('chat_template_sha256', 'workload_sha256'):
            value = values.get(key)
            if not missing(value) and (not isinstance(value, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', value)):
                errors.append(f'{label}.{key} 需要完整 SHA-256。')
        value = values.get('engine_base_sha')
        if not missing(value) and (not isinstance(value, str) or not re.fullmatch(r'(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})', value)):
            errors.append(f'{label}.engine_base_sha 需要完整 Git SHA。')
    gates = plan.get('gates', {})
    evidence = plan.get('evidence', {})
    if not isinstance(gates, dict):
        gates = {}
    if not isinstance(evidence, dict):
        evidence = {}
    gate_keys = PROTOTYPE_GATES + (BENCHMARK_GATES if phase == 'benchmark' else ())
    evidence_keys = PROTOTYPE_EVIDENCE + (BENCHMARK_EVIDENCE if phase == 'benchmark' else ())
    for key in gate_keys:
        if gates.get(key) is not True:
            errors.append(f'gates.{key} 未通过；只在有证据时填写 true。')
    for key in evidence_keys:
        if missing(evidence.get(key)):
            errors.append(f'evidence.{key} 缺少证据引用。')
    if phase == 'benchmark':
        commit = plan.get('candidate_commit')
        if not isinstance(commit, str) or not re.fullmatch(r'(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})', commit):
            errors.append('benchmark 前需要完整 candidate_commit。')
    return sorted(set(errors))


def summarize_errors(errors):
    """Group repeated missing-contract fields for a compact actionable report."""
    rest, fields, gates, evidence = [], {}, [], []
    for error in errors:
        match = re.fullmatch(r'invariants\.(baseline|candidate)\.(\w+) 缺失。', error)
        if match:
            fields.setdefault(match.group(1), []).append(match.group(2))
        elif error.startswith('gates.') and ' 未通过' in error:
            gates.append(error.split(' ', 1)[0].removeprefix('gates.'))
        elif error.startswith('evidence.') and ' 缺少证据引用' in error:
            evidence.append(error.split(' ', 1)[0].removeprefix('evidence.'))
        else:
            rest.append(error)
    if fields.get('baseline') and fields.get('baseline') == fields.get('candidate'):
        rest.append('baseline/candidate 共同缺失合同字段：' + ', '.join(fields['baseline']))
    else:
        for label, names in sorted(fields.items()):
            rest.append(label + ' 缺失合同字段：' + ', '.join(names))
    if gates:
        rest.append('尚无通过证据的门槛：' + ', '.join(gates))
    if evidence:
        rest.append('需要已有证据引用：' + ', '.join(evidence))
    return rest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--phase', choices=('prototype', 'benchmark'), required=True)
    parser.add_argument('--verbose', action='store_true', help='逐字段显示全部待补项')
    parser.add_argument('--research-root', type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    try:
        with args.plan.open(encoding='utf-8') as handle:
            plan = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f'BLOCKED: 无法读取计划：{exc}', file=sys.stderr)
        return 2
    errors = validate(plan, args.research_root.resolve(), args.phase)
    if errors:
        print('BLOCKED: 记录尚未满足结构门槛。')
        for error in (errors if args.verbose else summarize_errors(errors)):
            print(f'- {error}')
        print('复用已有运行记录补齐；不要为通过检查编造值。需要逐字段明细时加 --verbose。')
        return 2
    print('PASS_STRUCTURAL: 所填计划字段一致；未验证证据真实性、实际代码或 GPU 结果。')
    print('继续执行任务卡对应阶段；不得因此标记 validated。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
