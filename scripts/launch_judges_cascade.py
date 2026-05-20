"""Launch Tier-1 full-set 2-judge cascade (Qwen + InternVL inline).

Stage 1 only (this script):
  - Read filter jsonls directly per ds (no separate manifest stage)
  - Inline render yellow GT + red pred per record + send to VLM
  - Output: judge_{model}_{ds}_shard{NNN}.jsonl per shard

Stage 2 (later, separate script after stage 1 done):
  - Find Qwen ↔ InternVL disagreements → GPT-5 arbitrate

Resource budget = 224 GPU = 56 tasks (4 GPU/task):
  ai2/webolmo:        128 GPU = 32 tasks
  ai2/video-olmo-data: 64 GPU = 16 tasks
  ai2/webolmo-eval:    32 GPU =  8 tasks
"""
import argparse
import datetime
import os
import subprocess
import tempfile

SCRIPT_JUDGE = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data/judge_inline.py'
FILTER_ROOT  = '/weka/oe-training-default/oe-encoder/grounding_filter_molmo2'
JUDGE_OUT    = '/weka/oe-training-default/oe-encoder/judge_full'
HF_CACHE     = '/weka/oe-training-default/zixianm/yinuoy/.cache/huggingface'

# (ds, filter_dir, schema, qwen_shards, intern_shards)
# Pool sizes (strict F1<1 rescored):
# (ds, filter_dir, schema, qwen_shards, intern_shards, workspace, budget)
# ONE Beaker experiment per dataset → independent kill / inspect / monitor.
TIER1_PLAN = [
    ('vg',         'vg_filtered',         'rescored',  1,  1, 'ai2/webolmo-eval',    'ai2/oe-omai'),  # 61k
    ('seeclick',   'seeclick_filtered',   'new',       1,  1, 'ai2/webolmo-eval',    'ai2/oe-omai'),  # 73k
    ('rf100',      'rf100_filtered',      'new',       1,  1, 'ai2/webolmo-eval',    'ai2/oe-omai'),  # 127k
    ('openimages', 'openimages_filtered', 'rescored',  7,  9, 'ai2/video-olmo-data', 'ai2/oe-omai'),  # 523k
    ('pixmo',      'pixmo_filtered',      'rescored', 14, 18, 'ai2/webolmo',         'ai2/oe-omai'),  # 1.4M
]
# Total: 54 tasks × 4 GPU = 216 GPU. webolmo 32 / video-olmo-data 16 / webolmo-eval 6.

# Tier 2A: cc3m + cc12m (~5.2M). Launch AFTER Tier 1 frees GPU.
TIER2A_PLAN = [
    ('cc12m',      'cc12m_filtered',       'new',     9, 12, 'ai2/video-olmo-data', 'ai2/oe-omai'),  # 1.9M
    ('cc3m',       'cc3m_filtered_boost',  'new',    15, 20, 'ai2/webolmo',         'ai2/oe-omai'),  # 3.3M
]

# Tier 2B: grit_v2 (~10.3M) — needs distill 7B judge first, infeasible w/ current setup.
TIER2B_PLAN = [
    ('grit_v2',    'grit_v2_filtered',     'new',    50, 70, 'ai2/webolmo',         'ai2/oe-omai'),  # 10.3M
]

# (model_key, model_type, model_path, bs)
JUDGE_MODELS = {
    'qwen':   ('qwen3vl',  'Qwen/Qwen3-VL-32B-Instruct', 8),
    'intern': ('internvl3','OpenGVLab/InternVL3-78B',     6),
}

def filter_glob_for(ds, sub, schema):
    # 'rescored' schema (vg/oi/pixmo): strict-rescored jsonls (new field names, legacy key struct)
    if schema == 'rescored':
        return f'{FILTER_ROOT}/{sub}/rescored_worker_*.jsonl'
    return f'{FILTER_ROOT}/{sub}/*.jsonl'


def make_task(ds, sub, schema, model_key, model_type, model_path, bs, shard_i, shard_n):
    """One judge task YAML."""
    out_jsonl = f'{JUDGE_OUT}/{ds}/judge_{model_key}_{ds}_s{shard_i:03d}_of_{shard_n:03d}.jsonl'
    fglob = filter_glob_for(ds, sub, schema)
    f1_flag = '--f1_lt_1_only'  # only judge rejected pool
    return f"""
- name: judge-{model_key}-{ds}-s{shard_i:03d}
  image: {{ beaker: ai2/cuda12.8-ubuntu22.04-torch2.6.0 }}
  command: ['bash','-c']
  arguments:
  - |
    set -e
    apt-get update && apt-get install -y git wget curl || true
    pip install --upgrade pip
    pip install --force-reinstall "transformers>=4.57.0,<5.0.0" safetensors huggingface-hub --no-cache-dir
    pip install pillow tqdm numpy accelerate einops qwen-vl-utils sentencepiece timm torchvision --no-cache-dir
    export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export HF_HOME={HF_CACHE}
    mkdir -p {JUDGE_OUT}/{ds}
    python3 {SCRIPT_JUDGE} \\
      --ds {ds} --filter_glob "{fglob}" --schema {schema} {f1_flag} \\
      --shard {shard_i}/{shard_n} \\
      --model_type {model_type} --model_path {model_path} \\
      --out_jsonl "{out_jsonl}" \\
      --batch_size {bs} --max_new_tokens 512 --resume
  datasets:
  - mountPath: /weka/oe-training-default
    source: {{ weka: oe-training-default }}
  resources: {{ gpuCount: 4, sharedMemory: 400GiB }}
  context: {{ priority: high, preemptible: true }}
  constraints: {{ cluster: [ai2/jupiter, ai2/ceres, ai2/saturn] }}
  hostNetworking: true"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=['1','2a','2b'], default='1',
                    help='1=vg/oi/pixmo/sc/rf100, 2a=cc3m+cc12m, 2b=grit_v2')
    ap.add_argument('--submit', action='store_true', help='actually submit; default = dry-run')
    ap.add_argument('--only_ds', nargs='*', default=None)
    args = ap.parse_args()
    stamp = datetime.datetime.now().strftime('%H%M')
    PLAN = {'1': TIER1_PLAN, '2a': TIER2A_PLAN, '2b': TIER2B_PLAN}[args.tier]

    # ONE experiment per dataset.
    for ds, sub, schema, qwen_n, intern_n, ws, budget in PLAN:
        if args.only_ds and ds not in args.only_ds: continue
        tasks = []
        for mk, n_shards in [('qwen', qwen_n), ('intern', intern_n)]:
            mtype, mpath, bs = JUDGE_MODELS[mk]
            for i in range(n_shards):
                tasks.append(make_task(ds, sub, schema, mk, mtype, mpath, bs, i, n_shards))
        yaml_text = f"""version: v2
description: "judge cascade {ds} (Qwen {qwen_n}sh + InternVL {intern_n}sh = {len(tasks)} tasks)"
budget: {budget}

tasks:
""" + ''.join(tasks)
        exp_name = f'judge-cascade-{ds}-{stamp}'
        if not args.submit:
            print(f'\n=== DRY-RUN {ds} → {ws} / {exp_name} ({len(tasks)} tasks) ===')
            print(yaml_text[:1200] + ('\n... [truncated]' if len(yaml_text) > 1200 else ''))
            continue
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            f.write(yaml_text); path = f.name
        cmd = ['beaker','experiment','create','--workspace', ws,'--name', exp_name, path]
        print(f'$ {" ".join(cmd)}')
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode: print('ERR:', res.stderr)


if __name__ == '__main__':
    main()
