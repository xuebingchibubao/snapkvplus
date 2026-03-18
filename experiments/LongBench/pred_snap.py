import os
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch
from snapkv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral
from snapkv.monkeypatch.snapkv_utils import bind_tokenizer_to_model

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
LONG_BENCH_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]
LONG_BENCH_E_DATASETS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", 
        "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
        "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1", "llama-2-7B-32k-instruct", "mixtral-8x7B-instruct-v0.1","lwm-text-chat-1m", "lwm-text-1m"])
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument(
        '--obs-window-mode',
        type=str,
        default='adaptive',
        choices=['adaptive', 'fixed'],
        help="Observation window mode. Use 'fixed' to force window_sizes as-is; default 'adaptive' keeps the current sentence-aware behavior.",
    )
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        help="Dataset to evaluate on. Use 'all' to run the full suite, or provide a comma-separated list.",
    )
    parser.add_argument('--model-path-override', type=str, default=None, help="Use a local model path instead of config/model2path.json")
    parser.add_argument(
        '--data-root',
        type=str,
        default=os.environ.get("LONG_BENCH_DATA_ROOT"),
        help="Local LongBench data root. Defaults to LONG_BENCH_DATA_ROOT when set.",
    )
    parser.add_argument(
        '--local-files-only',
        dest='local_files_only',
        action='store_true',
        help="Load model/tokenizer using only local files (default).",
    )
    parser.add_argument(
        '--allow-remote',
        dest='local_files_only',
        action='store_false',
        help="Allow Hugging Face to download missing model/data files from the network.",
    )
    parser.set_defaults(local_files_only=True)
    return parser.parse_args(args)


def resolve_datasets(dataset_arg, use_e=False):
    available_datasets = LONG_BENCH_E_DATASETS if use_e else LONG_BENCH_DATASETS
    if dataset_arg is None:
        return available_datasets

    requested = dataset_arg.strip()
    if requested.lower() == "all":
        return available_datasets

    datasets = [name.strip() for name in requested.split(",") if name.strip()]
    invalid_datasets = [name for name in datasets if name not in available_datasets]
    if invalid_datasets:
        raise ValueError(
            f"Dataset(s) {invalid_datasets} not found. Available datasets: {available_datasets}"
        )
    return datasets

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        print('chatglm3')
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        print('chatglm')
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        print('longchat')
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2"  in model_name or "llama-2" in model_name or "lwm" in model_name:
        print('llama2', model_name)
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        print('xgen')
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        print('internlm')
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "mistral" in model_name or "mixtral" in model_name:
        print('mistral')
        # from fastchat.model import get_conversation_template
        # conv = get_conversation_template("mistral")
        # conv.append_message(conv.roles[0], prompt)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        prompt = prompt
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen, 
                        prompt_format, dataset, model_name, 
                        model2path, out_path, 
                        compress=False, 
                        local_files_only=False,
                        window_sizes = None,
                        max_capacity_prompts = None,
                        kernel_sizes = None,
                        pooling = None,
                        obs_window_mode = "adaptive",
                        model=None,
                        tokenizer=None):
    # device = torch.device(f'cuda:{rank}')
    # device = model.device
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            model2path[model_name],
            model_name,
            device="cuda",
            compress=compress,
            local_files_only=local_files_only,
        )
    device = model.device
    printed = False
    for json_obj in tqdm(data):
        ############################################################################################################
        # load compress args
        if compress:
            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
                model.model.layers[i].self_attn.config.obs_window_mode = obs_window_mode
        ############################################################################################################
        
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if not printed:
            print(prompt)
            printed = True
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def _load_local_longbench_dataset(data_root, dataset_name):
    data_root = Path(data_root).expanduser().resolve()

    # 1) load_from_disk directory layouts
    disk_candidates = [
        data_root / dataset_name,
        data_root / "LongBench" / dataset_name,
        data_root / "data" / dataset_name,
    ]
    for candidate in disk_candidates:
        if candidate.exists() and candidate.is_dir():
            dataset = load_from_disk(str(candidate))
            if hasattr(dataset, "keys"):
                if "test" in dataset:
                    return dataset["test"]
                first_split = next(iter(dataset.keys()))
                return dataset[first_split]
            return dataset

    # 2) raw exported files
    file_candidates = []
    for suffix in ["jsonl", "json", "parquet"]:
        file_candidates.extend([
            data_root / f"{dataset_name}.{suffix}",
            data_root / dataset_name / f"test.{suffix}",
            data_root / "LongBench" / f"{dataset_name}.{suffix}",
            data_root / "LongBench" / dataset_name / f"test.{suffix}",
            data_root / "data" / f"{dataset_name}.{suffix}",
            data_root / "data" / dataset_name / f"test.{suffix}",
        ])

    for candidate in file_candidates:
        if candidate.exists() and candidate.is_file():
            if candidate.suffix == ".parquet":
                return load_dataset("parquet", data_files=str(candidate), split="train")
            return load_dataset("json", data_files=str(candidate), split="train")

    raise FileNotFoundError(
        f"Could not find local LongBench dataset '{dataset_name}' under {data_root}. "
        f"Supported layouts: load_from_disk folder, or raw json/jsonl/parquet files such as "
        f"{data_root / 'data' / f'{dataset_name}.jsonl'}."
    )


def load_longbench_dataset(dataset, use_e=False, data_root=None, local_files_only=True):
    dataset_name = f"{dataset}_e" if use_e else dataset
    if data_root:
        return _load_local_longbench_dataset(data_root, dataset_name)
    if local_files_only:
        raise ValueError(
            "Local-only mode is enabled, but no LongBench data root was provided. "
            "Please pass --data-root /path/to/LongBench or set LONG_BENCH_DATA_ROOT."
        )
    return load_dataset('THUDM/LongBench', dataset_name, split='test')


def load_model_and_tokenizer(path, model_name, device, compress=False, local_files_only=False):
    try:
        if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=local_files_only)
            model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=local_files_only).to(device)
        elif "llama2" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=local_files_only).to(device)
        elif "longchat" in model_name or "vicuna" in model_name:
            if not compress:
                model = AutoModelForCausalLM.from_pretrained(
                        path,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        use_cache=True,
                        use_flash_attention_2=True,
                        local_files_only=local_files_only
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                        path,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        use_cache=True,
                        use_flash_attention_2=True,
                        local_files_only=local_files_only
                    )
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                use_fast=False,
                local_files_only=local_files_only,
            )
        elif "llama-2" in model_name or "lwm" in model_name:
            if not compress:
                model = AutoModelForCausalLM.from_pretrained(
                        path,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        use_cache=True,
                        use_flash_attention_2=True,
                        local_files_only=local_files_only
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                        path,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        use_cache=True,
                        use_flash_attention_2=True,
                        local_files_only=local_files_only
                    )
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                use_fast=False,
                local_files_only=local_files_only,
            )
        elif "mistral" in model_name:
            if not compress:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True,
                    local_files_only=local_files_only
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True,
                    local_files_only=local_files_only
                )
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                padding_side="right",
                use_fast=False,
                local_files_only=local_files_only,
            )
        elif "mixtral" in model_name:
            if not compress:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True,
                    local_files_only=local_files_only
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True,
                    local_files_only=local_files_only
                )
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                # padding_side="right",
                # use_fast=False,
                local_files_only=local_files_only,
            )
        else:
            raise ValueError(f"Model {model_name} not supported!")
    except Exception as exc:
        if local_files_only:
            raise RuntimeError(
                f"Failed to load model '{model_name}' locally from '{path}'. "
                "Please provide a local directory with --model-path-override, or pre-download the model into the local Hugging Face cache."
            ) from exc
        raise
    model = bind_tokenizer_to_model(model.eval(), tokenizer)
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    # world_size = torch.cuda.device_count()
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open(CONFIG_DIR / "model2path.json", "r"))
    model2maxlen = json.load(open(CONFIG_DIR / "model2maxlen.json", "r"))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    if model_name is None:
        raise ValueError("--model is required")
    # define your model
    max_length = model2maxlen[model_name]
    datasets = resolve_datasets(args.dataset, use_e=args.e)
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open(CONFIG_DIR / "dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(CONFIG_DIR / "dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    dataset = args.dataset
    # for dataset in datasets:
    if args.compress_args_path:
        compress_args = json.load(open(CONFIG_DIR / args.compress_args_path, "r"))
        compress = True
        write_model_name = model_name + args.compress_args_path.split(".")[0]
        replace_llama()
        replace_mistral()
        replace_mixtral()
    else:
        compress = False
        compress_args = None
        write_model_name = model_name
    if args.model_path_override:
        model2path[model_name] = args.model_path_override

    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name],
        model_name,
        device="cuda",
        compress=compress,
        local_files_only=args.local_files_only,
    )

    output_root = BASE_DIR / ("pred_e" if args.e else "pred") / write_model_name
    output_root.mkdir(parents=True, exist_ok=True)

    for index, dataset in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] Running dataset: {dataset}")
        data = load_longbench_dataset(
            dataset,
            use_e=args.e,
            data_root=args.data_root,
            local_files_only=args.local_files_only,
        )
        out_path = output_root / f"{dataset}.jsonl"
        if out_path.exists():
            out_path.unlink()

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        if compress_args is not None:
            get_pred_single_gpu(
                data_all,
                max_length,
                max_gen,
                prompt_format,
                dataset,
                model_name,
                model2path,
                out_path,
                compress,
                local_files_only=args.local_files_only,
                model=model,
                tokenizer=tokenizer,
                obs_window_mode=args.obs_window_mode,
                **compress_args,
            )
        else:
            get_pred_single_gpu(
                data_all,
                max_length,
                max_gen,
                prompt_format,
                dataset,
                model_name,
                model2path,
                out_path,
                compress,
                local_files_only=args.local_files_only,
                model=model,
                tokenizer=tokenizer,
                obs_window_mode=args.obs_window_mode,
            )
