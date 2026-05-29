
# PARALLEL IMPLEMENTATION
# -----------------------------------------------------------------------------------------------------------
def login_hf():
    config = configparser.ConfigParser()
    config.read(f"{main_path}config.ini")

    HUGGINGFACE_TOKEN = config.get("huggingface", "token", fallback=None)

    if not HUGGINGFACE_TOKEN:
        raise ValueError("HuggingFace token not found in config.ini")
    login(HUGGINGFACE_TOKEN)



# Multiprocessing worker function
def worker(worker_id, gpu_group, rows, config):
    # ---- logging ----
    lm = LogManager(f"worker{worker_id}_gpus{'_'.join(map(str, gpu_group))}")
    lm.printl(f"Worker {worker_id} using GPUs {gpu_group} starting with {len(rows)} rows")

    # ---- set visible GPUs ----
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_group))

    login_hf()
    # IMPORTANT: inside worker, GPU indices start from 0
    torch.cuda.set_device(0)
    # Compute K locally
    print_K:int = max(1, int(0.01 * len(rows)))

    dataset_prompt = DatasetPromptFactory.create(config["dataset_prefix"])
    tokenizer = AutoTokenizer.from_pretrained(
                config["llm_model"],
                use_fast=False,
                cache_dir=cache_path
            )

    model = AutoModelForCausalLM.from_pretrained(
        config["llm_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=cache_path
    )

    for chunk_idx, (i, row) in enumerate(rows):
        with measure_time(lm, when_print='every_K', i=chunk_idx, K=print_K, label=f"[GPU {gpu_group}] Answered patient {str(chunk_idx+1)}/{str(len(rows))} - original_index: {row['original_index']}"):
            system_prompt, user_prompt, options = dataset_prompt.build_question(row, return_options=config['return_options'], enhancer_name=config['enhancer_name'], llm_name=config['llm_name'])
            if tokenizer.chat_template is None:
                # Llama-2 Style (Standard for Meditron 7B/70B)
                tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'system' %}"
                    "{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}"
                    "{% elif message['role'] == 'user' %}"
                    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] }}"
                    "{% endif %}"
                    "{% endfor %}"
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 2. Apply template but DON'T use stop_strings in generate yet
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 3. MANUALLY APPEND the start of your JSON. 
            # This bypasses the <[ANS] tag hallucination entirely.
            prompt += '{"step_by_step_thinking": "'

            # ---- measure LLM latency (ONLY the model call) ----
            t0 = time.perf_counter()
            answer, snippets, scores = worker_answer(prompt, tokenizer, model, lm)
            llm_seconds = time.perf_counter() - t0
            # ---------------------------------------------------

            # lm.printl(f"Answer: {answer}")
            # lm.printl("--------------------------------------------------")
            new_row = process_huggingface_answer(lm, config['enhancer_name'], config['llm_name'], config['info'], answer, snippets, scores, row, ind, i)

            df_out = pd.DataFrame([new_row])
            gpu_file = f"{config['output_path']}{config['enhancer_name']}_gpu{worker_id}.csv"
            df_out.to_csv(gpu_file, mode="a", header=not os.path.exists(gpu_file), index=False)

            # write timing output to a separate per-worker file
            timing_row = {
                "worker_id": worker_id,
                "gpu_group": gpu_group,
                "chunk_idx": chunk_idx,
                "i": i,
                "original_index": row.get("original_index"),
                "llm_seconds": llm_seconds,
                # optional: makes later debugging easier
                "prompt_chars": len(prompt),
            }
            timing_df = pd.DataFrame([timing_row])
            timing_file = f"{config['output_path']}{config['enhancer_name']}_gpu{worker_id}_timing.csv"
            timing_df.to_csv(timing_file, mode="a", header=not os.path.exists(timing_file), index=False)


def worker_answer(prompt: str, tokenizer, model, lm) -> str:
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate as you were doing
    outputs = model.generate(
        **inputs, 
        max_new_tokens=2048, # JSON takes more tokens, don't cut it short!
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,      
        temperature=0.1      # Keep it low for structured data
    )
    
    # 5. Decode and Reconstruct
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Add back the part we "forced"
    generated_text = '{"step_by_step_thinking": "' + generated_text
    # If the model repeats itself, just cut at the first "}" 
    answer = generated_text.split("}")[0] + "}"

    snippets = []
    scores = []

    return answer, snippets, scores




# CLASS DEFINITION
# ---------------------------------------------------------------------------------------------------------------------------

class HuggingFaceInterface:
    def __init__(self, lm: LogManager, ch: Checkpoint, icm: IntegrityConstraintManager, dm: DirectoryManager,
                 dataset_prefix: str, enhancer_name: str, llm_name: str, rag: bool, return_options: bool,
                 output_path: str, confidence_computation: bool, cuda_visible_devices: str = None, parallel: bool = False):
        
        self.ch: Checkpoint = ch
        self.lm: LogManager = lm

        self.icm: IntegrityConstraintManager = icm
        self.dm: DirectoryManager = dm

        self.dataset_prefix: str = dataset_prefix
        self.dataset: str = f"{self.dataset_prefix}_preprocessed.csv"
        self.info: Dict[str, Union[str, List[str]]] = df_info[self.dataset_prefix]

        self.enhancer_name: str = enhancer_name

        self.llm_name: str = llm_name
        self.llm_model: str = llms_map[self.llm_name]
        self.rag: bool = rag
        self.return_options: bool = return_options
        self.to_update: bool = None # If True, update main dataframe; if False, save in temp path

        self.output_path: str = output_path
        self.confidence_computation: bool = confidence_computation
        self.cuda_visible_devices: str = cuda_visible_devices
        self.parallel: bool = parallel

        if not self.parallel and self.cuda_visible_devices is not None:
            # Select GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
            # Verify GPU selection
            self.lm.printl(f"Number of GPUs selected: {str(torch.cuda.device_count())}")
        
        self.cuda_devices = (
            [int(x) for x in self.cuda_visible_devices.split(",")]
            if self.cuda_visible_devices else [0]
        )

    @log_method
    def __login(self) -> None:
        config = configparser.ConfigParser()
        config.read(f"{main_path}config.ini")

        GITHUB_TOKEN = config.get("github", "token", fallback=None)

        if not GITHUB_TOKEN:
            raise ValueError("GitHub token not found in config.ini")
        login(GITHUB_TOKEN)

    

    def __export_worker_config(self):
        return {
            "dataset_prefix": self.dataset_prefix,
            "output_path": self.output_path,
            "enhancer_name": self.enhancer_name,
            "info": self.info,
            "return_options": self.return_options,
            'to_update': self.to_update,
            "llm_name": self.llm_name,
            "llm_model": self.llm_model,
        }
    
    def __answer(self, prompt: str) -> str:
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate as you were doing
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=2048, # JSON takes more tokens, don't cut it short!
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,      
            temperature=0.1      # Keep it low for structured data
        )
       
        # 5. Decode and Reconstruct
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Add back the part we "forced"
        generated_text = '{"step_by_step_thinking": "' + generated_text

        # If the model repeats itself, just cut at the first "}" 
        answer = generated_text.split("}")[0] + "}"

        snippets = []
        scores = []

        return answer, snippets, scores


    def __compute_timing_metrics_non_parallel(self):
        timing_file = f"{self.output_path}{self.enhancer_name}_timing.csv"
        if not os.path.exists(timing_file):
            raise FileNotFoundError(f"Timing file not found: {timing_file}")

        df_t = self.ch.read_dataframe(timing_file, dtype=dtype)

        if "llm_seconds" not in df_t.columns:
            raise ValueError("Expected column 'llm_seconds' in timing file.")

        df_t["llm_seconds"] = pd.to_numeric(df_t["llm_seconds"], errors="coerce")
        df_t = df_t.dropna(subset=["llm_seconds"])

        x = df_t["llm_seconds"].to_numpy()
        n = int(x.size)

        metrics = {
            "rows": n,
            "total_llm_seconds": float(np.sum(x)) if n else 0.0,
            "mean_llm_seconds": float(np.mean(x)) if n else 0.0,
            "std_llm_seconds": float(np.std(x, ddof=1)) if n > 1 else 0.0,
            "min_llm_seconds": float(np.min(x)) if n else 0.0,
            "max_llm_seconds": float(np.max(x)) if n else 0.0,
            "samples_per_llm_second": (float(n / np.sum(x)) if n and np.sum(x) > 0 else 0.0),
        }

        metrics_file = f"{self.output_path}{self.enhancer_name}_timing_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)


    @log_method
    def __run_sequential(self, rows_to_process):
        print_K = int(0.01 * len(rows_to_process))
        for i, row in rows_to_process:
            original_index = row["original_index"]
            with measure_time(self.lm, when_print='every_K', i=i, K=print_K, label=f"Answered patient {str(i+1)}/{str(len(rows_to_process))} - original_index: {original_index}"):
                self.dataset_prompt = DatasetPromptFactory.create(self.dataset_prefix)
                system_prompt, user_prompt, options = self.dataset_prompt.build_question(row, self.return_options, self.enhancer_name, self.llm_name)

                if self.tokenizer.chat_template is None:
                    # Llama-2 Style (Standard for Meditron 7B/70B)
                    self.tokenizer.chat_template = (
                        "{% for message in messages %}"
                        "{% if message['role'] == 'system' %}"
                        "{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}"
                        "{% elif message['role'] == 'user' %}"
                        "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
                        "{% elif message['role'] == 'assistant' %}"
                        "{{ message['content'] }}"
                        "{% endif %}"
                        "{% endfor %}"
                    )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # self.lm.printl(system_prompt)
                # self.lm.printl(user_prompt)
                # 2. Apply template but DON'T use stop_strings in generate yet
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                # 3. MANUALLY APPEND the start of your JSON. 
                # This bypasses the <[ANS] tag hallucination entirely.
                prompt += '{"step_by_step_thinking": "'

                t0 = time.perf_counter()
                answer, snippets, scores = self.__answer(prompt)
                llm_seconds = time.perf_counter() - t0

                self.lm.printl(answer)
                self.lm.printl("--------------------------------------------------")

                # save timing row (append)
                timing_row = {
                    "i": i,
                    "original_index": row.get("original_index", None),
                    "llm_seconds": llm_seconds,
                    "prompt_chars": len(prompt),   # optional but handy
                }
                timing_file = f"{self.output_path}{self.enhancer_name}_timing.csv"
                pd.DataFrame([timing_row]).to_csv(
                    timing_file,
                    mode="a",
                    header=not os.path.exists(timing_file),
                    index=False
                )

                # self.lm.printl(f"{answer}")
                # self.lm.printl("--------------------------------------------------")
                new_row = process_huggingface_answer(self.lm, self.enhancer_name, self.llm_name, self.info, answer, snippets, scores, row, ind, i)
                # Save / update dataframe
                if self.to_update:
                    self.ch.update_dataframe(new_row, f"{self.output_path}{self.enhancer_name}_df.csv")
                else:
                    new_row_df = pd.DataFrame([new_row])
                    self.ch.save_dataframe(new_row_df, f"{self.dm.temp_answer_path}index_{str(int(row[ind]))}_{self.enhancer_name}.csv")
        # After all rows are processed, compute timing metrics
        self.__compute_timing_metrics_non_parallel()


    @staticmethod
    def __gpu_sort_key(filename: str) -> int:
        # Extract the GPU ID number from the filename
        match = re.search(r"_gpu(\d+)\.csv$", filename)
        return int(match.group(1)) if match else -1


    @log_method
    def __merge_parallel_outputs(self):
        files = glob.glob(f"{self.output_path}{self.enhancer_name}_gpu[0-9]*.csv")
        files = [f for f in files if not f.endswith("_timing.csv")]
        # Sort files by name (lexicographically)
        files = sorted(files, key=self.__gpu_sort_key)

        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv(f"{self.output_path}{self.enhancer_name}_df.csv", index=False)


    @log_method
    def __merge_parallel_timing_and_metrics(self):
        # 1) Find timing files
        timing_files = glob.glob(f"{self.output_path}{self.enhancer_name}_gpu*_timing.csv")
        timing_files = sorted(timing_files, key=self.__gpu_sort_key)

        if not timing_files:
            raise FileNotFoundError(
                f"No timing files found matching: {self.output_path}{self.enhancer_name}_gpu*_timing.csv"
            )

        # 2) Merge all timing files
        df_t = pd.concat([pd.read_csv(f) for f in timing_files], ignore_index=True)

        if "llm_seconds" not in df_t.columns:
            raise ValueError("Expected column 'llm_seconds' in timing files.")

        df_t["llm_seconds"] = pd.to_numeric(df_t["llm_seconds"], errors="coerce")
        df_t = df_t.dropna(subset=["llm_seconds"])

        x = df_t["llm_seconds"].to_numpy()
        n = int(x.size)

        # 3) Compute simple metrics (NO percentiles)
        metrics = {
            "rows": n,
            "total_llm_seconds": float(np.sum(x)) if n else 0.0,
            "mean_llm_seconds": float(np.mean(x)) if n else 0.0,
            "std_llm_seconds": float(np.std(x, ddof=1)) if n > 1 else 0.0,
            "min_llm_seconds": float(np.min(x)) if n else 0.0,
            "max_llm_seconds": float(np.max(x)) if n else 0.0,
        }

        # Throughput (model-time based)
        if metrics["total_llm_seconds"] > 0:
            metrics["samples_per_llm_second"] = n / metrics["total_llm_seconds"]
        else:
            metrics["samples_per_llm_second"] = 0.0

        # 4) Save everything in ONE CSV
        metrics_file = f"{self.output_path}{self.enhancer_name}_timing_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    @log_method
    def __run_parallel(self, rows_to_process):
        gpu_list = self.cuda_devices
        gpus_per_worker = 2

        # ---- build GPU groups ----
        gpu_groups = [
            gpu_list[i:i + gpus_per_worker]
            for i in range(0, len(gpu_list), gpus_per_worker)
        ]

        num_workers = len(gpu_groups)

        self.lm.printl(f"Running parallel on GPU groups: {gpu_groups}")

        # ---- split rows per worker ----
        chunks = [rows_to_process[i::num_workers] for i in range(num_workers)]
       
        processes = []
        for worker_id, (gpu_group, chunk) in enumerate(zip(gpu_groups, chunks)):
            p = mp.Process(target=worker, args=(worker_id, gpu_group, chunk, self.__export_worker_config()),
                name=f"Worker-{worker_id}-GPUs{'_'.join(map(str, gpu_group))}")
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.lm.printl(f"Process {p.name} has finished.")

        self.__merge_parallel_outputs()
        self.__merge_parallel_timing_and_metrics()



    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------

    @log_method
    def initialize_huggingface(self):
        if self.parallel:
            return  # workers initialize themselves
        # Login to HuggingFace
        self.__login()
        # Initialize Model and Tokenizer
        
        with measure_time(self.lm, label="initialize_huggingface"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model,
                use_fast=False,
                cache_dir=cache_path
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_path
            )

    def set_to_update(self, to_update: bool):
        self.to_update = to_update

    @log_method
    def run_huggingface(self, rows_to_process: List[Tuple[int, pd.Series]]):
        if not self.parallel:
            self.__run_sequential(rows_to_process)
        else:
            self.__run_parallel(rows_to_process)

