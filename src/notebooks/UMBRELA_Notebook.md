**Objective**: Set up UMBRELA using Colab notebook (w/ GPT & HuggingFace Open-Source Models for eval)

Steps:

1.   Install Anserini
2.   Install Pyserini
3.   Install UMBRELA



**Anserini:** Need Java 21 and Maven 3.9+

Get JDK 21


```python
!apt-get install openjdk-21-jdk
```


```python
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'
os.environ['PATH'] = os.environ['PATH'] + ':' + os.environ['JAVA_HOME'] + '/bin'
```

Confirm Java Version


```python
!java -version
```

Get Maven (Latest)


```python
!wget https://dlcdn.apache.org/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz
```


```python
!tar -xvzf apache-maven-3.9.11-bin.tar.gz
```


```python
!sudo mv apache-maven-3.9.11 /opt/maven
```


```python
import os
os.environ['MAVEN_HOME'] = '/opt/maven'
os.environ['PATH'] = os.environ['MAVEN_HOME'] + '/bin:' + os.environ['PATH']
```

Confirm Maven is present


```python
!mvn --version
```

Clone Anserini


```python
!git clone https://github.com/castorini/anserini.git --recurse-submodules
```

Follow along the Anserini official repo installation process


```python
%cd anserini/
```


```python
!git submodule update --init --recursive
```


```python
!mvn clean package -DskipTests
```


```python
!cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
```


```python
!cd tools/eval/ndeval && make && cd ../../..
```


```python
%cd ../
```

**Pyserini**

Follow development installation instructions


```python
!git clone https://github.com/castorini/pyserini.git --recurse-submodules
```


```python
%cd pyserini/
```


```python
!cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
```


```python
!cd tools/eval/ndeval && make && cd ../../..
```


```python
!pip install -e .
```


```python
%cd ../
```


```python
%cd anserini/target/
```


```python
!ls -l
```


```python
!mv anserini-1.1.2-SNAPSHOT-fatjar.jar /content/pyserini/pyserini/resources/jars/
```


```python
%cd ../../
```


```python
%cd pyserini/
```


```python
#!python -m unittest
```


```python
%cd ../
```

Set up UMBRELA


```python
!git clone https://github.com/castorini/umbrela.git
```


```python
!pip install faiss-cpu torch
```


```python
%cd /content/umbrela
```


```python
!pip install -r requirements.txt
```


```python
!pip install -e .
```

Now, there are a dependency that need to be installed for the eval command to run


```python
!pip install retry
```

# **FOR CHATGPT**



```python
%cd src/
```


```python
%%writefile chatgptrun.py
from umbrela.gpt_judge import GPTJudge
from dotenv import load_dotenv

load_dotenv()

judge_gpt = GPTJudge(qrel="dl19-passage", prompt_type="bing", model_name="gpt-4o")

input_dict = {
    "query": {"text": "how long is life cycle of flea", "qid": "264014"},
    "candidates": [
        {
            "doc": {
                "segment": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day."
            },
            "docid": "4834547",
        },
    ]
}

judgments = judge_gpt.judge(request_dict=input_dict)
print(judgments)

```


```python
import os
os.environ['OPEN_AI_API_KEY'] = 'sk-dummy-key-for-testing'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://dummy.openai.azure.com/'
os.environ['AZURE_OPENAI_API_VERSION'] = '2024-02-15-preview'
os.environ['AZURE_OPENAI_API_KEY'] = 'dummy-key-for-testing'
os.environ['AZURE_OPENAI_API_BASE'] = 'https://dummy.openai.azure.com/'
os.environ['DEPLOYMENT_NAME'] = 'gpt-4o'
```


```python
!python chatgptrun.py
```

Error, as expected, since I did not use valid values for the env variables. However, generally, it would work otherwise.

# **FOR HUGGING FACE OPEN-SOURCE MODELS**

Below is a Claude-generated prompt to test small, medium, large open-source models on Hugging Face with judgement snippets


```python
import os
os.environ["HF_TOKEN"] = "...................."
os.environ["HF_CACHE_DIR"] = "/content/hf_cache"
```


```python
%%writefile huggingfacerun.py
"""
UMBRELA 30-Minute Benchmark for Colab Pro Decision
Optimized to test real TREC evaluation performance in minimal time
"""

import torch
import gc
import time
import json
import psutil
import subprocess
import os
import re # Import regex for parsing model output
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
import warnings
warnings.filterwarnings("ignore")

class QuickUmbrelaBenchmark:
    def __init__(self):
        self.results = []

        # Real TREC DL 2019 samples (from UMBRELA paper)
        self.trec_samples = [
            {
                "query": "how long is life cycle of flea",
                "passage": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day.",
                "expected_relevance": 3  # From UMBRELA paper
            },
            {
                "query": "medicare's definition of mechanical ventilation",
                "passage": "Continuous Positive Airway Pressure (CPAP) Continuous positive airway pressure – also called CPAP – is a treatment in which a mask is worn over the nose and/or mouth while you sleep. The mask is hooked up to a machine that delivers a continuous flow of air into the nose.",
                "expected_relevance": 1
            },
            {
                "query": "what is the daily life of thai people",
                "passage": "Thai Flag Meaning: The red stripes represent the blood spilt to maintain Thailand's independence. The white stands for purity and is the color of Buddhism which is the country's main religion. The flag of Thailand consists of five horizontal stripes.",
                "expected_relevance": 0
            },
            {
                "query": "define visceral pleura",
                "passage": "The visceral pleura is the thin membrane that covers the surface of the lungs. It is continuous with the parietal pleura, which lines the chest cavity. The pleural space between these two membranes contains pleural fluid.",
                "expected_relevance": 3
            },
            {
                "query": "causes of air pollution",
                "passage": "Air pollution is caused by various factors including vehicle emissions, industrial activities, burning of fossil fuels, deforestation, and agricultural practices. These activities release harmful substances into the atmosphere.",
                "expected_relevance": 2
            }
        ]

        # Models to test with approximate parameter counts (M = million, B = billion)
        self.test_models = [
            {
                "name": "google/flan-t5-small",
                "type": "t5",
                "size": "Small",
                "params_m": 80,
                "description": "Smallest T5, instruction-tuned"
            },
            {
                "name": "google/flan-t5-base",
                "type": "t5",
                "size": "Base",
                "params_m": 250,
                "description": "Mid-range T5"
            },
            {
                "name": "google/flan-t5-large",
                "type": "t5",
                "size": "Large",
                "params_m": 780,
                "description": "Larger T5"
            },
            {
                "name": "google/flan-t5-xl",
                "type": "t5",
                "size": "Extra Large",
                "params_b": 3, # 3 Billion
                "description": "Significant T5 model for better judgments"
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "type": "gpt",
                "size": "Medium",
                "params_m": 345, # From previous info
                "description": "Alternative GPT model"
            },
            {
                "name": "microsoft/Phi-3-mini-4k-instruct",
                "type": "gpt", # Phi-3 models are decoder-only, like GPT
                "size": "Mini (Instruct)",
                "params_b": 3.8, # 3.8 Billion parameters
                "description": "Microsoft's small, capable instruct model"
            },
            {
                "name": "google/gemma-2-9b",
                "type": "gpt", # Gemma models are decoder-only
                "size": "9B",
                "params_b": 9,
                "description": "Gemma 9 Billion instruction-tuned"
            },
            {
                "name": "Qwen/Qwen2-7B-Instruct",
                "type": "gpt", # Qwen models are decoder-only (causal LM)
                "size": "7B (Instruct)",
                "params_b": 7,
                "description": "Qwen 7 Billion instruction-tuned"
            },
            {
                "name": "HuggingFaceM4/idefics-9b-instruct",
                "type": "gpt", # IDEFICS is decoder-only, with multimodal capabilities
                "size": "9B (Instruct, Multi)",
                "params_b": 9,
                "description": "IDEFICS 9 Billion instruction-tuned, multimodal"
            },
            {
                "name": "utter-project/EuroLLM-9B-Instruct",
                "type": "gpt", # EuroLLM is decoder-only
                "size": "9B (Instruct, Multi-lingual)",
                "params_b": 9,
                "description": "EuroLLM 9 Billion instruction-tuned"
            },
            {
                "name": "01-ai/Yi-1.5-9B-Chat",
                "type": "gpt", # Yi models are decoder-only (chat variant)
                "size": "9B (Chat)",
                "params_b": 9,
                "description": "Yi 9 Billion chat model"
            },
            {
                "name": "meta-llama/Meta-Llama-3-8B-Instruct",
                "type": "gpt", # Llama models are decoder-only
                "size": "8B (Instruct)",
                "params_b": 8,
                "description": "Meta Llama 3 8 Billion instruction-tuned"
            }
        ]

    def check_system_resources(self):
        """Check available system resources and determine Colab tier"""
        print("\n" + "="*70)
        print("🔍 SYSTEM RESOURCES & COLAB TIER DETECTION")
        print("="*70)

        # RAM
        ram = psutil.virtual_memory()
        print(f"💾 RAM: {ram.available/1024**3:.1f}GB available / {ram.total/1024**3:.1f}GB total")

        # GPU Detection
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory/1024**3
            gpu_available = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3

            print(f"🚀 GPU: {gpu_name}")
            print(f"📊 GPU Memory: {gpu_memory:.1f}GB total, {gpu_available:.1f}GB available")

            # Determine Colab tier (updated for typical Colab GPU memory ranges)
            if "T4" in gpu_name and gpu_memory < 16:
                tier = "🆓 Colab FREE (T4, ~15GB)"
            elif "V100" in gpu_name or (gpu_memory > 20 and gpu_memory < 35):
                tier = "💰 Colab PRO (V100, ~32GB)"
            elif "A100" in gpu_name or gpu_memory >= 35:
                tier = "💎 Colab PRO+ (A100, ~40GB)"
            else:
                tier = f"❓ Unknown ({gpu_name}, {gpu_memory:.1f}GB)"

            print(f"🏷️  Detected: {tier}")
        else:
            gpu_name = "CPU Only"
            gpu_memory = 0
            print("❌ GPU: Not available (CPU mode)")
            tier = "CPU Only"

        print("="*70)
        return {"name": gpu_name, "memory": gpu_memory, "tier": tier}

    def create_umbrela_prompt(self, query, passage):
        """
        Create UMBRELA-style prompt based on Figure 1 of the paper,
        tailored to encourage numerical output in the specified format.
        """
        prompt_template = """Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = Irrelevant: The passage has nothing to do with the query.
1 = Related: The passage seems related to the query but does not answer it.
2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic and also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: {query}
Passage: {passage}

Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.
##final score: """

        prompt = prompt_template.format(query=query, passage=passage)
        return prompt

    def test_model_performance(self, model_info, num_samples=80): # Default samples set to 80
        """Test a specific model with timing and memory benchmarks and check judgment accuracy"""
        model_name = model_info["name"]
        model_type = model_info["type"]

        print(f"\n{'='*70}")
        print(f"🧪 TESTING: {model_name}")
        print(f"📋 Type: {model_type.upper()} | Size: {model_info['size']}")
        print("="*70)

        try:
            # Memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem_before = torch.cuda.memory_allocated() / 1024**3
            else:
                mem_before = 0

            # Load model
            print("⏳ Loading model...")
            start_time = time.time()

            # Use AutoModelForSeq2SeqLM for T5 models (encoder-decoder)
            # Use AutoModelForCausalLM for GPT-style models (decoder-only)
            if model_type == "t5":
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:  # GPT-style
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Specific models that might require trust_remote_code=True
                if any(m in model_name for m in ["Phi-3", "Qwen", "idefics", "EuroLLM", "Yi"]):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set for generation

            load_time = time.time() - start_time

            # Memory after loading
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**3
                memory_used = mem_after - mem_before
            else:
                memory_used = 0

            print(f"✅ Loaded in {load_time:.1f}s | Memory: {memory_used:.2f}GB")

            # Benchmark evaluations
            print(f"🔄 Running {num_samples} evaluations...")

            eval_times = []
            correct_judgments_count = 0
            sample_outputs = []

            for i in range(num_samples):
                # Cycle through our TREC samples
                sample = self.trec_samples[i % len(self.trec_samples)]
                query = sample["query"]
                passage = sample["passage"]

                eval_start = time.time()

                try:
                    prompt = self.create_umbrela_prompt(query, passage) # Prompt is now generic

                    inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
                    if torch.cuda.is_available():
                        inputs = inputs.to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_new_tokens=20, # Increased from 3/5 to allow more generation
                            temperature=0.1,  # Keep responses deterministic
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                        )

                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Attempt to parse the expected format "##final score: X"
                    parsed_response = None
                    match = re.search(r'##final score:\s*(\d+)', full_response)
                    if match:
                        try:
                            parsed_response = int(match.group(1).strip())
                        except ValueError:
                            pass # Not a valid integer

                    # Prepare response for display
                    display_response = str(parsed_response) if parsed_response is not None else full_response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
                    display_response = display_response[:20] if len(display_response) > 20 else display_response # Truncate if too long

                    is_correct = (parsed_response == sample["expected_relevance"])
                    if is_correct:
                        correct_judgments_count += 1

                    eval_time = time.time() - eval_start
                    eval_times.append(eval_time)

                    if i < 3: # Store first few examples for detailed output
                        sample_outputs.append({
                            "query": query[:50] + "...",
                            "response": display_response,
                            "time": eval_time,
                            "expected": sample["expected_relevance"],
                            "correct": is_correct
                        })

                except Exception as e:
                    print(f"  ⚠️ Evaluation {i+1} failed: {str(e)[:50]}...") # Truncate error message
                    eval_times.append(float('inf')) # Record as infinite time for failed eval

            # Filter out infinite times for avg calculation, but keep for success rate context
            valid_eval_times = [t for t in eval_times if t != float('inf')]
            avg_eval_time = sum(valid_eval_times) / max(len(valid_eval_times), 1)

            # The 'success rate' now reflects the rate of judgments being parsed correctly and compared
            # to expected. Not just execution success.
            # A correct judgment rate is more meaningful than just successful execution.
            correct_judgment_rate = (correct_judgments_count / num_samples) * 100 if num_samples > 0 else 0

            # Project to full TREC dataset (9260 evaluations)
            full_trec_evals = 9260
            projected_hours = (avg_eval_time * full_trec_evals) / 3600
            projected_minutes = ((avg_eval_time * full_trec_evals) % 3600) / 60

            # Determine Colab recommendation
            # NOTE: These thresholds are estimates. Actual performance varies.
            if memory_used > 12: # Over 12GB VRAM often requires A100 (Pro+)
                colab_rec = "PRO+ Required"
                colab_reason = f"Memory ({memory_used:.1f}GB) too high for Free/Pro"
            elif memory_used > 8: # Over 8GB VRAM often benefits from V100 (Pro)
                colab_rec = "PRO Recommended"
                colab_reason = f"Memory ({memory_used:.1f}GB) ideal for Pro"
            elif projected_hours > 24:
                colab_rec = "PRO Recommended"
                colab_reason = f"Time ({projected_hours:.1f}h) > 24 hours (Colab max lifetime)"
            elif projected_hours > 12:
                colab_rec = "PRO Helpful"
                colab_reason = f"Time ({projected_hours:.1f}h) > 12 hours (Colab max lifetime)"
            else:
                colab_rec = "FREE OK"
                colab_reason = "Within limits"

            # Print results
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"  ⚡ Avg evaluation: {avg_eval_time:.3f}s")
            print(f"  ✅ Correct Judgment Rate: {correct_judgment_rate}/{num_samples} ({correct_judgment_rate:.1f}%)")
            print(f"  💾 Memory used: {memory_used:.2f}GB")
            print(f"  🕐 Full TREC est: {projected_hours:.1f}h {projected_minutes:.0f}m")
            print(f"  🎯 Colab rec: {colab_rec} ({colab_reason})")
            print(f"Total model test completion: {(time.time() - start_time):.1f}s") # Time for this model's test

            # Show sample outputs
            if sample_outputs:
                print(f"\n🔍 SAMPLE OUTPUTS:")
                for i, sample_out in enumerate(sample_outputs):
                    correct_str = "✔️ Correct" if sample_out['correct'] else "❌ Incorrect"
                    print(f"  {i+1}. Query: {sample_out['query']}")
                    print(f"       Response: '{sample_out['response']}' (expected: {sample_out['expected']}) - {sample_out['time']:.3f}s [{correct_str}]")

            result = {
                "model_name": model_name,
                "model_type": model_type,
                "model_size": model_info["size"],
                "params_m": model_info.get("params_m"), # Use .get to handle missing key for XL (params_b)
                "params_b": model_info.get("params_b"),
                "load_time": load_time,
                "memory_used_gb": memory_used,
                "avg_eval_time": avg_eval_time,
                "correct_judgment_rate": correct_judgment_rate, # Updated field
                "projected_hours": projected_hours,
                "colab_recommendation": colab_rec,
                "colab_reason": colab_reason
            }

            self.results.append(result)

            # Cleanup
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return result

        except Exception as e:
            # Enhanced error handling for model loading failures
            print(f"❌ FAILED to test {model_name}: {str(e)[:100]}...") # Truncate long error messages
            print("  This often means the model is too large for the available memory or the model name/type is incorrect.")

            # Add a partial result for failed models to the report
            self.results.append({
                "model_name": model_name,
                "model_type": model_type,
                "model_size": model_info["size"],
                "params_m": model_info.get("params_m"),
                "params_b": model_info.get("params_b"),
                "load_time": float('inf'), # Indicate failure for load time
                "memory_used_gb": float('inf'), # Indicate unknown/exceeded memory
                "avg_eval_time": float('inf'), # Indicate failure for eval time
                "correct_judgment_rate": 0.0, # No correct judgments if failed
                "projected_hours": float('inf'),
                "colab_recommendation": "FAIL: Check Memory/Model Name",
                "colab_reason": str(e)[:50]
            })
            return None

    def run_quick_benchmark(self):
        """Run the optimized 30-minute benchmark"""
        print("🚀 QUICK UMBRELA BENCHMARK FOR COLAB PRO DECISION")
        print("🎯 Target: Evaluate TREC relevance assessment for multiple models")
        print("="*70)

        # System check
        system_info = self.check_system_resources()

        # Determine sample size to keep total time around 15-16 minutes
        # Sum of avg_eval_times for current models (estimated with XL) is ~11.071s
        # (11.071s * num_samples) / 60s/min = ~15 minutes => num_samples = 900 / 11.071 = ~81
        # With 11 models, 80 samples each: (roughly 45s per model test run) * 11 models / 60s/min = ~8.25 min
        sample_size = 80 # Aim for 80 samples per model to keep total time within reasonable limits

        print(f"\n🧪 Testing {len(self.test_models)} models with {sample_size} evaluations each")
        # Estimate based on a typical average run time for a model test, adjusted for number of models
        estimated_total_time_minutes = (len(self.test_models) * 45) / 60 # Rough average of 45 seconds per model test
        print(f"⏱️  Estimated total benchmark time: ~{estimated_total_time_minutes:.1f} minutes")

        start_benchmark = time.time()

        # Test each model
        for i, model_info in enumerate(self.test_models, 1):
            print(f"\n[{i}/{len(self.test_models)}] Testing {model_info['name']}")

            model_start = time.time()
            result = self.test_model_performance(model_info, sample_size)
            model_time = time.time() - model_start

            if result:
                print(f"✅ Model test completed in {model_time:.1f}s")
            else:
                print(f"❌ Model test failed after {model_time:.1f}s")

            # Brief cooldown to avoid immediate resource issues
            time.sleep(2) # Increased cooldown slightly

        total_time = time.time() - start_benchmark

        # Generate report
        self.generate_quick_report(system_info, total_time)

    def generate_quick_report(self, system_info, total_time):
        """Generate focused report for Colab Pro decision, including parameter counts"""
        print("\n" + "="*80)
        print("📊 COLAB PRO DECISION REPORT")
        print("="*80)

        if not self.results:
            print("❌ No results to analyze")
            return

        print(f"⏱️  Benchmark completed in {total_time/60:.1f} minutes")
        print(f"🖥️  System: {system_info['tier']}")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Sort by performance (fastest avg_eval_time first, handle inf values)
        sorted_results = sorted(self.results, key=lambda x: x["avg_eval_time"] if x["avg_eval_time"] != float('inf') else float('inf'))

        print(f"\n{'MODEL PERFORMANCE RANKING':^80}")
        print("-" * 80)
        print(f"{'Rank':<5} {'Model':<25} {'Parameters':<12} {'Speed':<8} {'Mem Used':<10} {'Full TREC Est':<15} {'Judgements %':<15} {'Colab Rec':<15}")
        print("-" * 80)

        for i, result in enumerate(sorted_results, 1):
            model_short = result["model_name"].split("/")[-1][:20]
            params_str = ""
            if result.get("params_m") is not None:
                params_str = f"{result['params_m']}M"
            elif result.get("params_b") is not None:
                params_str = f"{result['params_b']}B"

            mem_used_str = f"{result['memory_used_gb']:.1f}GB" if result['memory_used_gb'] != float('inf') else "N/A"
            avg_eval_time_str = f"{result['avg_eval_time']:.3f}s" if result['avg_eval_time'] != float('inf') else "N/A"
            projected_time_str = f"{result['projected_hours']:.1f}h" if result['projected_hours'] != float('inf') else "N/A"

            # Corrected f-string for the 'Judgements %' column
            correct_judgment_rate_val = result['correct_judgment_rate']
            correct_judgment_rate_display = f"{correct_judgment_rate_val:.1f}"
            if correct_judgment_rate_val == 0.0:
                suffix_for_display = ""
            elif correct_judgment_rate_val < 100.0:
                suffix_for_display = "%"
            else: # For 100.0%
                suffix_for_display = " " # Add a space for 100% to maintain alignment as per original intent

            # Ensure the padding aligns with the full string for Judgments %
            # The padding is applied to the combined string.
            formatted_judgement_percent_col = f"{correct_judgment_rate_display}{suffix_for_display:<14}"

            print(f"{i:<5} {model_short:<25} {params_str:<12} {avg_eval_time_str:<8} {mem_used_str:<10} {projected_time_str:<15} {formatted_judgement_percent_col} {result['colab_recommendation']:<15}")

        # Decision matrix
        print(f"\n{'DECISION MATRIX':^80}")
        print("-" * 80)

        free_ok_models = [r for r in sorted_results if "FREE" in r["colab_recommendation"] and r["projected_hours"] != float('inf')]
        pro_helpful = [r for r in sorted_results if "Helpful" in r["colab_recommendation"] and r["projected_hours"] != float('inf')]
        pro_recommended = [r for r in sorted_results if "Recommended" in r["colab_recommendation"] and r["projected_hours"] != float('inf')]
        pro_required = [r for r in sorted_results if "Required" in r["colab_recommendation"] and r["projected_hours"] != float('inf')]
        failed_models = [r for r in self.results if r["load_time"] == float('inf')] # Use self.results to get all including failed ones

        if free_ok_models:
            best_free = free_ok_models[0]
            print(f"✅ COLAB FREE is sufficient:")
            print(f"   • Best model: {best_free['model_name'].split('/')[-1]} ({best_free.get('params_m', '')}M{best_free.get('params_b', '')}B parameters)")
            print(f"   • Estimated time: {best_free['projected_hours']:.1f} hours")
            print(f"   • Memory usage: {best_free['memory_used_gb']:.1f}GB")
            print(f"   • Correct Judgment Rate: {best_free['correct_judgment_rate']:.1f}%")
            print(f"   • 💰 Cost: $0/month")

        if pro_recommended or pro_helpful:
            relevant_models = pro_recommended + pro_helpful
            # Filter out models that failed or are too slow for realistic Pro benefit
            relevant_models = [m for m in relevant_models if m["projected_hours"] < 24] # Only show if it's under 24 hours
            if relevant_models:
                best_pro = min(relevant_models, key=lambda x: x["projected_hours"])
                print(f"\n⚡ COLAB PRO would provide:")
                print(f"   • Better model options: {len(relevant_models)} models")
                print(f"   • Potentially faster completion: {best_pro['projected_hours']:.1f} hours with {best_pro['model_name'].split('/')[-1]} ({best_pro.get('params_m', '')}M{best_pro.get('params_b', '')}B parameters)")
                print(f"   • More reliable performance for larger models")
                print(f"   • 💸 Cost: $10/month")

        if pro_required:
            print(f"\n🔴 COLAB PRO+ needed for (memory-intensive models):")
            for model in pro_required:
                print(f"   • {model['model_name'].split('/')[-1]} ({model.get('params_m', '')}M{model.get('params_b', '')}B parameters): {model['colab_reason']}")

        if failed_models:
            print(f"\n❌ FAILED TO RUN (Likely memory issues or invalid model name):")
            for model in failed_models:
                print(f"   • {model['model_name'].split('/')[-1]} ({model.get('params_m', '')}M{model.get('params_b', '')}B parameters): {model['colab_reason']}")

        # Final recommendation
        print(f"\n{'🎯 FINAL RECOMMENDATION':^80}")
        print("=" * 80)

        if free_ok_models:
            fastest_free = free_ok_models[0]
            if fastest_free["projected_hours"] < 12:
                print("💡 VERDICT: Colab FREE is perfectly adequate for your current needs.")
                print(f"   ✅ You can complete a full TREC evaluation in {fastest_free['projected_hours']:.1f} hours using {fastest_free['model_name'].split('/')[-1]}.")
                print(f"   ✅ It achieves a correct judgment rate of {fastest_free['correct_judgment_rate']:.1f}%.")
                print(f"   💰 Save $10/month - use the free tier!")
            else:
                print("💡 VERDICT: Colab PRO recommended but not strictly required.")
                print(f"   ⚠️  Free tier will take {fastest_free['projected_hours']:.1f} hours with {fastest_free['model_name'].split('/')[-1]}, exceeding typical session limits.")
                print(f"   ⚡ Pro tier could significantly reduce this time and provide more stable performance.")
                print(f"   💸 Consider Pro if your time and consistent access are more valuable than $10/month.")
        else:
            print("💡 VERDICT: Colab PRO is necessary for any of the tested models.")
            print("   ❌ Free tier cannot handle the memory requirements of these models for benchmarking.")
            print("   ✅ A Pro tier ($10/month) is required for reasonable performance and to even run some models.")
            print("   💸 This investment is justified for these tasks.")

        # Command for your specific use case
        if free_ok_models:
            best_model_for_command = free_ok_models[0]["model_name"]
            print(f"\n📝 FOR YOUR UMBRELA COMMAND (using the best FREE tier compatible model):")
            print("-" * 70)
            print(f"# Your command for efficient testing on Colab Free:")
            print(f"!python running.py \\") # Changed to running.py as this is the current script name
            print(f"  --qrel dl19-passage \\") # These args would need to be parsed by running.py if you want to use them
            print(f"  --result_file output.txt \\")
            print(f"  --prompt_type bing \\")
            print(f"  --model {best_model_for_command} \\")
            print(f"  --few_shot_count 0 \\")
            print(f"  --device cuda \\")
            print(f"  --num_sample 1")
            print(f"# Estimated completion for full TREC evaluation with this model: {free_ok_models[0]['projected_hours']:.1f} hours")

        print("=" * 80)


# Run the quick benchmark
if __name__ == "__main__":
    benchmark = QuickUmbrelaBenchmark()
    benchmark.run_quick_benchmark()

```


```python
!python huggingfacerun.py
```

Now, eval for complete judgement using hugging face open-source models


```python
%%writefile output.txt
#test
```


```python
!nvidia-smi

```


```python
!python umbrela/hgfllm_judge.py --qrel dl19-passage --result_file output.txt --prompt_type bing --model Qwen/Qwen2-7B-Instruct --few_shot_count 0 --device cuda --num_sample 1
```
