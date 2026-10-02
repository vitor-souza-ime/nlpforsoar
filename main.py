from flask import Flask, request
import requests
import os
import glob
import tempfile
import time
import html
import re
import json
import statistics
from datetime import datetime
from werkzeug.utils import secure_filename

try:
    import soar_sml as sml
    SOAR_AVAILABLE = True
except ImportError:
    SOAR_AVAILABLE = False
    print("⚠️ WARNING: soar_sml not found. Soar execution tests are disabled.")

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ WARNING: scipy not found. Shapiro-Wilk testing is disabled.")


app = Flask(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
N_EXECUTIONS = 100
MAX_CYCLES = 10000
USE_RAG = True
TEMPO_REQ = 15
MAX_CONTEXT_CHARS = 200_000
MAX_OUTPUT_TOKENS = 20_000

# Replace with your API key or, preferably, use an environment variable.
key = os.getenv("OPENROUTER_API_KEY", "YOUR_APIKEY_HERE")

# AT(S) coefficients
ALPHA = 1
BETA = 1
GAMMA = 1
KAPPA = 1

AVAILABLE_MODELS = {
    "z-ai/glm-4.7": {
        "name": "GLM 4.7",
        "max_tokens": MAX_OUTPUT_TOKENS,
        "key": key,
    },
    "minimax/minimax-m2.1": {
        "name": "Minimax 2.1",
        "max_tokens": MAX_OUTPUT_TOKENS,
        "key": key,
    },
    "deepseek/deepseek-v3.2": {
        "name": "Deepseek 3.2",
        "max_tokens": MAX_OUTPUT_TOKENS,
        "key": key,
    },
}

SOURCE_FOLDER = "RAG"
CYCLES_DATA_FOLDER = "cycles_data"
os.makedirs(SOURCE_FOLDER, exist_ok=True)
os.makedirs(CYCLES_DATA_FOLDER, exist_ok=True)

last_generated_codes = {}
last_problem_type = None


def detect_problem_type(question):
    # Portuguese and English keywords are intentionally retained for bilingual input detection.
    question_lower = question.lower()

    wjp_keywords = [
        "jarro", "jarros", "litro", "litros", "capacidade", "volumes",
        "volume", "vazio", "vazios", "medir", "obter", "j1", "j2", "j3",
        "jug", "jugs", "water jug", "liter", "liters", "capacity", "empty",
        "measure",
    ]
    bw_keywords = [
        "bloco", "blocos", "mundo dos blocos", "pilha", "empilhar", "mesa",
        "sobre", "livre", "segurando", "block", "blocks", "blocks world",
        "stack", "table", "clear", "holding", "pick up", "put down", "unstack",
    ]
    hanoi_keywords = [
        "torre", "torres", "hanói", "hanoi", "disco", "discos", "pino",
        "pinos", "haste", "hastes", "mover", "transferir", "peg", "pegs",
        "rod", "rods", "tower", "towers", "disk", "disks",
    ]
    memory_keywords = [
        "memória semântica", "memória episódica", "memória procedural",
        "conceito", "conceitos", "memória", "semântica", "episódio", "episódios",
        "experiência", "evento", "eventos", "procedimento", "produção", "produções",
        "semantic memory", "episodic memory", "procedural memory", "memory",
        "concept", "episode", "experience", "event", "procedure", "production",
    ]
    rl_keywords = [
        "aprendizado por reforço", "reforço", "rl", "agente", "ambiente",
        "recompensa", "recompensas", "política", "políticas", "q-learning",
        "sarsa", "reinforcement learning", "agent", "environment", "reward",
        "rewards", "policy", "policies",
    ]

    wjp_count = sum(1 for k in wjp_keywords if k in question_lower)
    bw_count = sum(1 for k in bw_keywords if k in question_lower)
    hanoi_count = sum(1 for k in hanoi_keywords if k in question_lower)
    memory_count = sum(1 for k in memory_keywords if k in question_lower)
    rl_count = sum(1 for k in rl_keywords if k in question_lower)

    print(
        f"\n🔍 DETECTION: WJP:{wjp_count} BW:{bw_count} "
        f"Hanoi:{hanoi_count} Mem:{memory_count} RL:{rl_count}"
    )

    max_count = max(wjp_count, bw_count, hanoi_count, memory_count, rl_count)
    if max_count == 0:
        return "udf"
    if hanoi_count == max_count:
        return "hanoi"
    if wjp_count == max_count:
        return "wjp"
    if memory_count == max_count:
        return "mem"
    if rl_count == max_count:
        return "rl"
    return "bw"


def get_soar_files_by_type(problem_type):
    all_files = glob.glob(os.path.join(SOURCE_FOLDER, "*.soar"))
    print(f"\n🔍 Total .soar files found: {len(all_files)}")

    if problem_type == "udf":
        return sorted(all_files)

    filtered = [
        f for f in all_files
        if os.path.basename(f).lower().startswith(problem_type)
    ]
    return sorted(filtered)


def read_soar_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        print(f"❌ Error reading file {filepath}: {exc}")
        return None


def save_successful_code(code, problem_type, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[1].split(":")[0].replace("-", "_")
    new_filename = f"{problem_type}_{model_short}_{timestamp}.soar"
    filepath = os.path.join(SOURCE_FOLDER, new_filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ Code saved: {new_filename}")
        return new_filename
    except Exception as exc:
        print(f"❌ Error saving file: {exc}")
        return None


def save_cycles_data(cycles_list, model_name, problem_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[1].split(":")[0].replace("-", "_")
    filename = f"{problem_type}_{model_short}_{timestamp}_cycles.json"
    filepath = os.path.join(CYCLES_DATA_FOLDER, filename)

    data = {
        "timestamp": timestamp,
        "model": model_name,
        "problem_type": problem_type,
        "n_executions": len(cycles_list),
        "cycles": cycles_list,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"📊 Cycle data saved: {filename}")
        return filename
    except Exception as exc:
        print(f"❌ Error saving cycle data: {exc}")
        return None


def calculate_shapiro_wilk(cycles_list):
    if not SCIPY_AVAILABLE or not cycles_list or len(cycles_list) < 3:
        return None, None

    try:
        statistic, p_value = scipy_stats.shapiro(cycles_list)
        return statistic, p_value
    except Exception as exc:
        print(f"⚠️ Error calculating Shapiro-Wilk: {exc}")
        return None, None


def _strip_soar_comments(code):
    """Remove comments starting with # without affecting |...| symbols or strings."""
    cleaned_lines = []

    for line in code.splitlines():
        out = []
        in_pipe = False
        in_quote = False
        escaped = False

        for ch in line:
            if escaped:
                out.append(ch)
                escaped = False
                continue

            if ch == "\\":
                out.append(ch)
                escaped = True
                continue

            if ch == "|" and not in_quote:
                in_pipe = not in_pipe
                out.append(ch)
                continue

            if ch == '"' and not in_pipe:
                in_quote = not in_quote
                out.append(ch)
                continue

            if ch == "#" and not in_pipe and not in_quote:
                break

            out.append(ch)

        cleaned_lines.append("".join(out))

    return "\n".join(cleaned_lines)


def _contains_active_halt(text):
    """Detect an active (halt) action outside strings and |...| symbols."""
    in_pipe = False
    in_quote = False
    escaped = False
    i = 0

    while i < len(text):
        ch = text[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if ch == "\\":
            escaped = True
            i += 1
            continue

        if ch == "|" and not in_quote:
            in_pipe = not in_pipe
            i += 1
            continue

        if ch == '"' and not in_pipe:
            in_quote = not in_quote
            i += 1
            continue

        if ch == "(" and not in_pipe and not in_quote:
            match = re.match(r"\(\s*halt\s*\)", text[i:], flags=re.IGNORECASE)
            if match:
                return True

        i += 1

    return False


def find_halt_productions(code):
    """Return Soar productions that contain an active (halt) action."""
    if not code:
        return []

    clean_code = _strip_soar_comments(code)
    starts = list(
        re.finditer(
            r"(?im)^[ \t]*sp[ \t]*\{[ \t]*([^\s{}]+)",
            clean_code,
        )
    )

    halt_productions = []
    for index, match in enumerate(starts):
        block_start = match.start()
        block_end = (
            starts[index + 1].start()
            if index + 1 < len(starts)
            else len(clean_code)
        )
        production_block = clean_code[block_start:block_end]

        if _contains_active_halt(production_block):
            halt_productions.append(match.group(1))

    return halt_productions


def get_production_firing_count(agent, production_name):
    """Query a production firing count using the Soar CLI through SML."""
    try:
        result = agent.ExecuteCommandLine(
            f"production firing-counts {production_name}"
        )

        if result is None:
            return 0

        text = str(result)

        match = re.search(r"(?m)^\s*(\d+)\s*:", text)
        if match:
            return int(match.group(1))

        match = re.fullmatch(r"\s*(\d+)\s*", text)
        if match:
            return int(match.group(1))

        print(
            f"⚠️ Could not parse the firing count for "
            f"'{production_name}': {text!r}"
        )
        return 0

    except Exception as exc:
        print(
            f"⚠️ Error querying the firing count for "
            f"'{production_name}': {exc}"
        )
        return 0


def get_fired_halt_productions(agent, halt_productions):
    """Return productions containing (halt) that fired at least once."""
    fired = []

    for production_name in halt_productions:
        firing_count = get_production_firing_count(agent, production_name)
        if firing_count > 0:
            fired.append((production_name, firing_count))

    return fired


def calculate_at_score(min_cycles, avg_cycles, max_cycles, std_dev):
    """
    AT(S) is defined only for functionally admissible artifacts.
    There is no multiplicative SR term.
    """
    return (
        ALPHA * min_cycles
        + BETA * avg_cycles
        + GAMMA * max_cycles
        + KAPPA * std_dev
    )


def test_soar_code(
    code,
    model_name,
    n_executions=N_EXECUTIONS,
    max_cycles=MAX_CYCLES,
):
    """
    Execute a single Soar artifact n_executions times.

    An execution is successful only when an active production containing
    (halt), used in the experimental artifacts as the goal-test production,
    has a firing count > 0.

    The artifact is functionally admissible (SR=1) only when all
    n_executions executions are successful. Otherwise, SR=0 and AT(S)
    remains undefined.
    """
    if not SOAR_AVAILABLE:
        return None, None, None, None, None, None, None

    code_len = len(code) if code else 0
    print(
        f"\n🧪 TESTING {model_name} ({code_len:,} chars) - "
        f"{n_executions} executions..."
    )

    cycles_list = []
    success_count = 0

    halt_productions = find_halt_productions(code)
    if not halt_productions:
        print("❌ No active production containing (halt) was found.")
        return 0.0, None, None, None, None, None, None

    print("🎯 Detected halt productions:")
    for production_name in halt_productions:
        print(f"   • {production_name}")

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".soar",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
    except Exception as exc:
        print(f"❌ Error creating temporary file: {exc}")
        return None, None, None, None, None, None, None

    try:
        for i in range(n_executions):
            kernel = None

            try:
                kernel = sml.Kernel.CreateKernelInCurrentThread()
                if not kernel or kernel.HadError():
                    print(f"   ❌ Execution {i + 1}: failed to create kernel")
                    continue

                agent = kernel.CreateAgent(f"test_agent_{i}")
                if not agent:
                    print(f"   ❌ Execution {i + 1}: failed to create agent")
                    continue

                loaded = agent.LoadProductions(tmp_path)
                if not loaded:
                    print(f"   ❌ Execution {i + 1}: failed to load productions")
                    continue

                agent.RunSelf(max_cycles)
                cycles = agent.GetDecisionCycleCounter()

                fired_halt_rules = get_fired_halt_productions(
                    agent,
                    halt_productions,
                )

                if fired_halt_rules:
                    success_count += 1
                    cycles_list.append(cycles)
                    fired_names = ", ".join(
                        f"{name} (count={count})"
                        for name, count in fired_halt_rules
                    )
                    print(
                        f"   ✅ Execution {i + 1}/{n_executions}: "
                        f"goal-halt fired -> {fired_names} | cycles={cycles}"
                    )
                else:
                    stop_reason = (
                        "cycle limit reached"
                        if cycles >= max_cycles
                        else "stopped without firing a halt production"
                    )
                    print(
                        f"   ❌ Execution {i + 1}/{n_executions}: "
                        f"{stop_reason} | cycles={cycles}"
                    )

            except Exception as exc:
                print(
                    f"   ❌ Execution {i + 1}/{n_executions}: "
                    f"execution error: {exc}"
                )

            finally:
                if kernel:
                    try:
                        kernel.Shutdown()
                    except Exception:
                        pass

        success_rate = (success_count / n_executions) * 100.0

        # Only executions with a verified goal-halt are included in the cycle vector.
        if cycles_list and last_problem_type:
            save_cycles_data(cycles_list, model_name, last_problem_type)

        # Statistics are used for AT(S) only when SR=1 (100/100).
        if success_count == n_executions:
            min_cycles = min(cycles_list)
            max_cycles_result = max(cycles_list)
            avg_cycles = sum(cycles_list) / len(cycles_list)
            std_dev = (
                statistics.stdev(cycles_list)
                if len(cycles_list) > 1
                else 0.0
            )
            sw_statistic, sw_pvalue = calculate_shapiro_wilk(cycles_list)
        else:
            min_cycles = None
            max_cycles_result = None
            avg_cycles = None
            std_dev = None
            sw_statistic = None
            sw_pvalue = None

        print(
            f"✅ {model_name}: success={success_count}/{n_executions} "
            f"({success_rate:.1f}%)"
        )

        return (
            success_rate,
            min_cycles,
            max_cycles_result,
            avg_cycles,
            std_dev,
            sw_statistic,
            sw_pvalue,
        )

    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def build_context_from_files(soar_files, max_chars):
    print(f"\n📦 Files received: {len(soar_files)}")
    print(f"   Character limit: {max_chars:,}")

    if not soar_files:
        return "", []

    context_parts = []
    total_chars = 0
    included_files = []

    for filepath in soar_files:
        filename = os.path.basename(filepath)
        content = read_soar_file(filepath)

        if not content:
            continue

        file_block = (
            f"\n{'=' * 60}\nFILE: {filename}\n"
            f"{'=' * 60}\n{content}\n"
        )
        block_size = len(file_block)

        if total_chars + block_size > max_chars:
            break

        context_parts.append(file_block)
        total_chars += block_size
        included_files.append(filename)

    return "\n".join(context_parts), included_files


def generate_soar_prompt(user_question, context, included_files, problem_type):
    files_list = "\n".join(f"- {f}" for f in included_files)
    problem_description = {
        "wjp": "Water Jug Problem",
        "bw": "Blocks World Problem",
        "hanoi": "Tower of Hanoi Problem",
        "mem": "Memory Problem",
        "rl": "Reinforcement Learning",
        "udf": "Soar cognitive architecture",
    }.get(problem_type, "Soar cognitive architecture")

    return f"""You are a Soar expert assistant. You will receive:
1. A user question about {problem_description}
2. Multiple existing Soar code files from the knowledge base (filtered for this problem type)

Your task:
- Analyze the user's question carefully
- Review all provided Soar files for context and patterns
- Generate the complete updated Soar code based on the request
- Return ONLY the complete Soar code
- Do NOT add explanations, markdown formatting, or comments outside the code
- Do NOT include ```soar``` or any markdown tags
- Return pure Soar code ready to be saved directly as a .soar file
- Include all necessary rules for the solution
- Use the names given in the prompt
- Leave the prompt request as a comment in the code in Soar in the first line.

USER QUESTION:
{user_question}

KNOWLEDGE BASE FILES ({len(included_files)} files - filtered for {problem_type.upper()}):
{files_list}

SOAR CODE FILES:
{context}

Return the complete Soar code below (no explanations, no markdown):"""


def call_openrouter_single(prompt, model_name):
    headers = {
        "Authorization": f"Bearer {AVAILABLE_MODELS[model_name]['key']}",
        "Content-Type": "application/json",
    }
    max_tokens = AVAILABLE_MODELS[model_name]["max_tokens"]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.15,
        "max_tokens": max_tokens,
    }

    start_time = time.time()

    try:
        print(f"📤 Sending request to: {model_name}")
        response_http = requests.post(
            OPENROUTER_URL,
            json=payload,
            headers=headers,
            timeout=120,
        )
        elapsed = time.time() - start_time

        if response_http.status_code != 200:
            return (
                model_name,
                None,
                f"API error: {response_http.status_code}",
                elapsed,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        response_json = response_http.json()
        response = response_json["choices"][0]["message"].get("content", "")

        if not response or not response.strip():
            response = response_json["choices"][0]["message"].get(
                "reasoning",
                "",
            )

        response = response.replace("```soar", "").replace("```", "").strip()
        code_len = len(response) if response else 0
        print(
            f"✅ {model_name} completed in {elapsed:.1f}s "
            f"({code_len:,} chars)"
        )

        (
            success_rate,
            min_cycles,
            max_cycles_result,
            avg_cycles,
            std_dev,
            sw_stat,
            sw_pval,
        ) = test_soar_code(response, model_name)

        return (
            model_name,
            response,
            None,
            elapsed,
            success_rate,
            min_cycles,
            max_cycles_result,
            avg_cycles,
            std_dev,
            sw_stat,
            sw_pval,
        )

    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"❌ {model_name} exception: {exc}")
        return (
            model_name,
            None,
            str(exc),
            elapsed,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def call_all_models_sequential(prompt):
    print(
        f"\n🚀 PROCESSING {len(AVAILABLE_MODELS)} MODELS SEQUENTIALLY"
    )
    results = {}

    for i, model_name in enumerate(AVAILABLE_MODELS.keys(), 1):
        (
            model_name_ret,
            code,
            error,
            elapsed,
            success_rate,
            min_cycles,
            max_cycles_result,
            avg_cycles,
            std_dev,
            sw_stat,
            sw_pval,
        ) = call_openrouter_single(prompt, model_name)

        sr_binary = (
            1
            if success_rate is not None and success_rate == 100.0
            else 0
        )

        at_score = None
        if sr_binary == 1:
            at_score = calculate_at_score(
                min_cycles,
                avg_cycles,
                max_cycles_result,
                std_dev,
            )

        results[model_name_ret] = {
            "code": code,
            "error": error,
            "time": elapsed,
            "success_rate": success_rate,
            "SR": sr_binary,
            "at_score": at_score,
            "min_cycles": min_cycles,
            "max_cycles": max_cycles_result,
            "avg_cycles": avg_cycles,
            "std_dev": std_dev,
            "sw_statistic": sw_stat,
            "sw_pvalue": sw_pval,
        }

        print(
            f"✅ Result: model={model_name_ret}, SR={sr_binary}, "
            f"success_rate={success_rate}, AT(S)={at_score}, error={error}"
        )

        if i < len(AVAILABLE_MODELS):
            time.sleep(TEMPO_REQ)

    return results


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file provided", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    if not file.filename.endswith(".soar"):
        return "Only .soar files are allowed", 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = secure_filename(file.filename).replace(".soar", "")
        new_filename = f"udf_{original_name}_{timestamp}.soar"
        filepath = os.path.join(SOURCE_FOLDER, new_filename)
        file.save(filepath)
        return f"File uploaded successfully: {new_filename}", 200
    except Exception as exc:
        return f"Error uploading file: {exc}", 500


def _database_counts():
    all_files = glob.glob(os.path.join(SOURCE_FOLDER, "*.soar"))
    counts = {
        "wjp": len([
            f for f in all_files
            if os.path.basename(f).lower().startswith("wjp")
        ]),
        "bw": len([
            f for f in all_files
            if os.path.basename(f).lower().startswith("bw")
        ]),
        "hanoi": len([
            f for f in all_files
            if os.path.basename(f).lower().startswith("hanoi")
        ]),
        "mem": len([
            f for f in all_files
            if os.path.basename(f).lower().startswith("mem")
        ]),
        "rl": len([
            f for f in all_files
            if os.path.basename(f).lower().startswith("rl")
        ]),
    }
    counts["total"] = len(all_files)
    counts["other"] = (
        counts["total"]
        - counts["wjp"]
        - counts["bw"]
        - counts["hanoi"]
        - counts["mem"]
        - counts["rl"]
    )
    return counts


@app.route("/", methods=["GET", "POST"])
def index():
    global last_generated_codes, last_problem_type

    user_question = ""
    error_message = ""
    success_message = ""
    detected_type = ""

    counts = _database_counts()

    if request.method == "POST":
        action = request.form.get("action", "generate")

        if action == "confirm_success":
            model_to_save = request.form.get("model_to_save")

            if model_to_save and model_to_save in last_generated_codes:
                result = last_generated_codes[model_to_save]
                code = result.get("code")
                success_rate = result.get("success_rate")

                if not code:
                    error_message = "No code available to save"
                elif success_rate != 100.0:
                    error_message = (
                        "❌ Only functionally admissible code "
                        "(100/100 successful executions, SR=1) "
                        "can be added to the RAG database"
                    )
                else:
                    saved_filename = save_successful_code(
                        code,
                        last_problem_type,
                        model_to_save,
                    )
                    if saved_filename:
                        model_info = AVAILABLE_MODELS.get(model_to_save, {})
                        model_display = model_info.get(
                            "name",
                            model_to_save,
                        )
                        success_message = (
                            f"✅ Code from {model_display} added to database: "
                            f"{saved_filename}"
                        )
                        counts = _database_counts()
                    else:
                        error_message = "Failed to save code"

        elif action == "generate":
            user_question = request.form.get("user_question", "").strip()

            if not user_question:
                error_message = "Please enter a question."
            else:
                detected_type = detect_problem_type(user_question)
                last_problem_type = detected_type

                if USE_RAG:
                    soar_files = get_soar_files_by_type(detected_type)
                    if not soar_files:
                        error_message = (
                            f"No .soar files found for problem type "
                            f"'{detected_type}'"
                        )
                    else:
                        context, included_files = build_context_from_files(
                            soar_files,
                            MAX_CONTEXT_CHARS,
                        )
                        if not context:
                            error_message = "Could not read any .soar files"
                        else:
                            prompt = generate_soar_prompt(
                                user_question,
                                context,
                                included_files,
                                detected_type,
                            )
                            last_generated_codes = call_all_models_sequential(
                                prompt
                            )
                else:
                    # Controlled No-RAG condition: same prompt and settings,
                    # with only retrieved files/context removed.
                    context = ""
                    included_files = []
                    prompt = generate_soar_prompt(
                        user_question,
                        context,
                        included_files,
                        detected_type,
                    )
                    last_generated_codes = call_all_models_sequential(prompt)

    problem_names = {
        "wjp": "Water Jug",
        "bw": "Blocks World",
        "hanoi": "Tower of Hanoi",
        "mem": "Memory",
        "rl": "RL",
        "udf": "General",
    }

    results_html = ""
    if last_generated_codes:
        results_html = '<hr class="my-4"><h4>📊 Results from ALL Models:</h4>'

        for model_id, result in last_generated_codes.items():
            code = result.get("code")
            code_len = len(code) if code else 0
            model_info = AVAILABLE_MODELS.get(model_id, {})
            model_name = model_info.get("name", model_id)
            max_tokens = model_info.get("max_tokens", MAX_OUTPUT_TOKENS)
            model_short = model_id.split("/")[1].split(":")[0].replace("-", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{last_problem_type}_{model_short}_{timestamp}.soar"

            if code:
                success_rate = result.get("success_rate")
                sr_binary = result.get("SR", 0)
                at_score = result.get("at_score")
                min_c = result.get("min_cycles")
                max_c = result.get("max_cycles")
                avg_c = result.get("avg_cycles")
                std_c = result.get("std_dev")

                functionally_admissible = (
                    SOAR_AVAILABLE
                    and success_rate is not None
                    and success_rate == 100.0
                    and sr_binary == 1
                )

                if functionally_admissible:
                    can_add_to_db = True
                    disable_reason = ""
                else:
                    can_add_to_db = False
                    disable_reason = (
                        "Only functionally admissible code "
                        "(SR=1; 100/100 successful executions) can be added"
                    )

                show_download_copy = functionally_admissible

                if success_rate is None:
                    stats_html = (
                        '<div class="mt-2 alert alert-warning p-2"><small>'
                        "⚠️ Soar not available - tests skipped"
                        "</small></div>"
                    )
                else:
                    stats_badge = (
                        "success"
                        if success_rate == 100.0
                        else "danger"
                    )
                    min_str = str(min_c) if min_c is not None else "N/A"
                    max_str = str(max_c) if max_c is not None else "N/A"
                    avg_str = f"{avg_c:.1f}" if avg_c is not None else "N/A"
                    std_str = f"{std_c:.1f}" if std_c is not None else "N/A"
                    at_str = (
                        f"{at_score:.2f}"
                        if at_score is not None
                        else "undefined"
                    )

                    stats_html = f'''<div class="mt-2 p-2 bg-light rounded"><small>
                        <strong>📈 Test Results ({N_EXECUTIONS} runs, {code_len:,} chars):</strong><br>
                        <span class="badge bg-{stats_badge}">Execution success: {success_rate:.1f}%</span>
                        <span class="badge bg-{stats_badge}">SR: {sr_binary}</span><br>
                        <span class="badge bg-info">Min: {min_str} cycles</span>
                        <span class="badge bg-warning">Max: {max_str} cycles</span>
                        <span class="badge bg-secondary">Avg: {avg_str} cycles</span>
                        <span class="badge bg-primary">StdDev: {std_str}</span><br>
                        <strong>🎯 AT(S):</strong>
                        <span class="badge bg-{stats_badge}">{at_str}</span>
                    </small></div>'''

                if can_add_to_db:
                    add_button = (
                        '<form method="POST" style="display:inline;">'
                        '<input type="hidden" name="action" value="confirm_success">'
                        f'<input type="hidden" name="model_to_save" value="{html.escape(model_id)}">'
                        '<button type="submit" class="btn btn-success w-100">'
                        "✅ Add to Database</button></form>"
                    )
                else:
                    add_button = (
                        '<button type="button" class="btn btn-danger w-100" '
                        f'disabled title="{html.escape(disable_reason)}">'
                        f"🚫 {html.escape(disable_reason)}</button>"
                    )

                if show_download_copy:
                    download_copy_buttons = (
                        f'<button onclick="downloadCode_{model_short}()" '
                        f'class="btn btn-primary">💾 Download {filename}</button>'
                        f'<button onclick="copyCode_{model_short}()" '
                        f'class="btn btn-secondary">📋 Copy to Clipboard</button>'
                    )
                else:
                    download_copy_buttons = (
                        '<div class="alert alert-info p-2 mb-2"><small>'
                        "💡 Download and Copy buttons are available only for "
                        "functionally admissible artifacts (SR=1)."
                        "</small></div>"
                    )

                code_json = json.dumps(code)
                filename_json = json.dumps(filename)

                results_html += f'''<div class="card mb-3 border-success">
                    <div class="card-header bg-success text-white">
                        <strong>✅ {html.escape(model_name)}</strong> - Generated in {result["time"]:.1f}s |
                        Max: {max_tokens:,} tokens | {code_len:,} chars
                    </div>
                    <div class="card-body">
                        {stats_html}
                        <div class="d-grid gap-2 mt-3">
                            {download_copy_buttons}
                            {add_button}
                        </div>
                    </div>
                </div>
                <script>
                (function() {{
                    const code = {code_json};
                    const filename = {filename_json};
                    window["copyCode_{model_short}"] = function() {{
                        navigator.clipboard.writeText(code).then(() => {{
                            alert("✅ Copied!");
                        }}).catch(err => {{
                            alert("❌ Error: " + err);
                        }});
                    }};
                    window["downloadCode_{model_short}"] = function() {{
                        const blob = new Blob([code], {{type: "text/plain;charset=utf-8"}});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = filename;
                        document.body.appendChild(a);
                        a.click();
                        setTimeout(() => {{
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                        }}, 100);
                    }};
                }})();
                </script>'''

            else:
                error_escaped = html.escape(str(result.get("error", "Unknown error")))
                results_html += (
                    '<div class="card mb-3 border-danger">'
                    '<div class="card-header bg-danger text-white">'
                    f'<strong>❌ {html.escape(model_name)}</strong> - Failed in '
                    f'{result["time"]:.1f}s</div>'
                    '<div class="card-body">'
                    f'<p class="text-danger mb-0">Error: {error_escaped}</p>'
                    '</div></div>'
                )

    scipy_warning = ""
    if not SCIPY_AVAILABLE:
        scipy_warning = (
            '<div class="alert alert-warning">⚠️ '
            '<strong>scipy not installed</strong> - Shapiro-Wilk test disabled.'
            '</div>'
        )

    database_html = (
        f'<div class="alert alert-info"><strong>📊 Database:</strong> '
        f'WJP: {counts["wjp"]} | BW: {counts["bw"]} | '
        f'Hanoi: {counts["hanoi"]} | Memory: {counts["mem"]} | '
        f'RL: {counts["rl"]} | Other: {counts["other"]} | '
        f'Total: {counts["total"]}</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NLP for Soar</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body{{background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);min-height:100vh;padding:20px}}
.main-card{{background:white;border-radius:15px;max-width:1400px;margin:0 auto}}
a,button,.btn{{text-decoration:none!important}}
pre{{margin:0}}
</style>
</head>
<body>
<div class="container mt-4">
<div class="main-card">
<div class="card-header bg-primary text-white p-4">
<h1 class="h3">🤖 NLP for Soar - Sequential Multi-LLM Processing</h1>
<p class="mb-0">⚡ {len(AVAILABLE_MODELS)} models sequentially | 🧪 {N_EXECUTIONS} executions per artifact</p>
</div>
<div class="card-body p-4">
{scipy_warning}
{database_html}
{f'<div class="alert alert-success">{success_message}</div>' if success_message else ''}
{f'<div class="alert alert-danger">{error_message}</div>' if error_message else ''}
{f'<div class="alert alert-info">🎯 Detected: <strong>{problem_names.get(detected_type,"Unknown")}</strong> ({detected_type})</div>' if detected_type else ''}

<div class="mb-4">
<label class="form-label fw-bold">📤 Upload .soar file to database:</label>
<input type="file" id="fileInput" class="form-control" accept=".soar">
<button onclick="uploadFile()" class="btn btn-secondary mt-2" type="button">⬆️ Upload File</button>
<div id="uploadStatus" class="mt-2"></div>
</div>

<hr class="my-4">
<form method="POST" id="generateForm">
<input type="hidden" name="action" value="generate">
<div class="mb-4">
<label class="form-label fw-bold">❓ Your Question:</label>
<textarea class="form-control" name="user_question" rows="4" required>{html.escape(user_question)}</textarea>
<small class="text-muted">Domains: Water Jug, Blocks World, Tower of Hanoi, Memory, RL and General.</small>
</div>
<button type="submit" class="btn btn-primary btn-lg w-100" id="generateBtn">
🚀 Generate & Test with {len(AVAILABLE_MODELS)} Models (Sequential)
</button>
</form>
{results_html}
</div>
</div>
</div>
<script>
function uploadFile() {{
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    if (!fileInput.files || fileInput.files.length === 0) {{
        uploadStatus.innerHTML = '<div class="alert alert-warning">Please select a file</div>';
        return;
    }}
    const file = fileInput.files[0];
    if (!file.name.endsWith('.soar')) {{
        uploadStatus.innerHTML = '<div class="alert alert-danger">Only .soar files are allowed</div>';
        return;
    }}
    const formData = new FormData();
    formData.append('file', file);
    fetch('/upload', {{method:'POST', body:formData}})
        .then(response => response.text())
        .then(data => {{
            uploadStatus.innerHTML = '<div class="alert alert-success">' + data + '</div>';
            setTimeout(() => location.reload(), 1500);
        }})
        .catch(error => {{
            uploadStatus.innerHTML = '<div class="alert alert-danger">Error: ' + error + '</div>';
        }});
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("🚀 NLP for Soar - Sequential Multi-LLM Processing")
    print(f"📁 Source folder: {SOURCE_FOLDER}/")
    print(f"📊 Cycles data folder: {CYCLES_DATA_FOLDER}/")
    print(f"🤖 {len(AVAILABLE_MODELS)} models process sequentially")
    print(
        f"🧪 Auto-tests each artifact {N_EXECUTIONS} times "
        f"(max {MAX_CYCLES} cycles)"
    )
    print(
        "✅ Soar available - testing enabled"
        if SOAR_AVAILABLE
        else "⚠️ Soar NOT available - testing disabled"
    )
    print("🌐 http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
