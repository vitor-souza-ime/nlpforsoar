from flask import Flask, request, jsonify

import requests, os, glob, tempfile, time, html

from datetime import datetime

from werkzeug.utils import secure_filename

import statistics

import json

import re



try:

    #import Python_sml_ClientInterface as sml

    import soar_sml as sml

    SOAR_AVAILABLE = True

except ImportError:

    SOAR_AVAILABLE = False

    print("⚠️ AVISO: Python_sml_ClientInterface não encontrado. Execução de testes desabilitada.")



try:

    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True

except ImportError:

    SCIPY_AVAILABLE = False

    print("⚠️ AVISO: scipy não encontrado. Teste de Shapiro-Wilk desabilitado.")



app = Flask(__name__)

#OPENROUTER_API_KEY = "REDACTED_OPENROUTER_KEY"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

N_EXECUTIONS = 100

MAX_CYCLES = 10000

USE_RAG = True

TEMPO_REQ=15

MAX_CONTEXT_CHARS = 200_000

MAX_OUTPUT_TOKENS = 20_000

key="YOUR_APIKEY_HERE"

# Constantes da equação AT(S)

ALPHA = 1

BETA = 1

GAMMA = 1

KAPPA = 1  # Para o desvio padrão

DELTA = 1

EPSILON = 1



AVAILABLE_MODELS = {

    "z-ai/glm-4.7": 

    {"name": "GLM 4.7", "max_tokens": MAX_OUTPUT_TOKENS, 

     "key":key},



    "minimax/minimax-m2.1":

    {"name": "Minimax 2.1", "max_tokens": MAX_OUTPUT_TOKENS, 

     "key":key},



    "deepseek/deepseek-v3.2":

    {"name": "Deepseek 3.2", "max_tokens": MAX_OUTPUT_TOKENS, 

     "key":key},

    #"mistralai/devstral-2512:free": {"name": "Devstral 2512 (default)", "max_tokens": MAX_CONTEXT_CHARS/4}, 

    #"google/gemma-3-27b-it:free": {"name": "Gemma 3 27B", "max_tokens": MAX_CONTEXT_CHARS/4},

    #"openai/gpt-oss-20b:free":{"name": "GPT-OSS 20B","max_tokens": MAX_CONTEXT_CHARS/4 },

    #"qwen/qwen3-coder:free": 

    #{"name": "Qwen 3 Coder", "max_tokens": MAX_CONTEXT_CHARS/4,

    # "key":"REDACTED_OPENROUTER_KEY"},



    #"tngtech/tng-r1t-chimera:free":

    #{"name": "TNG R1T Chimera","max_tokens": MAX_CONTEXT_CHARS/4,

    #"key":"REDACTED_OPENROUTER_KEY"},



    #"tngtech/deepseek-r1t2-chimera:free": 

    #{"name": "DeepSeek R1T2 Chimera","max_tokens": MAX_CONTEXT_CHARS/4,

    #"key":"REDACTED_OPENROUTER_KEY"},   

    #"arcee-ai/trinity-large-preview:free":

    #{"name": "Trinity Large Preview","max_tokens": MAX_CONTEXT_CHARS/4,

    #"key":"REDACTED_OPENROUTER_KEY"},

    #"mistralai/mistral-small-3.1-24b-instruct:free": {"name": "Mistral Small 3.1 24B","max_tokens": MAX_CONTEXT_CHARS/4 },

    #"meta-llama/llama-3.2-3b-instruct:free": 

    #{"name": "Llama 3.2 3B",

    # "max_tokens": MAX_CONTEXT_CHARS/4,

    # "key":"REDACTED_OPENROUTER_KEY"},

    #}

}



SOURCE_FOLDER = 'RAG'

CYCLES_DATA_FOLDER = 'cycles_data'

os.makedirs(SOURCE_FOLDER, exist_ok=True)

os.makedirs(CYCLES_DATA_FOLDER, exist_ok=True)



last_generated_codes = {}

last_problem_type = None



def detect_problem_type(question):

    question_lower = question.lower()

    wjp_keywords = ['jarro', 'jarros', 'litro', 'litros', 'capacidade', 'volumes', 'volume', 'vazio', 'vazios', 'medir', 'obter', 'j1', 'j2', 'j3', 'jug', 'jugs', 'water jug', 'liter', 'liters', 'capacity', 'empty', 'measure']

    bw_keywords = ['bloco', 'blocos', 'mundo dos blocos', 'pilha', 'empilhar', 'mesa', 'sobre', 'livre', 'segurando', 'block', 'blocks', 'blocks world', 'stack', 'table', 'clear', 'holding', 'pick up', 'put down', 'unstack']

    hanoi_keywords = ['torre', 'torres', 'hanói', 'hanoi', 'disco', 'discos', 'pino', 'pinos', 'haste', 'hastes', 'mover', 'transferir', 'peg', 'pegs', 'rod', 'rods', 'tower', 'towers', 'disk', 'disks']

    memory_keywords = ['memória semântica', 'memória episódica', 'memória procedural', 'conceito', 'conceitos', 'memória', 'semântica', 'episódio', 'episódios', 'experiência', 'evento', 'eventos', 'procedimento', 'produção', 'produções', 'semantic memory', 'episodic memory', 'procedural memory', 'memory', 'concept', 'episode', 'experience', 'event', 'procedure', 'production']

    rl_keywords = ['aprendizado por reforço', 'reforço', 'rl', 'agente', 'ambiente', 'recompensa', 'recompensas', 'política', 'políticas', 'q-learning', 'sarsa', 'reinforcement learning', 'agent', 'environment', 'reward', 'rewards', 'policy', 'policies']

    wjp_count = sum(1 for k in wjp_keywords if k in question_lower)

    bw_count = sum(1 for k in bw_keywords if k in question_lower)

    hanoi_count = sum(1 for k in hanoi_keywords if k in question_lower)

    memory_count = sum(1 for k in memory_keywords if k in question_lower)

    rl_count = sum(1 for k in rl_keywords if k in question_lower)

    print(f"\n🔍 DETECÇÃO: WJP:{wjp_count} BW:{bw_count} Hanoi:{hanoi_count} Mem:{memory_count} RL:{rl_count}")

    max_count = max(wjp_count, bw_count, hanoi_count, memory_count, rl_count)

    if max_count == 0: return 'udf'

    elif hanoi_count == max_count: return 'hanoi'

    elif wjp_count == max_count: return 'wjp'

    elif memory_count == max_count: return 'mem'

    elif rl_count == max_count: return 'rl'

    else: return 'bw'



def get_soar_files_by_type(problem_type):

    all_files = glob.glob(os.path.join(SOURCE_FOLDER, '*.soar'))

    print(f"\n🔍 DEBUG get_soar_files_by_type:")

    print(f"   Total arquivos .soar encontrados: {len(all_files)}")

    if problem_type == 'udf':

        print(f"   Tipo UDF detectado - retornando TODOS os {len(all_files)} arquivos")

        return sorted(all_files)

    filtered = [f for f in all_files if os.path.basename(f).lower().startswith(problem_type)]

    print(f"   Tipo '{problem_type}' detectado - filtrados {len(filtered)} arquivos")

    for f in filtered:

        print(f"      • {os.path.basename(f)}")

    return sorted(filtered)



def read_soar_file(filepath):

    try:

        with open(filepath, 'r', encoding='utf-8') as f: return f.read()

    except Exception as e:

        print(f"❌ Erro ao ler arquivo {filepath}: {e}")

        return None



def save_successful_code(code, problem_type, model_name):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_short = model_name.split('/')[1].split(':')[0].replace('-', '_')

    new_filename = f"{problem_type}_{model_short}_{timestamp}.soar"

    filepath = os.path.join(SOURCE_FOLDER, new_filename)

    try:

        with open(filepath, 'w', encoding='utf-8') as f: f.write(code)

        print(f"✅ Código salvo: {new_filename}")

        return new_filename

    except Exception as e:

        print(f"❌ Erro ao salvar: {e}")

        return None



def save_cycles_data(cycles_list, model_name, problem_type):

    """Salva os dados de ciclos em um arquivo JSON para análise posterior"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_short = model_name.split('/')[1].split(':')[0].replace('-', '_')

    filename = f"{problem_type}_{model_short}_{timestamp}_cycles.json"

    filepath = os.path.join(CYCLES_DATA_FOLDER, filename)

    data = {

        "timestamp": timestamp,

        "model": model_name,

        "problem_type": problem_type,

        "n_executions": len(cycles_list),

        "cycles": cycles_list

    }

    try:

        with open(filepath, 'w', encoding='utf-8') as f:

            json.dump(data, f, indent=2)

        print(f"📊 Dados de ciclos salvos: {filename}")

        return filename

    except Exception as e:

        print(f"❌ Erro ao salvar dados de ciclos: {e}")

        return None



def calculate_shapiro_wilk(cycles_list):

    """Calcula o teste de Shapiro-Wilk para normalidade dos dados"""

    if not SCIPY_AVAILABLE or not cycles_list or len(cycles_list) < 3:

        return None, None

    try:

        statistic, p_value = scipy_stats.shapiro(cycles_list)

        return statistic, p_value

    except Exception as e:

        print(f"⚠️ Erro ao calcular Shapiro-Wilk: {e}")

        return None, None





def _strip_soar_comments(code):
    """Remove comentários iniciados por # sem afetar texto entre |...| ou aspas."""
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
    """Detecta (halt) fora de strings delimitadas por |...| ou aspas."""
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
    """Retorna os nomes das produções Soar que contêm uma ação (halt) ativa."""
    if not code:
        return []

    clean_code = _strip_soar_comments(code)
    starts = list(re.finditer(
        r"(?im)^[ \t]*sp[ \t]*\{[ \t]*([^\s{}]+)",
        clean_code,
    ))

    halt_productions = []

    for index, match in enumerate(starts):
        block_start = match.start()
        block_end = starts[index + 1].start() if index + 1 < len(starts) else len(clean_code)
        production_block = clean_code[block_start:block_end]

        if _contains_active_halt(production_block):
            halt_productions.append(match.group(1))

    return halt_productions


def get_production_firing_count(agent, production_name):
    """Consulta o firing count de uma produção usando a CLI do Soar via SML."""
    try:
        result = agent.ExecuteCommandLine(
            f"production firing-counts {production_name}"
        )

        if result is None:
            return 0

        text = str(result)

        # Saída típica do Soar: "  1: production-name"
        match = re.search(r"(?m)^\s*(\d+)\s*:", text)
        if match:
            return int(match.group(1))

        # Fallback conservador para wrappers que retornem apenas o número.
        match = re.fullmatch(r"\s*(\d+)\s*", text)
        if match:
            return int(match.group(1))

        print(
            f"⚠️ Não foi possível interpretar firing-count de "
            f"'{production_name}': {text!r}"
        )
        return 0

    except Exception as e:
        print(
            f"⚠️ Erro ao consultar firing-count de "
            f"'{production_name}': {e}"
        )
        return 0


def get_fired_halt_productions(agent, halt_productions):
    """Retorna as produções com (halt) que efetivamente dispararam na execução."""
    fired = []

    for production_name in halt_productions:
        firing_count = get_production_firing_count(agent, production_name)
        if firing_count > 0:
            fired.append((production_name, firing_count))

    return fired


def test_soar_code(code, model_name, n_executions=N_EXECUTIONS, max_cycles=MAX_CYCLES):

    if not SOAR_AVAILABLE: return None, None, None, None, None, None, None

    code_len = len(code) if code else 0

    print(f"\n🧪 TESTANDO {model_name} ({code_len:,} chars) - {n_executions} execuções...")

    cycles_list, success_count = [], 0

    halt_productions = find_halt_productions(code)

    if not halt_productions:
        print("❌ Nenhuma produção ativa contendo (halt) foi encontrada.")
        print("   O artefato será considerado sem execuções bem-sucedidas.")
        return 0.0, None, None, None, 0.0, None, None

    print("🎯 Produções de término detectadas:")
    for production_name in halt_productions:
        print(f"   • {production_name}")

    try:

        with tempfile.NamedTemporaryFile(mode='w', suffix='.soar', delete=False, encoding='utf-8') as tmp:

            tmp.write(code)

            tmp_path = tmp.name

    except Exception as e:

        print(f"❌ Erro ao criar arquivo temporário: {e}")

        return None, None, None, None, None, None, None

    try:

        for i in range(n_executions):

            kernel = None

            try:

                kernel = sml.Kernel.CreateKernelInCurrentThread()

                if not kernel or kernel.HadError(): continue

                agent = kernel.CreateAgent(f"test_agent_{i}")

                if not agent:

                    if kernel: kernel.Shutdown(); del kernel

                    continue

                result = agent.LoadProductions(tmp_path)

                if not result:

                    if kernel: kernel.Shutdown(); del kernel

                    continue

                agent.RunSelf(max_cycles)

                cycles = agent.GetDecisionCycleCounter()

                fired_halt_rules = get_fired_halt_productions(
                    agent, halt_productions
                )

                if fired_halt_rules:
                    success_count += 1
                    cycles_list.append(cycles)
                    fired_names = ", ".join(
                        f"{name} (count={count})"
                        for name, count in fired_halt_rules
                    )
                    print(
                        f"   ✅ Execução {i + 1}/{n_executions}: "
                        f"goal-halt disparou -> {fired_names} | "
                        f"cycles={cycles}"
                    )
                else:
                    stop_reason = (
                        "cycle limit reached"
                        if cycles >= max_cycles
                        else "stopped without firing a halt production"
                    )
                    print(
                        f"   ❌ Execução {i + 1}/{n_executions}: "
                        f"{stop_reason} | cycles={cycles}"
                    )

                if kernel: kernel.Shutdown(); del kernel

            except:

                if kernel:

                    try: kernel.Shutdown(); del kernel

                    except: pass

                continue

        # Salva os dados de ciclos se houver dados válidos

        if cycles_list and last_problem_type:

            save_cycles_data(cycles_list, model_name, last_problem_type)

        min_cycles = min(cycles_list) if cycles_list else None

        max_cycles_result = max(cycles_list) if cycles_list else None

        avg_cycles = sum(cycles_list) / len(cycles_list) if cycles_list else None

        std_dev = statistics.stdev(cycles_list) if len(cycles_list) > 1 else 0.0

        success_rate = (success_count / n_executions) * 100

        # Calcula Shapiro-Wilk

        sw_statistic, sw_pvalue = calculate_shapiro_wilk(cycles_list)

        min_str = str(min_cycles) if min_cycles is not None else 'N/A'

        max_str = str(max_cycles_result) if max_cycles_result is not None else 'N/A'

        avg_str = f"{avg_cycles:.1f}" if avg_cycles is not None else 'N/A'

        std_str = f"{std_dev:.1f}" if std_dev is not None else 'N/A'

        sw_str = f"{sw_statistic:.4f}" if sw_statistic is not None else 'N/A'

        sw_p_str = f"{sw_pvalue:.4f}" if sw_pvalue is not None else 'N/A'

        print(f"✅ {model_name} ({code_len:,} chars): SR={success_rate:.1f}% | Min={min_str} | Max={max_str} | Avg={avg_str} | StdDev={std_str} | SW={sw_str} (p={sw_p_str})")

        return success_rate, min_cycles, max_cycles_result, avg_cycles, std_dev, sw_statistic, sw_pvalue

    finally:

        try:

            if os.path.exists(tmp_path): os.unlink(tmp_path)

        except: pass



def build_context_from_files(soar_files, max_chars):

    print(f"\n📦 DEBUG build_context_from_files:")

    print(f"   Arquivos recebidos: {len(soar_files)}")

    print(f"   Limite de caracteres: {max_chars:,}")

    if not soar_files:

        print("   ❌ Nenhum arquivo fornecido!")

        return "", []

    context_parts, total_chars, included_files = [], 0, []

    for i, filepath in enumerate(soar_files, 1):

        filename = os.path.basename(filepath)

        content = read_soar_file(filepath)

        if not content:

            print(f"   ⚠️  Arquivo {i}/{len(soar_files)}: {filename} - VAZIO ou ERRO na leitura")

            continue

        file_block = f"\n{'='*60}\nFILE: {filename}\n{'='*60}\n{content}\n"

        block_size = len(file_block)

        if total_chars + block_size > max_chars:

            print(f"   ⛔ Arquivo {i}/{len(soar_files)}: {filename} - LIMITE ATINGIDO")

            print(f"      Caracteres atuais: {total_chars:,}")

            print(f"      Tentativa de adicionar: {block_size:,}")

            print(f"      Total seria: {total_chars + block_size:,} > {max_chars:,}")

            break

        context_parts.append(file_block)

        total_chars += block_size

        included_files.append(filename)

        print(f"   ✅ Arquivo {i}/{len(soar_files)}: {filename}")

        print(f"      Tamanho: {block_size:,} chars | Total acumulado: {total_chars:,} chars")

    final_context = "\n".join(context_parts)

    print(f"\n   📊 RESUMO FINAL:")

    print(f"      Arquivos incluídos: {len(included_files)}/{len(soar_files)}")

    print(f"      Total de caracteres: {total_chars:,}/{max_chars:,}")

    print(f"      Arquivos:")

    for f in included_files:

        print(f"         • {f}")

    return final_context, included_files



def generate_soar_prompt(user_question, context, included_files, problem_type):

    files_list = "\n".join([f"- {f}" for f in included_files])

    problem_description = {'wjp': 'Water Jug Problem', 'bw': 'Blocks World Problem', 'hanoi': 'Tower of Hanoi Problem', 'mem': 'Memory Problem', 'rl': 'Reinforcement Learning', 'udf': 'Soar cognitive architecture'}.get(problem_type, 'Soar cognitive architecture')

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

    headers = {"Authorization": f"Bearer {AVAILABLE_MODELS[model_name]['key']}", "Content-Type": "application/json"}

    max_tokens = AVAILABLE_MODELS[model_name]["max_tokens"]

    payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.15, "max_tokens": max_tokens}

    start_time = time.time()

    try:

        print(f"📤 Enviando para: {model_name}")

        r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)

        elapsed = time.time() - start_time

        if r.status_code == 200:

            response_json = r.json()

            response = response_json["choices"][0]["message"]["content"]

            if not response or len(response.strip()) == 0:

                response = response_json["choices"][0]["message"].get("reasoning", "")

            response = response.replace("```soar", "").replace("```", "").strip()

            code_len = len(response) if response else 0

            print(f"✅ {model_name} concluído em {elapsed:.1f}s ({code_len:,} chars)")

            sr, min_cycles, max_cycles_result, avg_cycles, std_dev, sw_stat, sw_pval = test_soar_code(response, model_name)

            return model_name, response, None, elapsed, sr, min_cycles, max_cycles_result, avg_cycles, std_dev, sw_stat, sw_pval

        else:

            print(f"❌ {model_name} erro {r.status_code}")

            return model_name, None, f"Erro API: {r.status_code}", elapsed, None, None, None, None, None, None, None

    except Exception as e:

        elapsed = time.time() - start_time

        print(f"❌ {model_name} exceção: {str(e)}")

        return model_name, None, str(e), elapsed, None, None, None, None, None, None, None



def call_all_models_sequential(prompt):

    print(f"\n🚀 PROCESSANDO {len(AVAILABLE_MODELS)} MODELOS SEQUENCIALMENTE")

    results = {}

    for i, model_name in enumerate(AVAILABLE_MODELS.keys(), 1):

        print(f"\n{'='*60}\n📍 MODELO {i}/{len(AVAILABLE_MODELS)}: {model_name}\n{'='*60}")

        model_name_ret, code, error, elapsed, sr, min_cycles, max_cycles_result, avg_cycles, std_dev, sw_stat, sw_pval = call_openrouter_single(prompt, model_name)

        results[model_name_ret] = {

            'code': code, 

            'error': error, 

            'time': elapsed, 

            'success_rate': sr, 

            'min_cycles': min_cycles, 

            'max_cycles': max_cycles_result, 

            'avg_cycles': avg_cycles, 

            'std_dev': std_dev,

            'sw_statistic': sw_stat,

            'sw_pvalue': sw_pval

        }

        code_len = len(code) if code else 0

        print(f"✅ Resultado: code_len={code_len:,}, sr={sr}, std_dev={std_dev}, sw={sw_stat}, error={error}")

        if i < len(AVAILABLE_MODELS):

            print(f"⏳ Aguardando {TEMPO_REQ}s antes do próximo modelo...")

            time.sleep(TEMPO_REQ)

    print(f"\n{'='*60}\n✅ TODOS OS {len(AVAILABLE_MODELS)} MODELOS PROCESSADOS!\n{'='*60}\n")

    return results



@app.route("/upload", methods=["POST"])

def upload_file():

    if 'file' not in request.files: return "No file provided", 400

    file = request.files['file']

    if file.filename == '': return "No file selected", 400

    if not file.filename.endswith('.soar'): return "Only .soar files are allowed", 400

    try:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        original_name = secure_filename(file.filename).replace('.soar', '')

        new_filename = f"udf_{original_name}_{timestamp}.soar"

        filepath = os.path.join(SOURCE_FOLDER, new_filename)

        file.save(filepath)

        print(f"✅ Arquivo enviado: {new_filename}")

        return f"File uploaded successfully: {new_filename}", 200

    except Exception as e:

        print(f"❌ Erro ao fazer upload: {e}")

        return f"Error uploading file: {str(e)}", 500



@app.route("/", methods=["GET", "POST"])

def index():

    global last_generated_codes, last_problem_type

    user_question = error_message = success_message = detected_type = ""

    all_files = glob.glob(os.path.join(SOURCE_FOLDER, '*.soar'))

    wjp_count = len([f for f in all_files if os.path.basename(f).lower().startswith('wjp')])

    bw_count = len([f for f in all_files if os.path.basename(f).lower().startswith('bw')])

    hanoi_count = len([f for f in all_files if os.path.basename(f).lower().startswith('hanoi')])

    memory_count = len([f for f in all_files if os.path.basename(f).lower().startswith('mem')])

    rl_count = len([f for f in all_files if os.path.basename(f).lower().startswith('rl')])

    other_count = len(all_files) - wjp_count - bw_count - hanoi_count - memory_count - rl_count

    total_count = len(all_files)

    if request.method == "POST":

        action = request.form.get("action", "generate")

        if action == "confirm_success":

            model_to_save = request.form.get("model_to_save")

            if model_to_save and model_to_save in last_generated_codes:

                result = last_generated_codes[model_to_save]

                code = result['code']

                success_rate = result.get('success_rate')

                if code:

                    if SOAR_AVAILABLE and success_rate is not None and success_rate == 0:

                        error_message = "❌ Cannot add code with 0% Success Rate to database"

                    else:

                        saved_filename = save_successful_code(code, last_problem_type, model_to_save)

                        if saved_filename:

                            model_info = AVAILABLE_MODELS.get(model_to_save, {})

                            model_name = model_info.get("name", model_to_save) if isinstance(model_info, dict) else model_info

                            success_message = f"✅ Code from {model_name} added to database: {saved_filename}"

                            all_files = glob.glob(os.path.join(SOURCE_FOLDER, '*.soar'))

                            wjp_count = len([f for f in all_files if os.path.basename(f).lower().startswith('wjp')])

                            bw_count = len([f for f in all_files if os.path.basename(f).lower().startswith('bw')])

                            hanoi_count = len([f for f in all_files if os.path.basename(f).lower().startswith('hanoi')])

                            memory_count = len([f for f in all_files if os.path.basename(f).lower().startswith('mem')])

                            rl_count = len([f for f in all_files if os.path.basename(f).lower().startswith('rl')])

                            other_count = len(all_files) - wjp_count - bw_count - hanoi_count - memory_count - rl_count

                            total_count = len(all_files)

                        else: error_message = "Failed to save code"

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

                            f"No .soar files found for problem type '{detected_type}'"

                        )

                    else:

                        context, included_files = build_context_from_files(

                            soar_files,

                            MAX_CONTEXT_CHARS

                        )



                        if not context:

                            error_message = "Could not read any .soar files"

                        else:

                            prompt = generate_soar_prompt(

                                user_question,

                                context,

                                included_files,

                                detected_type

                            )



                            print(

                                f"\n📁 FILES SENT TO LLM ({len(included_files)}):"

                            )

                            for f in included_files:

                                print(f"   • {f}")



                            last_generated_codes = call_all_models_sequential(prompt)



                else:

                    # Controlled No-RAG condition:

                    # same structured prompt, but no retrieved files or context

                    context = ""

                    included_files = []



                    prompt = generate_soar_prompt(

                        user_question,

                        context,

                        included_files,

                        detected_type

                    )



                    print("\n🚫 CONTROLLED NO-RAG CONDITION")

                    print("   Files sent to LLM: 0")

                    print("   RAG context: empty")



                    last_generated_codes = call_all_models_sequential(prompt)



    problem_names = {'wjp': 'Water Jug', 'bw': 'Blocks World', 'hanoi': 'Tower of Hanoi', 'mem': 'Memory', 'rl': 'RL', 'udf': 'General'}

    results_html = ""

    if last_generated_codes:

        results_html = '<hr class="my-4"><h4>📊 Results from ALL Models:</h4>'

        print(f"\n🔍 DEBUG: Processando {len(last_generated_codes)} modelos no HTML")

        for model_id, result in last_generated_codes.items():

            code = result.get('code')

            code_len = len(code) if code else 0

            print(f"   • Modelo: {model_id} | Código: {bool(code)} | Len: {code_len:,} | SR: {result.get('success_rate')}")

            model_info = AVAILABLE_MODELS.get(model_id, {})

            model_name = model_info.get("name", model_id) if isinstance(model_info, dict) else model_id

            max_tokens = model_info.get("max_tokens", 8192) if isinstance(model_info, dict) else 8192

            model_short = model_id.split('/')[1].split(':')[0].replace('-', '_')

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"{last_problem_type}_{model_short}_{timestamp}.soar"

            if result.get('code') or (result.get('success_rate') is not None and result.get('success_rate') > 0):

                sr = result.get('success_rate')

                min_c = result.get('min_cycles')

                max_c = result.get('max_cycles')

                avg_c = result.get('avg_cycles')

                std_c = result.get('std_dev')

                sw_stat = result.get('sw_statistic')

                sw_pval = result.get('sw_pvalue')

                has_valid_code = code and len(str(code).strip()) > 0

                can_add_to_db, disable_reason = True, ""

                if not has_valid_code: can_add_to_db, disable_reason = False, "No code generated"

                elif SOAR_AVAILABLE and sr is not None and sr == 0: can_add_to_db, disable_reason = False, "Cannot add code with 0% Success Rate"

                show_download_copy = False

                if SOAR_AVAILABLE and sr is not None and sr == 100.0:

                    show_download_copy = True

                elif not SOAR_AVAILABLE and has_valid_code:

                    show_download_copy = True

                stats_html = ""

                if sr is not None:

                    stats_badge = "success" if sr >= 80 else "warning" if sr >= 50 else "danger"

                    min_str = str(min_c) if min_c is not None else 'N/A'

                    max_str = str(max_c) if max_c is not None else 'N/A'

                    avg_str = f"{avg_c:.1f}" if avg_c is not None else 'N/A'

                    std_str = f"{std_c:.1f}" if std_c is not None else 'N/A'

                    sw_str = f"{sw_stat:.4f}" if sw_stat is not None else 'N/A'

                    sw_p_str = f"{sw_pval:.4f}" if sw_pval is not None else 'N/A'

                    # Análise interpretativa do Shapiro-Wilk

                    sw_interpretation = ""

                    if sw_pval is not None:

                        if sw_pval > 0.05:

                            sw_interpretation = "✅ Normal distribution (p > 0.05)"

                            sw_badge = "success"

                        else:

                            sw_interpretation = "⚠️ Non-normal distribution (p ≤ 0.05)"

                            sw_badge = "warning"

                    else:

                        sw_interpretation = "N/A"

                        sw_badge = "secondary"

                    # Cálculo do AT(S) com a fórmula completa incluindo desvio padrão e Shapiro-Wilk

                    if min_str != "N/A" and max_str != "N/A" and avg_str != "N/A" and std_str != "N/A":

                        # W({Ci(S)}^N_{i=1}) = termo de peso baseado no Shapiro-Wilk

                        # Se os dados são normais (SW alto), o peso é favorável

                        # Usamos o statistic do SW diretamente como peso (varia de 0 a 1)

                        w_weight = sw_stat if sw_stat is not None else 0.5  # Default 0.5 se não disponível

                        at_value = (sr/100) * (

                            ALPHA * int(min_str) + 

                            BETA * (1/N_EXECUTIONS) * (float(avg_str) * N_EXECUTIONS) + 

                            GAMMA * int(max_str) + 

                            KAPPA * float(std_str)                             

                        )

                        at_html = f'{at_value:.2f}'

                    else:

                        at_html = 'N/A'

                    stats_html = f'''<div class="mt-2 p-2 bg-light rounded"><small>

                        <strong>📈 Test Results ({N_EXECUTIONS} runs, {code_len:,} chars):</strong><br>

                        <span class="badge bg-{stats_badge}">SR: {sr:.1f}%</span> 

                        <span class="badge bg-info">Min: {min_str} cycles</span> 

                        <span class="badge bg-warning">Max: {max_str} cycles</span> 

                        <span class="badge bg-secondary">Avg: {avg_str} cycles</span> 

                        <span class="badge bg-primary">StdDev: {std_str}</span><br>                      

                        <strong>🎯 Final Score:</strong> 

                        <span class="badge bg-{stats_badge}">AT(S): {at_html}</span>

                    </small></div>'''



                        #<!--  <strong>📊 Shapiro-Wilk Test:</strong><br>

                        #<span class="badge bg-{sw_badge}">SW: {sw_str} (p={sw_p_str})</span>

                        #<span class="badge bg-{sw_badge}">{sw_interpretation}</span><br> -->



                elif not SOAR_AVAILABLE: 

                    stats_html = f'<div class="mt-2 alert alert-warning p-2"><small>⚠️ Soar not available - tests skipped ({code_len:,} chars)</small></div>'

                add_button = f'<form method="POST" style="display:inline;"><input type="hidden" name="action" value="confirm_success"><input type="hidden" name="model_to_save" value="{html.escape(model_id)}"><button type="submit" class="btn btn-success w-100">✅ Add to Database</button></form>' if can_add_to_db else f'<button type="button" class="btn btn-danger w-100" disabled title="{html.escape(disable_reason)}">🚫 {html.escape(disable_reason)}</button>'

                download_copy_buttons = ""

                if show_download_copy:

                    download_copy_buttons = f'<button onclick="downloadCode_{model_short}()" class="btn btn-primary">💾 Download {filename}</button><button onclick="copyCode_{model_short}()" class="btn btn-secondary">📋 Copy to Clipboard</button>'

                else:

                    download_copy_buttons = '<div class="alert alert-info p-2 mb-2"><small>💡 Download and Copy buttons available only with 100% Success Rate</small></div>'

                # Usar json.dumps para escape correto de JavaScript

                code_json = json.dumps(code if code else '')

                filename_json = json.dumps(filename)

                results_html += f'''<div class="card mb-3 border-success">

                 <div class="card-header bg-success text-white">

                        <strong>✅ {html.escape(model_name)}</strong> - Generated in {result["time"]:.1f}s | Max: {max_tokens:,} tokens | {code_len:,} chars

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

                        if (!code || code.trim() === '') {{

                            alert("❌ No code");

                            return;

                        }}

                        navigator.clipboard.writeText(code).then(() => {{

                            alert("✅ Copied!");

                        }}).catch(err => {{

                            alert("❌ Error: " + err);

                        }});

                    }};

                    window["downloadCode_{model_short}"] = function() {{

                        if (!code || code.trim() === '') {{

                            alert("❌ No code");

                            return;

                        }}

                        try {{

                            const blob = new Blob([code], {{ type: "text/plain;charset=utf-8" }});

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

                        }} catch(err) {{

                            alert("❌ Error: " + err);

                        }}

                    }};

                }})();

                </script>'''

            else:

                error_msg = str(result.get('error', 'Unknown error'))

                error_escaped = html.escape(str(result.get('error', 'Unknown error')))

                results_html += f'<div class="card mb-3 border-danger"><div class="card-header bg-danger text-white"><strong>❌ {html.escape(model_name)}</strong> - Failed in {result["time"]:.1f}s</div><div class="card-body"><p class="text-danger mb-0">Error: {error_escaped}</p></div></div>'

    progress_html = f'''

    <div id="progressContainer" style="display:none;" class="mb-4">

        <div class="alert alert-info">

            <strong>⏳ Processing...</strong>

            <div class="progress mt-2" style="height: 25px;">

                <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>

            </div>

            <small id="progressText" class="mt-2 d-block">Initializing...</small>

        </div>

    </div>

    '''

    scipy_warning = ""

    if not SCIPY_AVAILABLE:

        scipy_warning = '<div class="alert alert-warning">⚠️ <strong>scipy not installed</strong> - Shapiro-Wilk test disabled. Install with: <code>pip install scipy</code></div>'

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>NLP for Soar - Sequential Processing with Shapiro-Wilk</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"><style>body{{background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);min-height:100vh;padding:20px}}.main-card{{background:white;border-radius:15px;max-width:1400px;margin:0 auto}}a,button,.btn{{text-decoration:none!important}}pre{{margin:0}}</style></head><body><div class="container mt-4"><div class="main-card"><div class="card-header bg-primary text-white p-4"><h1 class="h3">🤖 NLP for Soar - Sequential Multi-LLM Processing</h1><p class="mb-0">⚡ Processes {len(AVAILABLE_MODELS)} models sequentially | 🧪 Auto-tests with {N_EXECUTIONS} runs | 📊 Saves cycle data to {CYCLES_DATA_FOLDER}/</p></div><div class="card-body p-4">{scipy_warning}{f'<div class="alert alert-info"><strong>📊 Database:</strong> WJP: {wjp_count} | BW: {bw_count} | Hanoi: {hanoi_count} | Memory: {memory_count} | RL: {rl_count} | Other: {other_count} | Total: {total_count}</div>' if total_count>0 else '<div class="alert alert-warning">No files found in RAG folder</div>'}{f'<div class="alert alert-success">{success_message}</div>' if success_message else ''}{f'<div class="alert alert-danger">{error_message}</div>' if error_message else ''}{f'<div class="alert alert-info">🎯 Detected: <strong>{problem_names.get(detected_type,"Unknown")}</strong> ({detected_type})</div>' if detected_type else ''}{progress_html}<div class="mb-4"><label class="form-label fw-bold">📤 Upload .soar file to database (saved as udf_*):</label><input type="file" id="fileInput" class="form-control" accept=".soar"><button onclick="uploadFile()" class="btn btn-secondary mt-2" type="button">⬆️ Upload File</button><div id="uploadStatus" class="mt-2"></div></div><hr class="my-4"><form method="POST" id="generateForm"><input type="hidden" name="action" value="generate"><div class="mb-4"><label class="form-label fw-bold">❓ Your Question:</label><textarea class="form-control" name="user_question" rows="4" {'disabled' if total_count==0 else 'required'}>{user_question}</textarea><small class="text-muted">💡 Domains: Water Jug, Blocks World, Tower of Hanoi, Memory, RL and General.</small></div><button type="submit" class="btn btn-primary btn-lg w-100" id="generateBtn" {'disabled' if total_count==0 else ''}>🚀 Generate & Test with {len(AVAILABLE_MODELS)} Models (Sequential)</button></form>{results_html}</div></div></div><script>const totalModels={len(AVAILABLE_MODELS)};let currentModel=0;function uploadFile(){{const fileInput=document.getElementById('fileInput');const uploadStatus=document.getElementById('uploadStatus');if(!fileInput.files||fileInput.files.length===0){{uploadStatus.innerHTML='<div class="alert alert-warning">Please select a file</div>';return}}const file=fileInput.files[0];if(!file.name.endsWith('.soar')){{uploadStatus.innerHTML='<div class="alert alert-danger">Only .soar files are allowed</div>';return}}const formData=new FormData();formData.append('file',file);uploadStatus.innerHTML='<div class="alert alert-info">Uploading...</div>';fetch('/upload',{{method:'POST',body:formData}}).then(response=>response.text()).then(data=>{{uploadStatus.innerHTML='<div class="alert alert-success">'+data+'</div>';setTimeout(()=>location.reload(),1500)}}).catch(error=>{{uploadStatus.innerHTML='<div class="alert alert-danger">Error: '+error+'</div>'}});}}document.getElementById('generateForm').addEventListener('submit',function(e){{const progressContainer=document.getElementById('progressContainer');const progressBar=document.getElementById('progressBar');const progressText=document.getElementById('progressText');const generateBtn=document.getElementById('generateBtn');progressContainer.style.display='block';generateBtn.disabled=true;generateBtn.innerHTML='⏳ Processing...';currentModel=0;const interval=setInterval(function(){{currentModel++;const percent=Math.min((currentModel/totalModels)*100,95);progressBar.style.width=percent+'%';progressBar.setAttribute('aria-valuenow',percent);progressBar.innerHTML=Math.round(percent)+'%';progressText.innerHTML='Processing model '+currentModel+' of '+totalModels+'...';if(currentModel>=totalModels){{clearInterval(interval);progressText.innerHTML='Finalizing results...';progressBar.style.width='100%';progressBar.setAttribute('aria-valuenow',100);progressBar.innerHTML='100%';}}}},{TEMPO_REQ*1000});}});</script></body></html>"""



if __name__ == "__main__":

    print("🚀 NLP for Soar - Sequential Multi-LLM Processing + Auto-Testing + Shapiro-Wilk")

    print(f"📁 Source folder: {SOURCE_FOLDER}/")

    print(f"📊 Cycles data folder: {CYCLES_DATA_FOLDER}/")

    print("🎯 Auto-detects: WJP | BW | Hanoi | Memory | RL | UDF")

    print(f"🤖 {len(AVAILABLE_MODELS)} models process sequentially (one at a time)")

    print(f"🧪 Auto-tests each code {N_EXECUTIONS} times (max {MAX_CYCLES} cycles)")

    print("✅ Soar available - testing enabled" if SOAR_AVAILABLE else "⚠️ Soar NOT available - testing disabled")

    print("✅ scipy available - Shapiro-Wilk enabled" if SCIPY_AVAILABLE else "⚠️ scipy NOT available - Shapiro-Wilk disabled")

    print("🌐 http://127.0.0.1:5000")

    print("⚠️ Debug mode DISABLED")

    app.run(host='127.0.0.1', port=5000, debug=False)
