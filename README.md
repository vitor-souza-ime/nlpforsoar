# NLP for Soar

Public implementation accompanying the manuscript:

**NLP for Soar: A Hybrid Symbolic-Connectionist Architecture for Automatic Code Generation and Evaluation via the AT(S) Metric**

## Scope

This repository contains the Python implementation used to support natural-language-driven generation, execution-based validation, and contextual reuse of Soar production-rule artifacts.

The experimental study reported in the manuscript evaluates three canonical domains:

- Water Jug Problem
- Tower of Hanoi
- Blocks World

The reported results are a controlled proof of concept. They do not establish general performance in unseen Soar domains or formal verification over the complete reachable state space.

The current implementation also contains preliminary keyword-detection paths for memory-related and reinforcement-learning inputs. These paths were not part of the experimental evaluation reported in the manuscript, and no performance claims are made for them.

## Evaluation protocol

For each generated Soar artifact, the implementation:

1. detects active productions containing `(halt)`;
2. executes the fixed artifact in 100 separate Soar kernel runs;
3. queries the production firing count through SML;
4. counts a run as successful only when the goal-test halt production fires;
5. sets binary functional admissibility `SR = 1` only when all 100 executions are successful;
6. computes AT(S) only for artifacts with `SR = 1`.

AT(S) is therefore a success-conditioned execution-efficiency/consistency metric. It does not multiply execution cost by a fractional success rate.

## Retrieval behavior

The current RAG implementation uses deterministic domain-filtered contextual retrieval:

- problem-domain detection by keyword counting;
- file filtering by domain prefix;
- lexicographic filename ordering;
- concatenation up to a 200,000-character context limit.

The current retrieval module does **not** perform embedding similarity, semantic ranking, chunk-level retrieval, or AT(S)-based retrieval ranking.

## Models and principal settings

The implementation uses OpenRouter with the following model identifiers:

- `z-ai/glm-4.7`
- `minimax/minimax-m2.1`
- `deepseek/deepseek-v3.2`

Principal settings:

- temperature: `0.15`
- maximum output: `20,000` tokens
- maximum RAG context: `200,000` characters
- Soar execution limit: `10,000` decision cycles
- validation executions per generated artifact: `100`
- interval between model requests: `15` seconds

## Requirements

The manuscript reports Python 3.12.3 and Soar 9.6.5. Python dependencies are pinned in `requirements.txt`:

```text
Flask==3.1.0
requests==2.32.3
scipy==1.15.1
Werkzeug==3.1.3
soar-sml==9.6.5
```

Install with:

```bash
pip install -r requirements.txt
```

## OpenRouter API key

Set the API key through the environment variable:

```bash
export OPENROUTER_API_KEY="YOUR_KEY"
```

On Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="YOUR_KEY"
```

## Running the application

From the repository root:

```bash
python main.py
```

The Flask application generates candidate Soar code, performs execution-based validation when Soar/SML is available, reports the binary SR result and AT(S) for admissible artifacts, and permits incorporation into the RAG knowledge base only for artifacts satisfying `SR = 1`.

## Repository structure

```text
.
├── main.py
├── requirements.txt
├── README.md
└── RAG/
```

The `RAG/` directory contains canonical seeds and generated Soar artifacts used as contextual examples.

## Reproducibility notes

The manuscript cites code snapshot:

```text
cb4dfb3f05922d9c76c7b4aa4a3c119ef75f0292
```

This snapshot corresponds to the revised implementation used for validation and revision analyses. The repository was populated after the original generation experiments. Historical provider allocation, exact per-call retrieved-file sets, and some other serving metadata were not persisted for every original API request; these limitations are explicitly reported in the manuscript.

The 100 Soar executions characterize execution-level behavior of one fixed generated artifact. They are not 100 independent LLM generations and should not be interpreted as an estimate of LLM generation probability.

## License and research use

This repository is provided for academic and research use in support of reproducibility and further investigation of LLM-assisted knowledge engineering for the Soar cognitive architecture.

## Author

**Vitor Amadeu Souza**
vitor.souza@ime.eb.br
