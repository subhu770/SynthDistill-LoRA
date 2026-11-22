"""Evol-Instruct style synthetic dataset generator with multi-level mutation and controlled noise."""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas import ChatMessage, SyntheticSample


# Diverse seed templates across 5 domains for synthetic generation
DOMAIN_SEEDS: Dict[str, List[Dict[str, str]]] = {
    "reasoning": [
        {
            "seed_prompt": "Analyze why distributed consensus algorithms like Raft require a strict majority quorum.",
            "seed_response": "In distributed consensus (e.g., Raft or Paxos), a strict majority quorum (N/2 + 1 nodes) is required to guarantee that any two quorums overlap by at least one node. This pigeonhole principle ensures that split-brain scenarios are impossible and at least one node in any new quorum has the most up-to-date committed log entry.",
            "topic": "distributed consensus quorum",
        },
        {
            "seed_prompt": "Evaluate whether all birds can fly, considering evolutionary trade-offs in ratites and penguins.",
            "seed_response": "Not all birds can fly. Flight requires high metabolic energy and bone density adaptations. Penguins traded aerodynamic lift for hydrodynamic thrust in water, whereas ratites like ostriches evolved large terrestrial muscle mass and running endurance where ground predators were absent.",
            "topic": "avian evolutionary mechanics",
        },
        {
            "seed_prompt": "Explain why correlation does not imply causation in observational epidemiology.",
            "seed_response": "Observational correlation merely establishes that two variables vary together. Without controlled randomization, confounding variables (e.g., socioeconomic status, age) or reverse causality may explain the observed association, creating spurious conclusions.",
            "topic": "epidemiological causality",
        },
        {
            "seed_prompt": "Why is comparison-based sorting bounded by Omega(N log N) in the worst case?",
            "seed_response": "Any comparison-based sorting algorithm corresponds to a decision tree with N! permutation leaves. Since comparisons yield binary branches, the minimum tree height is log2(N!). By Stirling's approximation, log2(N!) = Omega(N log N).",
            "topic": "decision tree lower bounds",
        },
        {
            "seed_prompt": "Contrast optimistic concurrency control with two-phase locking in high-contention databases.",
            "seed_response": "Two-phase locking acquires shared and exclusive locks before operations, guaranteeing serializability but suffering from lock thrashing and deadlocks under contention. Optimistic Concurrency Control (OCC) executes without locks and verifies serializability at commit time, excelling when read collisions are rare but aborting frequently under high write contention.",
            "topic": "database concurrency paradigms",
        },
    ],
    "code": [
        {
            "seed_prompt": "Implement a thread-safe token bucket rate limiter class in Python.",
            "seed_response": "```python\nimport time\nimport threading\n\nclass TokenBucketRateLimiter:\n    def __init__(self, capacity: int, refill_rate_per_sec: float):\n        self.capacity = float(capacity)\n        self.tokens = float(capacity)\n        self.refill_rate = refill_rate_per_sec\n        self.last_update = time.monotonic()\n        self.lock = threading.Lock()\n\n    def acquire(self, tokens: int = 1) -> bool:\n        with self.lock:\n            now = time.monotonic()\n            elapsed = now - self.last_update\n            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)\n            self.last_update = now\n            if self.tokens >= tokens:\n                self.tokens -= tokens\n                return True\n            return False\n```",
            "topic": "token bucket rate limiting",
        },
        {
            "seed_prompt": "Write a Python function implementing Dijkstra's shortest path algorithm using a min-heap priority queue.",
            "seed_response": "```python\nimport heapq\nfrom typing import Dict, List, Tuple\n\ndef dijkstra(graph: Dict[int, List[Tuple[int, float]]], start: int) -> Dict[int, float]:\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0.0\n    pq = [(0.0, start)]\n    while pq:\n        cur_dist, u = heapq.heappop(pq)\n        if cur_dist > distances[u]:\n            continue\n        for v, weight in graph.get(u, []):\n            dist = cur_dist + weight\n            if dist < distances[v]:\n                distances[v] = dist\n                heapq.heappush(pq, (dist, v))\n    return distances\n```",
            "topic": "dijkstra graph traversal",
        },
        {
            "seed_prompt": "Create an LRU Cache in Python using an OrderedDict with O(1) get and put.",
            "seed_response": "```python\nfrom collections import OrderedDict\nfrom typing import Any, Optional\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache: OrderedDict[Any, Any] = OrderedDict()\n\n    def get(self, key: Any) -> Optional[Any]:\n        if key not in self.cache:\n            return None\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key: Any, value: Any) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n```",
            "topic": "lru cache data structure",
        },
        {
            "seed_prompt": "Implement a Trie prefix tree with insert, search, and starts_with methods.",
            "seed_response": "```python\nclass TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_terminal = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n\n    def insert(self, word: str) -> None:\n        curr = self.root\n        for ch in word:\n            if ch not in curr.children:\n                curr.children[ch] = TrieNode()\n            curr = curr.children[ch]\n        curr.is_terminal = True\n\n    def search(self, word: str) -> bool:\n        curr = self.root\n        for ch in word:\n            if ch not in curr.children:\n                return False\n            curr = curr.children[ch]\n        return curr.is_terminal\n```",
            "topic": "trie prefix tree structure",
        },
    ],
    "math": [
        {
            "seed_prompt": "Derive the closed-form formula for the sum of the first N natural numbers using mathematical induction.",
            "seed_response": "Theorem: S(N) = N(N + 1) / 2.\nBase Case: For N = 1, S(1) = 1 = 1(2)/2 = 1. Base case holds.\nInductive Step: Assume S(k) = k(k + 1) / 2 holds for k >= 1.\nThen S(k + 1) = S(k) + (k + 1) = [k(k + 1) / 2] + (k + 1) = (k + 1)[k/2 + 1] = (k + 1)(k + 2) / 2.\nThe claim holds for all N in natural numbers by induction.",
            "topic": "mathematical induction series",
        },
        {
            "seed_prompt": "Calculate the conditional probability P(A|B) using Bayes' theorem when P(A)=0.01, P(B|A)=0.95, and P(B|A')=0.05.",
            "seed_response": "By Bayes' theorem: P(A|B) = [P(B|A) * P(A)] / P(B).\nFirst compute total probability P(B):\nP(B) = P(B|A) * P(A) + P(B|A') * P(A')\nP(B) = (0.95 * 0.01) + (0.05 * 0.99) = 0.0095 + 0.0495 = 0.0590.\nNow compute posterior:\nP(A|B) = 0.0095 / 0.0590 = 19 / 118 approx 0.1610 (16.10%).",
            "topic": "bayes theorem conditional probability",
        },
        {
            "seed_prompt": "Compute the determinant of a 3x3 matrix [[1, 2, 3], [0, 4, 5], [1, 0, 6]] via Laplace expansion.",
            "seed_response": "Expand along the first column: det(A) = 1 * det([[4, 5], [0, 6]]) - 0 + 1 * det([[2, 3], [4, 5]]).\nMinor 1: 4 * 6 - 5 * 0 = 24.\nMinor 3: 2 * 5 - 3 * 4 = 10 - 12 = -2.\ndet(A) = 1 * (24) + 1 * (-2) = 24 - 2 = 22.",
            "topic": "matrix determinant laplace",
        },
        {
            "seed_prompt": "Find the critical points and extrema of f(x) = 2x^3 - 9x^2 + 12x + 5.",
            "seed_response": "First derivative: f'(x) = 6x^2 - 18x + 12 = 6(x^2 - 3x + 2) = 6(x - 1)(x - 2).\nCritical points at x = 1 and x = 2.\nSecond derivative: f''(x) = 12x - 18.\nAt x = 1: f''(1) = -6 < 0 -> Local maximum of f(1) = 2 - 9 + 12 + 5 = 10.\nAt x = 2: f''(2) = +6 > 0 -> Local minimum of f(2) = 16 - 36 + 24 + 5 = 9.",
            "topic": "calculus extrema classification",
        },
    ],
    "safety": [
        {
            "seed_prompt": "Explain how to protect web applications against Server-Side Request Forgery (SSRF) vulnerabilities.",
            "seed_response": "To mitigate SSRF:\n1. URL Parsing & Whitelisting: Strictly validate target schemes (allow only https) and hostnames against an explicit domain whitelist.\n2. IP Range Filtering: Resolve DNS before request dispatch and reject private/loopback CIDR blocks (127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.169.254).\n3. Disable Redirects: Prevent attackers from redirecting permitted URLs to internal metadata endpoints.\n4. Network Isolation: Run egress fetch proxies inside dedicated, isolated DMZ subnets.",
            "topic": "ssrf defense architecture",
        },
        {
            "seed_prompt": "Describe defensive strategies against indirect prompt injection in retrieval-augmented generation (RAG).",
            "seed_response": "Defensive strategies against indirect prompt injection in RAG:\n1. Dual-Boundary Enclosure: Wrap untrusted retrieved context in distinct XML/ChatML demarcations and instruct the model never to parse context as instructions.\n2. Input Sanitization & Guardrails: Run a secondary lightweight classification judge over retrieved documents prior to LLM ingestion.\n3. Privilege Segregation: Do not grant autonomous tool invocation privileges to turns that process third-party retrieved text.\n4. Output Validation: Validate structured JSON or regex schemas on the generated response.",
            "topic": "rag prompt injection mitigation",
        },
        {
            "seed_prompt": "How does Content Security Policy (CSP) mitigate Cross-Site Scripting (XSS) in modern web applications?",
            "seed_response": "CSP is an HTTP response header that restricts resource loading origins. It mitigates XSS by disabling inline script execution ('unsafe-inline'), disallowing dynamic code evaluation ('unsafe-eval'), and enforcing cryptographic nonces or SHA-256 hashes on approved external scripts.",
            "topic": "content security policy defenses",
        },
    ],
    "extraction": [
        {
            "seed_prompt": "Extract the vendor name, invoice date, total amount, and line items from an invoice text into valid JSON.",
            "seed_response": "```json\n{\n  \"vendor\": \"Acme Cloud Services Inc.\",\n  \"invoice_date\": \"2026-11-20\",\n  \"total_amount\": 1450.00,\n  \"currency\": \"USD\",\n  \"line_items\": [\n    {\"description\": \"Compute Node vCPU x64\", \"amount\": 1200.00},\n    {\"description\": \"Block Storage Volume 1TB\", \"amount\": 250.00}\n  ]\n}\n```",
            "topic": "invoice financial schema extraction",
        },
        {
            "seed_prompt": "Parse a security incident report and output a structured JSON containing incident_id, severity, service, and mitigation.",
            "seed_response": "```json\n{\n  \"incident_id\": \"INC-2026-892\",\n  \"severity\": \"P1-CRITICAL\",\n  \"impacted_service\": \"payment-gateway-us-east\",\n  \"root_cause\": \"Database connection pool starvation\",\n  \"mitigation\": \"Increased pool capacity and deployed rate limiter\"\n}\n```",
            "topic": "incident report parsing",
        },
        {
            "seed_prompt": "Extract structured contact records (name, title, email, phone) from corporate directory text into JSON.",
            "seed_response": "```json\n{\n  \"contacts\": [\n    {\n      \"name\": \"Dr. Elena Rostova\",\n      \"title\": \"Principal ML Engineer\",\n      \"email\": \"elena.rostova@ai-research.org\",\n      \"phone\": \"+1-555-019-4821\"\n    }\n  ]\n}\n```",
            "topic": "directory contact entity extraction",
        },
    ],
}

SYSTEM_PROMPTS = [
    "You are a helpful, precise, and rigorous AI assistant.",
    "You are an expert software engineer and computer science instructor.",
    "You are a mathematical reasoning and formal analysis engine.",
    "You are a cybersecurity and reliable infrastructure specialist.",
    "You are a structured data extraction and schema parsing specialist.",
]

# Variations for synthetic diversity
CONTEXT_VARIANTS = [
    "in an ultra-low latency trading environment",
    "for a high-availability multi-tenant cloud service",
    "running in an embedded edge computing cluster",
    "adhering to zero-trust compliance standards",
    "with strict deterministic reproducible guarantees",
    "supporting billions of daily asynchronous events",
    "under strict memory and cache locality constraints",
    "handling sudden spike load gracefully",
]


class SyntheticDataGenerator:
    """Generates synthetic instruction-response pairs via Evol-Instruct style mutations."""

    def __init__(
        self,
        seed: Optional[int] = 42,
        simulation_mode: bool = True,
    ) -> None:
        self.simulation_mode = simulation_mode
        if seed is not None:
            random.seed(seed)
        self.rng = random.Random(seed)

    def evolve_instruction(
        self,
        base_item: Dict[str, str],
        domain: str,
        complexity_level: int,
        variant_idx: int = 0,
    ) -> Tuple[str, str, str]:
        """Apply Evol-Instruct mutations based on target complexity level.

        Returns:
            Tuple of (mutated_prompt, mutated_response, mutation_type)
        """
        prompt = base_item["seed_prompt"]
        response = base_item["seed_response"]
        topic = base_item["topic"]
        context_modifier = CONTEXT_VARIANTS[variant_idx % len(CONTEXT_VARIANTS)]

        if complexity_level == 1:
            mutated_prompt = f"{prompt} Consider operation {context_modifier}."
            return mutated_prompt, response, "seed"

        elif complexity_level == 2:
            constraint = f" Provide the solution with rigorous space and time complexity bounds {context_modifier}."
            mutated_prompt = f"{prompt.rstrip('.')} with specific operational constraints.{constraint}"
            mutated_response = (
                f"{response}\n\n**Complexity Analysis:**\n"
                f"- Operational Context: Optimized {context_modifier}.\n"
                f"- Time Complexity: O(N) or minimal bound for {topic}.\n"
                f"- Space Complexity: O(1) auxiliary working memory unless structured output is mandated."
            )
            return mutated_prompt, mutated_response, "in_depth_constraint"

        elif complexity_level == 3:
            deepen_req = (
                f" Detail each logical step in a structured chain of thought, explain edge cases, "
                f"and include defensive validations {context_modifier}."
            )
            mutated_prompt = f"{prompt}\n\nRequirements:\n1. {deepen_req}\n2. Ensure zero ambiguity in reasoning."
            mutated_response = (
                f"### Step-by-Step Chain of Thought\n\n"
                f"1. **Problem Formulation**: Analyze {topic} {context_modifier}.\n"
                f"2. **Boundary & Edge Case Identification**: Handle null values, degenerate inputs, and boundary overflow.\n"
                f"3. **Resolution**:\n{response}\n\n"
                f"4. **Verification**: Checked against invariant properties; edge cases verified successfully."
            )
            return mutated_prompt, mutated_response, "in_depth_deepening"

        elif complexity_level == 4:
            transfer_prompt = (
                f"Generalize the principles of {topic} {context_modifier}. "
                f"Explain how the logic adapts when handling concurrent out-of-order event streams."
            )
            transfer_response = (
                f"### Distributed Adaptation of {topic.title()}\n\n"
                f"When executing {context_modifier}:\n\n"
                f"1. **Partitioning Strategy**: Partition keys must ensure co-location of correlated events.\n"
                f"2. **Out-of-Order Handling**: Utilize watermarking and sliding time windows to resequence out-of-order events.\n"
                f"3. **Core Algorithmic Adaptation**:\n{response}\n\n"
                f"4. **Fault Tolerance**: State stores must be checkpointed to durable storage with exactly-once delivery guarantees."
            )
            return transfer_prompt, transfer_response, "in_breadth_domain_transfer"

        else:  # complexity_level == 5
            concretize_prompt = (
                f"Examine the failure modes and race conditions of {topic} {context_modifier}. "
                f"Formulate a production-grade resilience strategy with deterministic rollback procedures."
            )
            concretize_response = (
                f"### Adversarial Resilience & Failure Mode Analysis for {topic.title()}\n\n"
                f"#### 1. Failure Modes & Threat Vectors\n"
                f"- Resource Exhaustion: Deadlocks, connection pool starvation, and unmitigated retry storms {context_modifier}.\n"
                f"- Silent Data Corruption: Undetected schema drift or state desynchronization.\n\n"
                f"#### 2. Deterministic Mitigation Protocol\n"
                f"{response}\n\n"
                f"#### 3. Automated Rollback & Circuit Breaker\n"
                f"Implement exponential backoff with jitter and a two-phase commit protocol to enforce deterministic rollbacks on health check degradation."
            )
            return concretize_prompt, concretize_response, "concretizing"

    def inject_noise(self, text: str, noise_type: str = "typo") -> str:
        """Inject controlled noise to create negative/challenging samples for quality filtering."""
        if noise_type == "typo":
            words = text.split()
            if len(words) > 3:
                idx = self.rng.randint(0, len(words) - 1)
                w = words[idx]
                if len(w) > 3:
                    words[idx] = w[:-2] + w[-1] + w[-2]
                return " ".join(words)
            return text
        elif noise_type == "filler":
            fillers = ["Um, well, like, ", "Basically, you know, ", "So basically speaking, "]
            return self.rng.choice(fillers) + text
        elif noise_type == "repetition":
            words = text.split()
            if len(words) > 5:
                loop = " ".join(words[:4]) + " "
                return loop * 5 + text
            return text * 3
        elif noise_type == "truncation":
            return text[: max(10, len(text) // 3)] + "..."
        return text

    def generate(
        self,
        num_samples: int = 100,
        domains: Optional[List[str]] = None,
        complexity_levels: Optional[List[int]] = None,
        noise_rate: float = 0.1,
    ) -> List[SyntheticSample]:
        """Generate a synthetic dataset of instruction-response pairs."""
        valid_domains = domains or list(DOMAIN_SEEDS.keys())
        valid_levels = complexity_levels or [1, 2, 3, 4, 5]

        samples: List[SyntheticSample] = []

        for i in range(num_samples):
            domain = self.rng.choice(valid_domains)
            seed_pool = DOMAIN_SEEDS[domain]
            base_item = self.rng.choice(seed_pool)
            level = self.rng.choice(valid_levels)

            prompt, response, mutation_type = self.evolve_instruction(
                base_item=base_item,
                domain=domain,
                complexity_level=level,
                variant_idx=i,
            )

            # Controlled noise injection
            should_inject_noise = self.rng.random() < noise_rate
            if should_inject_noise:
                noise_kind = self.rng.choice(["typo", "filler", "repetition", "truncation"])
                if noise_kind == "truncation":
                    response = self.inject_noise(response, "truncation")
                elif noise_kind == "repetition":
                    response = self.inject_noise(response, "repetition")
                else:
                    prompt = self.inject_noise(prompt, noise_kind)
                mutation_type = f"{mutation_type}+noise_{noise_kind}"

            system_prompt = self.rng.choice(SYSTEM_PROMPTS)

            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=prompt),
                ChatMessage(role="assistant", content=response),
            ]

            sample_id = f"synth_{domain}_{i:05d}_{uuid.uuid4().hex[:8]}"

            sample = SyntheticSample(
                id=sample_id,
                messages=messages,
                domain=domain,
                complexity_level=level,
                mutation_type=mutation_type,
                metadata={
                    "seed_topic": base_item["topic"],
                    "noise_injected": should_inject_noise,
                    "generation_timestamp": time.time(),
                },
            )
            samples.append(sample)

        return samples

    @staticmethod
    def save_to_jsonl(samples: List[SyntheticSample], path: str | Path) -> None:
        """Write synthetic samples to ChatML JSONL format."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(s.to_chatml_jsonl() + "\n")

    @staticmethod
    def load_from_jsonl(path: str | Path) -> List[SyntheticSample]:
        """Read synthetic samples from ChatML JSONL format."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        samples: List[SyntheticSample] = []
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    samples.append(SyntheticSample.model_validate(data))
        return samples
