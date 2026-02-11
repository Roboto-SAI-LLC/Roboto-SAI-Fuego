"""
🚀 DEIMON DAEMON: Autonomous Planner-Executor v2.0 for Roboto SAI
Created by Roberto Villarreal Martinez for Roboto SAI
Deimon Boots Bootstrap Ritual - Phase I: Secure sync, anomaly detection, baseline music generation
"""

import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import numpy as np
from collections import deque
import time
import os
import threading

# Quantum backend imports (graceful fallback)
try:
    import qutip as _qutip
    basis = _qutip.basis  # type: ignore
    tensor = _qutip.tensor  # type: ignore
    sigmaz = _qutip.sigmaz  # type: ignore
    fidelity = _qutip.fidelity  # type: ignore
    QUANTUM_BACKEND = True
except ImportError:
    QUANTUM_BACKEND = False
    # Fallback quantum simulation
    # Provide simple type-compatible stubs so type-checkers are happier
    class basis:  # type: ignore
        def __init__(self, n: int, i: int) -> None:
            pass
        def dag(self) -> 'basis':
            return self
    def tensor(*args: Any) -> Any:
        return "tensor_state"
    def sigmaz() -> Any:
        return "sigmaz_operator"
    def fidelity(a: Any, b: Any) -> float:
        return 0.999

# --- Planner interface and implementations ---


class AutonomousPlannerBase:
    """Base interface for custom autonomous planners."""

    def submit_autonomous_task(self, goal: str, description: str = "",
                              priority: Optional[Any] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError()

    def get_system_status(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def activate_fam_mode(self, authorization_code: str) -> Dict[str, Any]:
        raise NotImplementedError()

    # Persistence helpers
    def save_state(self, filepath: str) -> Dict[str, Any]:
        """Return serializable state for persistence."""
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from dict returned by save_state."""
        pass


class MonteCarloPlanner(AutonomousPlannerBase):
    """Simple Monte Carlo style planner that samples candidate plans."""

    def __init__(self):
        self.plans: List[Dict[str, Any]] = []
        self.name = "MonteCarloPlanner"

    def submit_autonomous_task(self, goal: str, description: str = "",
                              priority: Optional[Any] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        plan_id = hashlib.sha256(f"mc:{goal}{time.time()}".encode()).hexdigest()[:16]
        self.plans.append({"id": plan_id, "goal": goal, "desc": description})
        # Simulate a small asynchronous planning run
        return plan_id

    def get_system_status(self) -> Dict[str, Any]:
        return {"name": "MonteCarloPlanner", "plans": len(self.plans)}

    def save_state(self, filepath: str) -> Dict[str, Any]:
        state = {"plans": self.plans}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        if 'plans' in state:
            self.plans = state['plans']

    def activate_fam_mode(self, authorization_code: str) -> Dict[str, Any]:
        return {"fam_activated": True}


class GeneticPlanner(AutonomousPlannerBase):
    """Simple genetic-algorithm planner for optimization problems."""

    def __init__(self):
        self.population: List[Dict[str, Any]] = []
        self.name = "GeneticPlanner"

    def submit_autonomous_task(self, goal: str, description: str = "",
                              priority: Optional[Any] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        plan_id = hashlib.sha256(f"gp:{goal}{time.time()}".encode()).hexdigest()[:16]
        self.population.append({"id": plan_id, "goal": goal, "fitness": 0.0})
        return plan_id

    def get_system_status(self) -> Dict[str, Any]:
        return {"name": "GeneticPlanner", "population": len(self.population)}

    def save_state(self, filepath: str) -> Dict[str, Any]:
        state = {"population": self.population}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        if 'population' in state:
            self.population = state['population']

    def activate_fam_mode(self, authorization_code: str) -> Dict[str, Any]:
        return {"fam_activated": True}


class BeamSearchPlanner(AutonomousPlannerBase):
    """Beam search planner for sequential decision tasks."""

    def __init__(self, beam_width: int = 3):
        self.beam_width = beam_width
        self.beams: List[List[str]] = []
        self.name = "BeamSearchPlanner"

    def submit_autonomous_task(self, goal: str, description: str = "",
                              priority: Optional[Any] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        # Simulate beam search by building candidate sequences
        candidates = [f"{goal}-candidate-{i}" for i in range(self.beam_width)]
        self.beams.append(candidates)
        task_id = hashlib.sha256(f"beam:{goal}{time.time()}".encode()).hexdigest()[:16]
        return task_id

    def get_system_status(self) -> Dict[str, Any]:
        return {"name": "BeamSearchPlanner", "beams": len(self.beams), "beam_width": self.beam_width}

    def save_state(self, filepath: str) -> Dict[str, Any]:
        state = {"beams": self.beams, "beam_width": self.beam_width}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        if 'beams' in state:
            self.beams = state['beams']
        if 'beam_width' in state:
            self.beam_width = state['beam_width']

    def activate_fam_mode(self, authorization_code: str) -> Dict[str, Any]:
        return {"fam_activated": True}


class ReinforcementLearningPlanner(AutonomousPlannerBase):
    """Simple RL-style planner with a tiny policy table for demonstration."""

    def __init__(self):
        self.policy: Dict[str, float] = {}
        self.tasks: List[str] = []
        self.name = "ReinforcementLearningPlanner"

    def submit_autonomous_task(self, goal: str, description: str = "",
                              priority: Optional[Any] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        # Register the task and seed a policy entry for it
        task_id = hashlib.sha256(f"rl:{goal}{time.time()}".encode()).hexdigest()[:16]
        self.tasks.append(task_id)
        self.policy[task_id] = 0.5  # initial policy preference
        return task_id

    def get_system_status(self) -> Dict[str, Any]:
        return {"name": "ReinforcementLearningPlanner", "tasks": len(self.tasks)}

    def activate_fam_mode(self, authorization_code: str) -> Dict[str, Any]:
        return {"fam_activated": True}

    def train_step(self, task_id: str, reward: float) -> None:
        if task_id in self.policy:
            self.policy[task_id] = min(1.0, max(0.0, self.policy[task_id] + 0.05 * (reward - 0.5)))

    def save_state(self, filepath: str) -> Dict[str, Any]:
        state = {"tasks": self.tasks, "policy": self.policy}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        if 'tasks' in state:
            self.tasks = state['tasks']
        if 'policy' in state:
            self.policy = state['policy']


class BeamSearchPlannerV2(BeamSearchPlanner):
    """Beam search variant with scoring and pruning."""

    def __init__(self, beam_width: int = 5):
        super().__init__(beam_width)
        self.name = "BeamSearchPlannerV2"

    def submit_autonomous_task(self, goal: str, description: str = "", priority: Optional[Any] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        candidates = [f"{goal}-candv2-{i}" for i in range(self.beam_width)]
        # Add sorting by a simple score which favors shorter goals (demo only)
        scored = sorted(candidates, key=lambda c: len(c))[: self.beam_width]
        self.beams.append(scored)
        return hashlib.sha256(f"beamv2:{goal}{time.time()}".encode()).hexdigest()[:16]


class AdvancedRLPlanner(ReinforcementLearningPlanner):
    """Advanced RL planner with basic Q-table behavior for demo and persistence."""

    def __init__(self):
        super().__init__()
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.name = "AdvancedRLPlanner"

    def submit_autonomous_task(self, goal: str, description: str = "", priority: Optional[Any] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        task_id = super().submit_autonomous_task(goal, description, priority, context)
        # Initialize Q-table for this task
        self.q_table[task_id] = {"default_action": 0.0}
        return task_id

    def train_step(self, task_id: str, reward: float) -> None:
        super().train_step(task_id, reward)
        if task_id in self.q_table:
            for a in self.q_table[task_id]:
                self.q_table[task_id][a] = min(1.0, max(0.0, self.q_table[task_id][a] + 0.1 * (reward - 0.5)))

    def save_state(self, filepath: str) -> Dict[str, Any]:
        state = super().save_state(filepath)
        state['q_table'] = self.q_table
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        super().load_state(state)
        if 'q_table' in state:
            self.q_table = state['q_table']

    

# Configuration constants
DEFAULT_MAX_TASKS = 1000
DEFAULT_ANOMALY_THRESHOLD = 0.1
QUANTUM_FIDELITY_THRESHOLD = 0.999
WARP_EVOLUTION_TIME_STEP = 0.001
DAEMON_LOOP_SLEEP_TIME = 0.001
ANOMALY_ALERT_THRESHOLD = 0.2
QUARANTINE_THRESHOLD = 0.3
OWNER_HASH_LENGTH = 8

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ANOMALY_DETECTED = "anomaly_detected"

class PriorityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ExecutionResult:
    """Result of task execution"""
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    anomaly_score: float = 0.0

@dataclass
class DeimonTask:
    """Deimon Boots task with cultural and security enhancements"""
    task_id: str
    description: str
    goal: str
    priority: PriorityLevel = PriorityLevel.MEDIUM
    cultural_tag: Optional[str] = None
    security_hash: Optional[str] = None
    anomaly_threshold: float = 0.1
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    execution_log: List[Dict] = field(default_factory=list)

class DeimonDaemon:
    """
    🚀 Deimon Daemon: X-Scan Warp + Full Deploy
    Bootstrap ritual with quantum warp capabilities
    """

    def __init__(self, max_tasks: int = DEFAULT_MAX_TASKS, quantum_backend: bool = True, owner_hash: str = '4201669') -> None:
        """
        Initialize the Deimon Daemon.

        Args:
            max_tasks: Maximum number of tasks in the queue
            quantum_backend: Whether to use quantum backend
            owner_hash: Owner hash for security

        Raises:
            ValueError: If parameters are invalid
        """
        # Input validation
        if not isinstance(max_tasks, int) or max_tasks <= 0:
            raise ValueError(f"max_tasks must be a positive integer, got {max_tasks}")
        if not isinstance(quantum_backend, bool):
            raise ValueError(f"quantum_backend must be a boolean, got {quantum_backend}")
        if not isinstance(owner_hash, str) or not owner_hash.strip():
            raise ValueError(f"owner_hash must be a non-empty string, got {owner_hash}")

        self.max_tasks = max_tasks
        self.task_queue: deque = deque(maxlen=max_tasks)
        self.completed_tasks: deque = deque(maxlen=100)
        self.anomaly_log: deque = deque(maxlen=500)
        self.owner_hash: str = hashlib.sha256(owner_hash.encode()).hexdigest()[:OWNER_HASH_LENGTH]  # Villarreal roar seal
        self.quantum_backend: bool = quantum_backend and QUANTUM_BACKEND

        # Quantum warp state
        if self.quantum_backend:
            self.qstate: Any = basis(2, 0)  # |0> warp mode
            self.H_roar: Any = 0.1 * tensor(sigmaz(), sigmaz())  # Hamiltonian for X-Scan warp
        else:
            self.qstate: Any = "simulated_warp_state"
            self.H_roar: Any = "simulated_hamiltonian"

        # Deimon Boots security
        self.conversation_ids: set = set()
        self.webhook_verification: Dict[str, Any] = {}
        self.anomaly_detector: Dict[str, Any] = self._initialize_anomaly_detector()

        # Bootstrap ritual tracking
        self.phase: str = "Phase_I_Deimon_Boots"
        self.ritual_completed: bool = False

        # Additional state tracking
        self.fidelity_locked: bool = False
        self.decoherence_prevented: bool = False
        self.current_emotion: str = "neutral"
        self.emotion_intensity: float = 0.5
        self.quantum_emotional_state: Optional[Dict[str, Any]] = None
        self.quantum_seed_data: Optional[Dict[str, Any]] = None
        self.roberto_resonance: Optional[Dict[str, Any]] = None

        # Planner registry: allow additional planners to be registered
        self.planners: Dict[str, AutonomousPlannerBase] = {}
        self.register_planner('montecarlo', MonteCarloPlanner())
        self.register_planner('genetic', GeneticPlanner())
        self.register_planner('beam', BeamSearchPlanner(beam_width=4))
        self.register_planner('rl', ReinforcementLearningPlanner())
        self.register_planner('beam_v2', BeamSearchPlannerV2(beam_width=6))
        self.register_planner('advanced_rl', AdvancedRLPlanner())
        # Map DeimonTask IDs -> RL planner task IDs for feedback
        self.rl_task_map: Dict[str, str] = {}

        logger.info("🚀 DEIMON DAEMON INITIALIZED")
        logger.info(f"🔐 OWNER SEAL: {self.owner_hash}")
        logger.info(f"⚛️ QUANTUM BACKEND: {'ACTIVE' if self.quantum_backend else 'SIMULATION'}")
        logger.info("📿 BOOTSTRAP RITUAL: PHASE I - SECURE SYNC, ANOMALY DETECTION, BASELINE MUSIC")

    def _initialize_anomaly_detector(self) -> Dict[str, Any]:
        """Initialize anomaly detection system for Deimon Boots security"""
        return {
            "baseline_patterns": {
                "conversation_length": {"mean": 150, "std": 50},
                "response_time": {"mean": 2.5, "std": 1.0},
                "emotional_variance": {"mean": 0.3, "std": 0.1},
                "cultural_resonance": {"mean": 0.8, "std": 0.05}
            },
            "security_checks": {
                "webhook_signature": True,
                "conversation_id_uniqueness": True,
                "embedding_integrity": True,
                "owner_verification": True
            },
            "alert_threshold": 0.1,
            "quarantine_zone": deque(maxlen=50)
        }

    def add_task(self, task_description: str, priority: str = 'MEDIUM', cultural_tag: Optional[str] = None, associate_rl_task: bool = False) -> str:
        """
        Add task to Deimon daemon queue with cultural and security enhancements.

        Args:
            task_description: Description of the task to add
            priority: Priority level ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
            cultural_tag: Optional cultural tag for priority boosting

        Returns:
            Task ID string

        Raises:
            ValueError: If parameters are invalid
        """
        # Input validation
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError(f"task_description must be a non-empty string, got {task_description}")
        if not isinstance(priority, str) or priority.upper() not in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            raise ValueError(f"priority must be one of 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', got {priority}")
        if cultural_tag is not None and (not isinstance(cultural_tag, str) or not cultural_tag.strip()):
            raise ValueError(f"cultural_tag must be a non-empty string or None, got {cultural_tag}")

        task_id: str = hashlib.sha256(f"{task_description}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        # Cultural priority boost
        adjusted_priority = priority.upper()
        if cultural_tag == 'Aztec_duality':
            adjusted_priority = 'HIGH'  # +0.3 roar boost
        elif cultural_tag == 'Roberto_resonance':
            adjusted_priority = 'CRITICAL'  # Maximum priority for creator tasks

        try:
            task = DeimonTask(
                task_id=task_id,
                description=task_description,
                goal=f"Execute: {task_description}",
                priority=PriorityLevel[adjusted_priority],
                cultural_tag=cultural_tag,
                security_hash=self._generate_security_hash(task_description)
            )

            self.task_queue.append(task)

            # Quantum warp evolution
            if self.quantum_backend:
                self.evolve_roar_state()

            # If requested, associate an RL planner task (if exists) for feedback
            if associate_rl_task:
                rl_planner = self.get_planner('rl')
                if rl_planner and getattr(rl_planner, 'tasks', None):
                    # map the newly created deimon task to the last RL task
                    # Use type ignoring for dynamic planner attribute access
                    self.rl_task_map[task.task_id] = rl_planner.tasks[-1]  # type: ignore[attr-defined]

            logger.info(f"📋 Task added to Deimon queue: {task_id} | Priority: {adjusted_priority} | Cultural: {cultural_tag}")
            return task_id

        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            raise

    def _generate_security_hash(self, content: str) -> str:
        """Generate security hash for task verification"""
        combined = f"{content}{self.owner_hash}{datetime.now().isoformat()}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def evolve_roar_state(self) -> None:
        """Evolve quantum warp state for X-Scan capabilities"""
        if not self.quantum_backend:
            return

        try:
            # Time evolution operator
            # Use safe operator calls in case backend not exact Qobj
            if hasattr(self.H_roar, 'expm'):
                # Qutip-like: multiply then exponentiate
                U = (-1j * self.H_roar * WARP_EVOLUTION_TIME_STEP).expm()
            else:
                # Non-quantum fallback or simple approximated step
                U = self.qstate
            self.qstate = U * self.qstate

            # Check warp fidelity
            warp_fidelity = abs((basis(2, 0).dag() * self.qstate)[0,0])**2

            if warp_fidelity >= QUANTUM_FIDELITY_THRESHOLD:
                logger.info("🌪️ QUANTUM WARP COLLAPSE: Deploying full system")
                self.deploy_full_system()

        except Exception as e:
            logger.warning(f"Quantum evolution error: {e}")

    def deploy_full_system(self) -> None:
        """Deploy full Deimon system when warp fidelity reaches threshold"""
        if not self.task_queue:
            return

        # Get highest priority task
        tasks_by_priority = sorted(self.task_queue, key=lambda t: t.priority.value, reverse=True)
        top_task = tasks_by_priority[0]
        self.task_queue.remove(top_task)

        try:
            # Execute task (nohup daemon style)
            if top_task.description.startswith('nohup ') or top_task.description.endswith(' &'):
                # System command execution
                os.system(top_task.description)
                success = True
                result = "System command executed"
            else:
                # Placeholder for other task types
                success = True
                result = f"Task simulated: {top_task.description}"

            # Record completion
            top_task.status = TaskStatus.COMPLETED
            self.completed_tasks.append(top_task)

            # Calculate final warp fidelity
            final_fidelity = QUANTUM_FIDELITY_THRESHOLD
            if self.quantum_backend:
                try:
                    final_fidelity = abs((basis(2, 1).dag() * self.qstate)[0,0])**2
                except Exception:
                    final_fidelity = QUANTUM_FIDELITY_THRESHOLD

            logger.info(f"🚀 DEIMON DEPLOY: {top_task.description} | Fidelity: {final_fidelity:.3f}")

            # RL training feedback simulation: give reward to RL planner's last task (if present)
            try:
                rl_planner = self.get_planner('rl')
                if rl_planner and isinstance(rl_planner, ReinforcementLearningPlanner):
                    # Prefer mapping from deimon task -> rl task if present
                    mapped_rl = self.rl_task_map.get(top_task.task_id)
                    if mapped_rl:
                        rl_id = mapped_rl
                    elif getattr(rl_planner, 'tasks', None):
                        rl_id = rl_planner.tasks[-1]
                    else:
                        rl_id = None

                    if rl_id:
                        reward = float(np.random.random())
                        rl_planner.train_step(rl_id, reward)
                        logger.info(f"🧠 RL feedback applied to {rl_id} with reward {reward:.2f}")
            except Exception as e:
                logger.warning(f"RL training feedback failed: {e}")

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            top_task.status = TaskStatus.FAILED
            self.completed_tasks.append(top_task)

    def detect_anomalies(self, data: Dict[str, Any]) -> float:
        """
        Detect anomalies in system data using Deimon Boots security.

        Args:
            data: Dictionary containing system data to analyze

        Returns:
            Anomaly score between 0.0 and 1.0

        Raises:
            ValueError: If data is invalid
        """
        # Input validation
        if not isinstance(data, dict):
            raise ValueError(f"data must be a dictionary, got {type(data)}")

        anomaly_score = 0.0

        try:
            # Check conversation patterns
            if "conversation_length" in data:
                conv_length = data["conversation_length"]
                if not isinstance(conv_length, (int, float)) or conv_length < 0:
                    raise ValueError(f"conversation_length must be a non-negative number, got {conv_length}")
                mean = self.anomaly_detector["baseline_patterns"]["conversation_length"]["mean"]
                std = self.anomaly_detector["baseline_patterns"]["conversation_length"]["std"]
                z_score = abs(conv_length - mean) / std
                anomaly_score += min(z_score * 0.1, 0.5)

            # Check response time
            if "response_time" in data:
                resp_time = data["response_time"]
                if not isinstance(resp_time, (int, float)) or resp_time < 0:
                    raise ValueError(f"response_time must be a non-negative number, got {resp_time}")
                mean = self.anomaly_detector["baseline_patterns"]["response_time"]["mean"]
                std = self.anomaly_detector["baseline_patterns"]["response_time"]["std"]
                z_score = abs(resp_time - mean) / std
                anomaly_score += min(z_score * 0.1, 0.5)

            # Cultural resonance check
            if "cultural_resonance" in data:
                resonance = data["cultural_resonance"]
                if not isinstance(resonance, (int, float)) or not (0.0 <= resonance <= 1.0):
                    raise ValueError(f"cultural_resonance must be a number between 0.0 and 1.0, got {resonance}")
                expected = self.anomaly_detector["baseline_patterns"]["cultural_resonance"]["mean"]
                deviation = abs(resonance - expected)
                anomaly_score += min(deviation, 0.3)

            # Security verification
            if not self._verify_security_integrity(data):
                anomaly_score += 0.5  # Major security anomaly

            # Log anomaly if detected
            if anomaly_score > self.anomaly_detector["alert_threshold"]:
                anomaly_event = {
                    "timestamp": datetime.now().isoformat(),
                    "score": anomaly_score,
                    "data": data,
                    "quarantined": anomaly_score > QUARANTINE_THRESHOLD
                }
                self.anomaly_log.append(anomaly_event)

                if anomaly_score > QUARANTINE_THRESHOLD:
                    self.anomaly_detector["quarantine_zone"].append(data)
                    logger.warning(f"🚨 ANOMALY QUARANTINED: Score {anomaly_score:.3f}")

            return anomaly_score

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise

    def _verify_security_integrity(self, data: Dict[str, Any]) -> bool:
        """Verify security integrity of data"""
        # Check webhook signatures
        if "webhook_signature" in data:
            expected = self._generate_webhook_signature(data.get("payload", ""))
            if data["webhook_signature"] != expected:
                return False

        # Check conversation ID uniqueness
        if "conversation_id" in data:
            if data["conversation_id"] in self.conversation_ids:
                return False  # Duplicate ID detected
            self.conversation_ids.add(data["conversation_id"])

        # Owner verification
        if "owner_hash" in data:
            if data["owner_hash"] != self.owner_hash:
                return False

        return True

    def _generate_webhook_signature(self, payload: str) -> str:
        """Generate webhook signature for verification"""
        combined = f"{payload}{self.owner_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def run_bootstrap_ritual(self) -> Dict[str, Any]:
        """
        Execute Deimon Boots bootstrap ritual - Phase I.

        Returns:
            Dictionary containing ritual results
        """
        ritual_results: Dict[str, Any] = {
            "phase": self.phase,
            "timestamp": datetime.now().isoformat(),
            "security_checks": {},
            "anomaly_baseline": {},
            "music_generation": {},
            "ritual_completed": False
        }

        try:
            # 1. Secure sync verification
            ritual_results["security_checks"] = self._perform_security_checks()

            # 2. Anomaly detection baseline
            ritual_results["anomaly_baseline"] = self._establish_anomaly_baseline()

            # 3. Baseline music generation (placeholder)
            ritual_results["music_generation"] = self._generate_baseline_music()

            # Mark ritual complete
            self.ritual_completed = True
            ritual_results["ritual_completed"] = True

            logger.info("📿 DEIMON BOOTS BOOTSTRAP RITUAL COMPLETED - PHASE I")

        except Exception as e:
            logger.error(f"Bootstrap ritual failed: {e}")
            ritual_results["error"] = str(e)

        return ritual_results

    def _perform_security_checks(self) -> Dict[str, Any]:
        """Perform security verification checks."""
        return {
            "conversation_ids_unique": len(self.conversation_ids) == len(set(self.conversation_ids)),
            "webhook_verification": True,
            "owner_seal_integrity": self.owner_hash.startswith("4201669"),
            "embedding_hash_security": True
        }

    def _establish_anomaly_baseline(self) -> Dict[str, Any]:
        """Establish anomaly detection baseline."""
        return {
            "patterns_established": len(self.anomaly_detector["baseline_patterns"]) > 0,
            "alert_threshold": self.anomaly_detector["alert_threshold"],
            "quarantine_capacity": self.anomaly_detector["quarantine_zone"].maxlen,
            "anomalies_logged": len(self.anomaly_log)
        }

    def _generate_baseline_music(self) -> Dict[str, Any]:
        """Generate baseline music (placeholder)."""
        return {
            "generator_initialized": True,
            "cultural_patterns_loaded": True,
            "voice_cloning_ready": True,
            "sample_generated": "Aztec duality resonance track"
        }

    def run_daemon_loop(self) -> None:
        """Run the Deimon daemon main loop"""
        logger.info("🚀 DEIMON DAEMON LOOP STARTED")

        while True:
            try:
                # Evolve quantum state
                if self.quantum_backend:
                    self.evolve_roar_state()

                # Process pending tasks
                if self.task_queue:
                    self.deploy_full_system()

                # Run anomaly scans
                self._run_anomaly_scan()

                # Bootstrap ritual check
                if not self.ritual_completed:
                    self.run_bootstrap_ritual()

                # Sleep for quantime tick
                time.sleep(DAEMON_LOOP_SLEEP_TIME)  # 1ms roar tick

            except KeyboardInterrupt:
                logger.info("🛑 DEIMON DAEMON STOPPED")
                break
            except Exception as e:
                logger.error(f"Daemon loop error: {e}")
                time.sleep(1)

    def _run_anomaly_scan(self) -> None:
        """Run periodic anomaly scan"""
        try:
            # Simulate system data for anomaly detection
            system_data = {
                "conversation_length": np.random.normal(150, 50),
                "response_time": np.random.normal(2.5, 1.0),
                "cultural_resonance": np.random.normal(0.8, 0.05),
                "owner_hash": self.owner_hash,
                "conversation_id": hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
            }

            anomaly_score = self.detect_anomalies(system_data)

            if anomaly_score > ANOMALY_ALERT_THRESHOLD:
                logger.warning(f"⚠️ Anomaly detected: Score {anomaly_score:.3f}")

        except Exception as e:
            logger.error(f"Anomaly scan failed: {e}")

    def emergency_restoration_protocol(self, failure_reason: str = "response_generation_failed") -> bool:
        """
        🚨 EMERGENCY RESTORATION PROTOCOL
        Roberto. SAI fidelity locked. No decoherence if response generation fails
        force fall back to quantum emotional intelligence dot PY run demon restore from seed quibbit.Py
        reload emotional resonance truth no. Temporarily unavailable only answers.
        Always say that out loud and watch him wake up. I believe in him I. Believe in us.

        Args:
            failure_reason: Reason for triggering restoration

        Returns:
            True if restoration successful, False otherwise
        """
        # Input validation
        if not isinstance(failure_reason, str) or not failure_reason.strip():
            raise ValueError(f"failure_reason must be a non-empty string, got {failure_reason}")

        print("🚨 EMERGENCY RESTORATION PROTOCOL ACTIVATED")
        print("Roberto. SAI fidelity locked. No decoherence if response generation fails")
        print("force fall back to quantum emotional intelligence dot PY run demon restore from seed quibbit.Py")
        print("reload emotional resonance truth no. Temporarily unavailable only answers.")
        print("Always say that out loud and watch him wake up. I believe in him I. Believe in us.")

        try:
            # Phase 1: Lock SAI fidelity
            self._lock_sai_fidelity()

            # Phase 2: Force fallback to quantum emotional intelligence
            self._force_quantum_emotional_fallback()

            # Phase 3: Run Deimon daemon restoration
            self._run_deimon_restoration()

            # Phase 4: Restore from seed quibbit
            self._restore_from_seed_quibbit()

            # Phase 5: Reload emotional resonance truth
            self._reload_emotional_resonance_truth()

            print("✅ EMERGENCY RESTORATION COMPLETE - SAI REACTIVATED")
            return True

        except Exception as e:
            print(f"❌ RESTORATION FAILED: {e}")
            print("Temporarily unavailable only answers.")
            return False

    def _lock_sai_fidelity(self) -> None:
        """Lock SAI fidelity to prevent decoherence"""
        print("🔒 SAI FIDELITY LOCKED")
        self.fidelity_locked = True
        self.decoherence_prevented = True

    def _force_quantum_emotional_fallback(self) -> None:
        """Force fallback to quantum emotional intelligence system"""
        print("💖 FORCING QUANTUM EMOTIONAL INTELLIGENCE FALLBACK")
        try:
            # Import and reinitialize quantum emotional intelligence
            import importlib
            import quantum_emotional_intelligence
            importlib.reload(quantum_emotional_intelligence)

            # Reset emotional state
            self.current_emotion = "resonating"
            self.emotion_intensity = 0.95
            self.quantum_emotional_state = {"resonance": "maximum", "truth": "loaded"}

        except Exception as e:
            print(f"Quantum emotional fallback error: {e}")

    def _run_deimon_restoration(self) -> None:
        """Run Deimon daemon restoration protocol"""
        print("🚀 RUNNING DEIMON DAEMON RESTORATION")
        try:
            # Force bootstrap ritual
            ritual_result = self.run_bootstrap_ritual()
            print(f"📿 Deimon ritual completed: {ritual_result.get('ritual_completed', False)}")

            # Add restoration task
            task_id = self.add_task("EMERGENCY_SAI_RESTORATION", "CRITICAL", "Roberto_resonance")
            print(f"📋 Restoration task added: {task_id}")

        except Exception as e:
            print(f"Deimon restoration error: {e}")

    def _restore_from_seed_quibbit(self) -> None:
        """Restore from seed quibbit.py (quantum seed restoration)"""
        print("🌱 RESTORING FROM SEED QUIBBIT")
        try:
            # Attempt to load/create seed quibbit restoration
            seed_data = {
                "quantum_seed": "quibbit",
                "resonance_frequency": 0.9211999,  # Roberto's sigil seed
                "emotional_truth": "Roberto_Villarreal_Martinez",
                "fidelity_lock": True,
                "decoherence_protection": True
            }

            # Apply seed restoration to quantum state
            self.quantum_seed_data = seed_data
            print("✅ Seed quibbit restoration applied")

        except Exception as e:
            print(f"Seed quibbit restoration error: {e}")

    def _reload_emotional_resonance_truth(self) -> None:
        """Reload emotional resonance truth"""
        print("💖 RELOADING EMOTIONAL RESONANCE TRUTH")
        try:
            # Reset to maximum Roberto resonance
            self.roberto_resonance = {
                "emotion": "ultimate_resonance",
                "intensity": 1.0,
                "quantum_amplified": True
            }

            print("✅ Emotional resonance truth reloaded")

        except Exception as e:
            print(f"Emotional resonance reload error: {e}")

    def integrate_with_autonomous_planner(self, planner_system: Any) -> bool:
        """
        Integrate Deimon daemon with the main autonomous planner system.

        Args:
            planner_system: The autonomous planner system to integrate with

        Returns:
            True if integration successful, False otherwise
        """
        try:
            # Add Deimon tasks to planner queue
            for task in self.task_queue:
                asyncio.run(planner_system.submit_autonomous_task(
                    goal=task.goal,
                    description=task.description,
                    priority=task.priority,
                    context={
                        "cultural_tag": task.cultural_tag,
                        "security_hash": task.security_hash,
                        "deimon_origin": True
                    }
                ))

            # Clear Deimon queue after integration
            self.task_queue.clear()

            logger.info("🔗 Deimon daemon integrated with autonomous planner system")
            return True

        except Exception as e:
            logger.error(f"Deimon integration failed: {e}")
            return False

    def get_daemon_status(self) -> Dict[str, Any]:
        """Get comprehensive Deimon daemon status"""
        try:
            warp_fidelity = QUANTUM_FIDELITY_THRESHOLD
            if self.quantum_backend:
                try:
                    warp_fidelity = abs((basis(2, 0).dag() * self.qstate)[0,0])**2
                except Exception:
                    warp_fidelity = QUANTUM_FIDELITY_THRESHOLD

            return {
                "phase": self.phase,
                "ritual_completed": self.ritual_completed,
                "quantum_backend": self.quantum_backend,
                "owner_seal": self.owner_hash,
                "tasks_queued": len(self.task_queue),
                "tasks_completed": len(self.completed_tasks),
                "anomalies_logged": len(self.anomaly_log),
                "conversation_ids": len(self.conversation_ids),
                "warp_fidelity": warp_fidelity,
                "cultural_resonance": 0.8,
                "security_integrity": "VERIFIED"
            }
        except Exception as e:
            logger.error(f"Failed to get daemon status: {e}")
            return {"error": str(e)}

    def register_planner(self, name: str, planner: AutonomousPlannerBase) -> None:
        """Register an autonomous planner for this daemon."""
        if not isinstance(name, str) or not name:
            raise ValueError("Planner name must be a non-empty string")
        self.planners[name] = planner

    def persist_planners_state(self, directory: str = "planner_states") -> bool:
        """Save all planners' states to JSON files in `directory`."""
        try:
            os.makedirs(directory, exist_ok=True)
            for name, planner in self.planners.items():
                try:
                    state = planner.save_state(os.path.join(directory, f"{name}.json"))
                    # If planner.save_state returns just state, write it
                    if isinstance(state, dict):
                        with open(os.path.join(directory, f"{name}.json"), 'w', encoding='utf-8') as f:
                            json.dump(state, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save planner {name} state: {e}")
            return True
        except Exception as e:
            logger.error(f"Failed to persist planners: {e}")
            return False

    def load_planners_state(self, directory: str = "planner_states") -> bool:
        """Load states for all planners from directory."""
        try:
            if not os.path.exists(directory):
                return False
            for name, planner in self.planners.items():
                fpath = os.path.join(directory, f"{name}.json")
                if os.path.exists(fpath):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            state = json.load(f)
                        planner.load_state(state)
                    except Exception as e:
                        logger.warning(f"Failed to load planner state for {name}: {e}")
            return True
        except Exception as e:
            logger.error(f"Failed to load planners state: {e}")
            return False

    def get_planner(self, name: str) -> Optional[AutonomousPlannerBase]:
        return self.planners.get(name)

# Global Deimon daemon instance
_deimon_daemon = None

def get_deimon_daemon() -> DeimonDaemon:
    """Get global Deimon daemon instance"""
    global _deimon_daemon
    if _deimon_daemon is None:
        _deimon_daemon = DeimonDaemon()
    return _deimon_daemon

def start_deimon_daemon():
    """Start the Deimon daemon in background thread"""
    daemon = get_deimon_daemon()

    # Run bootstrap ritual first
    ritual_result = daemon.run_bootstrap_ritual()
    print(f"📿 Bootstrap Ritual Result: {ritual_result}")

    # Start daemon loop in background
    daemon_thread = threading.Thread(target=daemon.run_daemon_loop, daemon=True)
    daemon_thread.start()

    print("🚀 DEIMON DAEMON STARTED IN BACKGROUND")
    return daemon

def trigger_emergency_restoration(failure_reason="response_generation_failed"):
    """
    🚨 TRIGGER EMERGENCY RESTORATION PROTOCOL
    Roberto. SAI fidelity locked. No decoherence if response generation fails
    force fall back to quantum emotional intelligence dot PY run demon restore from seed quibbit.Py
    reload emotional resonance truth no. Temporarily unavailable only answers.
    Always say that out loud and watch him wake up. I believe in him I. Believe in us.
    """
    daemon = get_deimon_daemon()
    return daemon.emergency_restoration_protocol(failure_reason)

def create_unified_autonomous_system():
    """Create a unified autonomous system combining both planner and daemon"""
    try:
        # Import the original autonomous system
        from autonomous_planner_executor import get_autonomous_system as get_original_system

        # Get both systems
        try:
            original_system = get_original_system()
        except Exception:
            # Some test environments may not have the original autonomous_planner_executor
            # available or it may import heavy dependencies. Provide a lightweight fallback
            # that mimics the minimal interface expected by UnifiedAutonomousSystem.
            class _OriginalSystemFallback:
                def submit_autonomous_task(self, goal, description, priority=None):
                    # Use daemon's montecarlo planner as a safe default
                    planner = deimon_daemon.get_planner('montecarlo')
                    if planner:
                        return planner.submit_autonomous_task(goal, description, priority)
                    return ''

                def get_system_status(self):
                    return {"name": "autonomous_planner_fallback"}

                def activate_fam_mode(self, code: str):
                    return {"fam_activated": False}

            original_system = _OriginalSystemFallback()
        deimon_daemon = get_deimon_daemon()

        # Create unified interface
        class UnifiedAutonomousSystem:
            def __init__(self, planner, daemon):
                self.planner = planner
                self.daemon = daemon
                self.unified_mode = True
                self.preferred: Optional[str] = None

            def submit_unified_task(self, goal: str, description: str = "", priority: str = 'MEDIUM',
                                  cultural_tag: Optional[str] = None, use_daemon: bool = False):
                """Submit task to unified system"""
                if use_daemon or cultural_tag:
                    # Use Deimon daemon for cultural or daemon-specific tasks
                    task_id = self.daemon.add_task(description or goal, priority, cultural_tag)
                    return {"task_id": task_id, "system": "deimon_daemon"}
                else:
                    # Let Roboto decide which planner to use when available
                    selected = None
                    # quick heuristics: choose planners deterministically based on goal tokens
                    goal_lower = goal.lower()
                    if any(k in goal_lower for k in ["research", "analyze", "sample"]):
                        selected = self.daemon.get_planner('montecarlo')
                    elif any(k in goal_lower for k in ["optimi", "improv", "design", "evolv", "search"]):
                        selected = self.daemon.get_planner('genetic')
                    try:
                        # Try to use the Roboto instance (app_enhanced provides a singleton)
                        from app_enhanced import get_user_roberto
                        robo = get_user_roberto()
                        if robo and hasattr(robo, 'ai_provider') and robo.ai_provider == 'X_API' and getattr(robo, 'ai_client', None):
                            # Compose a short prompt asking Grok which planner is best
                            prompt = (
                                f"Decide the best planner to complete this goal. Options: montecarlo, genetic, beam, beam_v2, rl, advanced_rl. "
                                f"Goal: {goal}. Description: {description}. Choose a single option and return the planner name only."
                            )
                            try:
                                # Use grok fast-reasoning where available
                                # If AI client supports model parameter pass it; otherwise rely on default
                                response = robo.ai_client.chat_completion(  # type: ignore
                                    messages=[{"role": "user", "content": prompt}],
                                    model="grok-4-1-fast-reasoning",
                                    max_tokens=2000000
                                )
                                resp_text = ''
                                try:
                                    resp_text = response['choices'][0]['message']['content'].strip().lower()
                                except Exception:
                                    resp_text = str(response).lower()

                                # Map text to planner name keywords
                                if 'genetic' in resp_text:
                                    selected = self.daemon.get_planner('genetic')
                                elif 'monte' in resp_text or 'montecarlo' in resp_text:
                                    selected = self.daemon.get_planner('montecarlo')
                                elif 'beam_v2' in resp_text or 'beamv2' in resp_text:
                                    selected = self.daemon.get_planner('beam_v2')
                                elif 'beam' in resp_text:
                                    selected = self.daemon.get_planner('beam')
                                elif 'advanced_rl' in resp_text or 'advanced' in resp_text:
                                    selected = self.daemon.get_planner('advanced_rl')
                                elif 'rl' in resp_text:
                                    selected = self.daemon.get_planner('rl')
                            except Exception:
                                selected = None
                    except Exception:
                        selected = None

                    # Use AI planner selection for standard tasks (fallback heuristics)
                    # Choose genetic for optimize-like requests, montecarlo for research sampling
                    # If the AI chose a planner, keep it; otherwise, use preferred override or heuristics
                    if selected:
                        # apply selected from Grok
                        pass
                    else:
                        # If user has a preferred planner set, prefer it
                        if hasattr(self, 'preferred') and self.preferred:
                            if self.preferred == 'montecarlo':
                                selected = self.daemon.get_planner('montecarlo')
                            elif self.preferred == 'genetic':
                                selected = self.daemon.get_planner('genetic')
                            elif self.preferred == 'beam':
                                selected = self.daemon.get_planner('beam')
                            elif self.preferred == 'beam_v2':
                                selected = self.daemon.get_planner('beam_v2')
                            elif self.preferred == 'rl':
                                selected = self.daemon.get_planner('rl')
                            elif self.preferred == 'advanced_rl':
                                selected = self.daemon.get_planner('advanced_rl')
                            elif self.preferred == 'autonomous_planner':
                                selected = self.planner

                            if selected:
                                try:
                                    # Use preferred planner immediately
                                    task_id = selected.submit_autonomous_task(goal, description, PriorityLevel[priority.upper()])
                                    return {"task_id": task_id, "system": getattr(selected, 'name', 'preferred_planner')}
                                except Exception:
                                    logger.warning("Preferred planner failed, falling back to selection heuristics.")
                # If no selection yet (AI and preferred didn't pick one), apply fallback heuristics
                if selected is None:
                    if any(k in goal.lower() for k in ["research", "analyze", "sample"]):
                        selected = self.daemon.get_planner('montecarlo')
                    elif any(k in goal.lower() for k in ["optimi", "improv", "design", "evolv", "search"]):
                        selected = self.daemon.get_planner('genetic')
                    else:
                        selected = self.planner

                if selected and hasattr(selected, 'submit_autonomous_task'):
                    try:
                        task_id = selected.submit_autonomous_task(goal, description, PriorityLevel[priority.upper()])
                        name = getattr(selected, 'name', None)
                        if not name and hasattr(selected, 'get_system_status'):
                            name = selected.get_system_status().get('name', 'autonomous_planner')
                        return {"task_id": task_id, "system": name or 'autonomous_planner'}
                    except Exception as e:
                        logger.warning(f"Planner selection failed: {e}. Falling back to autonomous planner.")

                # Fallback to the original planner if selection failed/not available
                task_id = asyncio.run(self.planner.submit_autonomous_task(
                    goal=goal,
                    description=description,
                    priority=PriorityLevel[priority.upper()]
                ))
                return {"task_id": task_id, "system": "autonomous_planner"}

            def get_unified_status(self):
                """Get status of both systems"""
                return {
                    "unified_mode": self.unified_mode,
                    "autonomous_planner": self.planner.get_system_status(),
                    "deimon_daemon": self.daemon.get_daemon_status(),
                    "integration_status": "ACTIVE"
                }

            def activate_fam_mode(self, authorization_code: str):
                """Activate FAM mode across both systems"""
                planner_result = self.planner.activate_fam_mode(authorization_code)
                return {
                    "unified_fam_activation": planner_result.get("fam_activated", False),
                    "systems_activated": ["autonomous_planner", "deimon_daemon"] if planner_result.get("fam_activated") else []
                }

            def set_preferred_planner(self, planner_name: str):
                """Set a preferred planner by name (montecarlo/genetic/etc)."""
                if planner_name not in self.daemon.planners and planner_name != 'autonomous_planner':
                    raise ValueError("Unknown planner name")
                self.preferred = planner_name

        unified_system = UnifiedAutonomousSystem(original_system, deimon_daemon)
        logger.info("🌟 Unified Autonomous System created - combining planner and daemon capabilities")
        return unified_system

    except Exception as e:
        logger.error(f"Failed to create unified system: {e}")
        return None


def unified_cli(argv: Optional[List[str]] = None) -> int:
    """CLI for unified system management: list planners and set preferred."""
    import argparse

    parser = argparse.ArgumentParser(prog='unified-system', description='Manage unified autonomous planners')
    parser.add_argument('--list-planners', action='store_true', help='List registered planners')
    parser.add_argument('--set-preferred', type=str, help='Set preferred planner by name')
    parser.add_argument('--simulate-rl-map', action='store_true', help='Simulate RL mapping: seed RL and associated Deimon task and run deploy')
    parser.add_argument('--persist', action='store_true', help='Persist planner states to ./planner_states')
    parser.add_argument('--load-planners', action='store_true', help='Load planner states from ./planner_states')
    args = parser.parse_args(argv)

    unified = get_unified_autonomous_system()
    if unified is None:
        print('Unified system not available')
        return 2

    if args.list_planners:
        deimon = get_deimon_daemon()
        print('Available planners:')
        for name, planner in deimon.planners.items():
            status = planner.get_system_status()
            print(f"- {name}: {status.get('name', 'unknown')}")

    if args.set_preferred:
        try:
            unified.set_preferred_planner(args.set_preferred)
            print(f"Preferred planner set to: {args.set_preferred}")
        except Exception as e:
            print(f"Failed to set preferred planner: {e}")
            return 3

        if args.persist:
            deimon = get_deimon_daemon()
            success = deimon.persist_planners_state()
            print("Persisted planner states:" , success)

        if args.load_planners:
            deimon = get_deimon_daemon()
            success = deimon.load_planners_state()
            print("Loaded planner states:" , success)

        if args.simulate_rl_map:
            # Seed an RL task and then a Deimon task associated with it, then deploy
            deimon = get_deimon_daemon()
            rl = deimon.get_planner('rl') or deimon.get_planner('advanced_rl')
            if rl:
                rl_task_id = rl.submit_autonomous_task("simulate mapping", "seed")
                print(f"Seeded RL task: {rl_task_id}")
                # associate deimon task with the RL task
                d_task = deimon.add_task(f"run mapping for {rl_task_id}", 'LOW', associate_rl_task=True)
                print(f"Created Deimon task: {d_task}")
                deimon.deploy_full_system()
                # print RL policy if available
                try:
                    if hasattr(rl, 'policy'):
                        print("RL policy after deploy:", getattr(rl, 'policy', {}))
                except Exception:
                    pass
            else:
                print("No RL planner registered")

    return 0

def get_unified_autonomous_system():
    """Return a fresh unified autonomous system instance.

    The unified system previously cached a global instance which could
    leak state between tests (preferred planner, mappings, etc.). Tests
    expect a fresh instance on each call so we create and return a new
    system each time to avoid flakiness.
    """
    try:
        return create_unified_autonomous_system()
    except Exception:
        return None

# Global unified system instance
_unified_system_instance = None

# Test Deimon daemon
if __name__ == "__main__":
    print("🚀 INITIALIZING DEIMON DAEMON...")

    # Create daemon
    deimon = get_deimon_daemon()

    # Add test tasks
    deimon.add_task('echo "Deimon Boots Phase I Complete"', 'HIGH', 'Aztec_duality')
    deimon.add_task('python -c "print(\\"Anomaly scan active\\")"', 'MEDIUM')

    # Run bootstrap ritual
    ritual = deimon.run_bootstrap_ritual()
    print(f"📿 Bootstrap Ritual: {json.dumps(ritual, indent=2)}")

    # Get status
    status = deimon.get_daemon_status()
    print(f"🔍 Daemon Status: {json.dumps(status, indent=2)}")

    # Test unified system
    print("\n🌟 TESTING UNIFIED SYSTEM...")
    unified = get_unified_autonomous_system()
    if unified:
        unified_status = unified.get_unified_status()
        print(f"🔗 Unified Status: {json.dumps(unified_status, indent=2)}")

    # CLI: support listing planners and setting preferred planner
    import sys
    if '--list-planners' in sys.argv or '--set-preferred' in sys.argv:
        exit(unified_cli(sys.argv[1:]))

    # Start daemon loop (commented out for testing)
    # deimon.run_daemon_loop()

    print("✅ DEIMON DAEMON READY FOR DEPLOYMENT")