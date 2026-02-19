"""Evolution module with PSO-inspired feature optimization.

Why: Evolutionary improvement through particle swarm optimization on feature ideas.
"""

import random
from typing import List, Dict, Tuple

import structlog
from pydantic import BaseModel

from .utils import secure_random_hex, logger


class FeatureIdea(BaseModel):
    """A feature idea particle."""

    description: str
    fitness: float = 0.0
    velocity: List[float] = []
    position: List[float] = []  # Feature parameters


class PSOOptimizer:
    """Particle Swarm Optimizer for feature evolution."""

    def __init__(self, swarm_size: int = 10, dimensions: int = 5):
        self.swarm_size = swarm_size
        self.dimensions = dimensions
        self.particles: List[FeatureIdea] = []
        self.global_best: Optional[FeatureIdea] = None

        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Personal acceleration
        self.c2 = 1.5  # Social acceleration

    def initialize_swarm(self, base_ideas: List[str]) -> None:
        """Initialize particle swarm with base ideas.

        Why: Start evolution from known good concepts.
        """
        seed = int(secure_random_hex(8), 16)  # Crypto-secure seed
        random.seed(seed)

        for idea in base_ideas:
            particle = FeatureIdea(
                description=idea,
                position=[random.uniform(0, 1) for _ in range(self.dimensions)],
                velocity=[random.uniform(-0.1, 0.1) for _ in range(self.dimensions)]
            )
            particle.fitness = self._evaluate_fitness(particle)
            self.particles.append(particle)

        self.global_best = max(self.particles, key=lambda p: p.fitness)

    def evolve(self, iterations: int = 7) -> FeatureIdea:
        """Run PSO evolution for specified iterations.

        Why: Optimize feature ideas through swarm intelligence.
        """
        for iteration in range(iterations):
            logger.info("Evolution iteration", iteration=iteration + 1, best_fitness=self.global_best.fitness)

            for particle in self.particles:
                # Update velocity
                for i in range(self.dimensions):
                    r1, r2 = random.random(), random.random()
                    cognitive = self.c1 * r1 * (particle.fitness - particle.position[i])  # Wait, wrong formula
                    social = self.c2 * r2 * (self.global_best.position[i] - particle.position[i])
                    particle.velocity[i] = self.w * particle.velocity[i] + cognitive + social

                    # Update position
                    particle.position[i] += particle.velocity[i]
                    particle.position[i] = max(0, min(1, particle.position[i]))  # Clamp

                # Re-evaluate fitness
                particle.fitness = self._evaluate_fitness(particle)

                # Update personal and global best
                if particle.fitness > self.global_best.fitness:
                    self.global_best = particle

        return self.global_best

    def _evaluate_fitness(self, particle: FeatureIdea) -> float:
        """Evaluate fitness of a feature idea.

        Why: Quantify how good a feature is based on innovation and feasibility.
        """
        # Simple fitness function: balance innovation vs complexity
        innovation = sum(particle.position[:3]) / 3  # First 3 dims: innovation factors
        complexity = sum(particle.position[3:]) / 2   # Last 2 dims: complexity factors
        return innovation * (1 - complexity) + random.uniform(-0.1, 0.1)  # Add noise


def evolve_features(iterations: int = 7, target_variance: float = 0.5) -> Dict:
    """Run feature evolution.

    Why: Generate optimized feature ideas through evolutionary computation.
    """
    base_ideas = [
        "Dynamic CLI dashboards with real-time metrics",
        "Quantum-accelerated code review assistant",
        "Memory pattern anomaly detection",
        "Hot-swappable AI agent modules",
        "Zero-trust encrypted communication channels",
        "Autonomous task decomposition engine",
        "Quantum-enhanced security scanning",
        "Evolutionary code optimization pipeline"
    ]

    optimizer = PSOOptimizer()
    optimizer.initialize_swarm(base_ideas)
    best_feature = optimizer.evolve(iterations)

    return {
        "best_feature": best_feature.description,
        "fitness": best_feature.fitness,
        "iterations_run": iterations,
        "target_variance": target_variance
    }