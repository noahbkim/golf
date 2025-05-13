from __future__ import annotations

import numpy as np
import pandas as pd

import abc
from dataclasses import dataclass, field
from typing import TypeAlias

Vector: TypeAlias = np.array

# https://scicomp.stackexchange.com/a/21063
# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods


class Projectile(abc.ABC):
    """Simulates projectile motion."""

    @abc.abstractmethod
    def project(
        self,
        delta: float,
        velocity: Vector,
        position: Vector,
    ) -> Vector:
        """Compute acceleration over a given time interval.

        Parameters:
            delta: the amount of time to simulate acceleration over (s).
            velocity: the current velocity of the projectile (m/s).
            position: the current position of the projectile (m/s).

        Returns:
            The acceleration of the projectile after `delta` seconds.
        """
    
    def update(
        self,
        delta: float,
        velocity: Vector,
        position: Vector,
    ) -> None:
        """Update the projectile after acceleration is simulated.
        
        Parameters:
            delta: the amount of time to pass in this update (s).
            velocity: the current velocity of the projectile (m/s).
            position: the current position of the projectile (m/s).
        """
    
    def simulate(
        self,
        delta: float,
        velocity: Vector | None = None,
        position: Vector | None = None,
    ) -> Iterator[tuple[float, Vector, Vector]]:
        """Iteratively simulate the motion of the projectile.
        
        This method uses Runge-Kutta to calculate the next velocity and
        position at each time step.
        
        Parameters:
            delta: the time step (s).
            velocity: an initial velocity to launch the projectile (m/s).
            position: an initial position to start the projectile at (m).

        Returns:
            An iterator that yields the time, velocity, and position of
            the projectile at every interval. The iterator will yield
            values forever, so it's up to users to choose an end state,
            e.g. the projectile reaching the ground.
        """
    
        t = 0
        h = delta
        v = velocity if velocity is not None else Vector((0, 0, 0))
        x = position if position is not None else Vector((0, 0, 0))
        while True:
            k1_x = v
            k1_v = self.project(h, k1_x, x)
            k2_x = v + k1_v * delta / 2
            k2_v = self.project(h, k2_x, x)
            k3_x = v + k2_v * delta / 2
            k3_v = self.project(h, k3_x, x)
            k4_x = v + k3_v * delta
            k4_v = self.project(h, k4_x, x)
            v = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * h / 6
            x = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) * h / 6
            self.update(h, v, x)
            yield t, v, x
            t += delta
