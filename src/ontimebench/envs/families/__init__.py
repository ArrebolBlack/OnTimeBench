"""Simulation environments, one per implementation family.

The paper groups the 14 tasks into five families. Some modules, error messages
and comments use the internal codes T1-T5 instead:

    grasp                        -   Grasp    harmonic_grasp
    driven_interception          T1  Grasp    conveyor, translating, rotating, occluded grasp
    moving_placement             T2  Place    moving and rotating receptacle
    rotating_interaction         T3  Pass     wire passage, full wire passage
    free_dynamics_interception   T4  Capture  ball, ball on table, slider and can capture
    replanning                   T5  Impact   post-impact grasp

Each environment decides success and failure at the physics rate (500 Hz).
"""

__all__: list[str] = []
