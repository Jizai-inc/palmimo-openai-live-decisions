"""Fixed invocations of the existing SDK motion tools."""

from typing import Any

ACTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    name: (name, {"seconds": 1.5})
    for name in (
        "forward",
        "backward",
        "creep",
        "dance",
        "body_tilt",
        "pushup",
        "wave",
        "wave_both",
        "clap",
        "bow",
        "stretch",
        "nod",
        "head_shake",
    )
}
ACTIONS.update(
    {
        "turn_left": ("turn", {"direction": "left", "seconds": 1.0}),
        "turn_right": ("turn", {"direction": "right", "seconds": 1.0}),
        "strafe_left": ("strafe", {"direction": "left", "seconds": 1.0}),
        "strafe_right": ("strafe", {"direction": "right", "seconds": 1.0}),
        "look_left": ("look", {"yaw": 15.0}),
        "look_right": ("look", {"yaw": -15.0}),
        "look_up": ("look", {"pitch": 10.0}),
        "look_down": ("look", {"pitch": -10.0}),
        "look_center": ("look_center", {}),
    }
)

ACTIONS["dance"] = ("dance", {"seconds": 4.0})
ACTIONS["clap"] = ("clap", {"seconds": 3.0})
CONTINUOUS_ACTIONS = {"forward", "backward", "creep", "strafe_left", "strafe_right"}

# Describe intent rather than prescribing speech in a particular language.
ACTION_DESCRIPTIONS = {
    "forward": "walk forward until stopped",
    "backward": "walk backward until stopped",
    "creep": "walk slowly until stopped",
    "strafe_left": "walk sideways to the left until stopped",
    "strafe_right": "walk sideways to the right until stopped",
    "turn_left": "turn left",
    "turn_right": "turn right",
    "dance": "dance for four seconds",
    "body_tilt": "tilt the body",
    "pushup": "do a pushup",
    "wave": "wave one hand",
    "wave_both": "wave both hands",
    "clap": "clap",
    "bow": "bow",
    "stretch": "stretch",
    "nod": "nod",
    "head_shake": "shake the head",
    "look_left": "look left",
    "look_right": "look right",
    "look_up": "look up",
    "look_down": "look down",
    "look_center": "look straight ahead",
}
