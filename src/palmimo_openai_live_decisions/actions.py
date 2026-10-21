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

ACTION_PHRASES = {
    "forward": "前に歩くね。止まってで止まるよ。",
    "backward": "後ろに歩くね。止まってで止まるよ。",
    "creep": "ゆっくり歩くね。止まってで止まるよ。",
    "strafe_left": "左に歩くね。止まってで止まるよ。",
    "strafe_right": "右に歩くね。止まってで止まるよ。",
    "turn_left": "左を向くね。",
    "turn_right": "右を向くね。",
    "dance": "踊るね。",
    "body_tilt": "体を傾けるね。",
    "pushup": "腕立てするね。",
    "wave": "手を振るね。",
    "wave_both": "両手を振るね。",
    "clap": "拍手するね。",
    "bow": "お辞儀するね。",
    "stretch": "伸びをするね。",
    "nod": "うなずくね。",
    "head_shake": "首を横に振るね。",
    "look_left": "左を見るね。",
    "look_right": "右を見るね。",
    "look_up": "上を見るね。",
    "look_down": "下を見るね。",
    "look_center": "正面を向くね。",
}
