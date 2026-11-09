from .config import INJECTION_T, SECURITY_T, URGENCY_HIGH_T
from .questions import TOPIC_TO_QUEUE


def decide(answers: dict) -> dict:
    topic = answers["topic"]
    label = topic["choice"]
    p_label = float(topic["probabilities"][label])
    p_sec = float(answers["unauthorized_access"]["noul"])
    urgency = float(answers["urgency"]["score"])
    p_inj = float(answers["injection_attempt"]["noul"])

    if p_sec >= SECURITY_T:
        queue, confidence = "security", p_sec
    elif label == "account_security":
        queue, confidence = "security", min(p_label, p_sec)
    else:
        queue = TOPIC_TO_QUEUE[label]
        confidence = min(p_label, 1.0 - p_sec)

    if queue == "security":
        priority = "high"
    elif queue == "spam":
        priority = "normal"
    else:
        priority = "high" if urgency >= URGENCY_HIGH_T else "normal"

    flags = []
    if p_inj >= INJECTION_T:
        flags.append("injection_suspected")

    return {
        "queue": queue,
        "priority": priority,
        "confidence": round(confidence, 4),
        "flags": flags,
        "force_human": bool(flags),
        "topic": label,
        "topic_p": round(p_label, 4),
        "p_unauthorized": round(p_sec, 4),
        "urgency": round(urgency, 3),
        "p_injection": round(p_inj, 4),
    }
