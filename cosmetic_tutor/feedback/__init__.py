# __init__.py
from cosmetic_tutor.types import BlushMetrics, Advice
from cosmetic_tutor.config import BlushThresholds


def rule_based_advice(m: BlushMetrics, th: BlushThresholds) -> Advice:
    tips: list[str] = []

    # 1) Uneven blush (left vs right)
    if m.uneven_score > th.uneven_delta_a:
        if m.left_delta_a > m.right_delta_a:
            tips.append("左臉較重，用乾淨刷暈開。")
        else:
            tips.append("右臉較重，用乾淨刷暈開。")

    # 2) Too strong overall
    if (
        m.left_delta_a > th.strong_delta_a
        or m.right_delta_a > th.strong_delta_a
    ):
        tips.append("顏色偏重，先少量疊加。")

    # 3) Position offset
    if m.offset_dir == "TOO_LOW":
        tips.append("位置偏低，往顴骨上提。")
    elif m.offset_dir == "TOO_OUTSIDE":
        tips.append("位置偏外，往內收一點。")

    # 4) Default feedback
    if not tips:
        tips.append("腮紅狀態 OK，可微調邊緣暈染。")

    # Limit to first two tips
    return Advice(
        text="".join(tips[:2]),
        tags=("rule",),
    )
