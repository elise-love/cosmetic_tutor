#prompt.py
from cosmetic_tutor.types import BlushMetrics

def build_blush_prompt(m: BlushMetrics) -> str:
    return f"""
你是一位彩妝教學助理。
以下是使用者目前腮紅分析結果（數值越大代表越紅）：

左臉頰濃淡增量：{m.left_delta_a:.2f}
右臉頰濃淡增量：{m.right_delta_a:.2f}
左右差異值：{m.uneven_score:.2f}
左右覆蓋率：L={m.left_coverage:.2f}, R={m.right_coverage:.2f}
位置偏移：{m.offset_dir}

請友善的給出具體、可操作的腮紅修正建議，

""".strip()