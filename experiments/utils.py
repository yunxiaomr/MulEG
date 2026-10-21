import numpy as np

class RunningMean:
    def __init__(self, p=0.9999):
        self.value = 0
        self.count = -1
        self.p = p

    def update(self, values):
        # [FIX] 自动兼容标量(float/int)和列表
        if not isinstance(values, (list, tuple, np.ndarray)):
            values = [values]
            
        for _v in values:
            self.update_one(_v)

    def update_one(self, value):
        self.count += 1
        # 防止除零错误 (虽实际上count从0开始不会触发，但增加鲁棒性)
        denom = (1 - self.p ** (self.count + 1))
        if abs(denom) < 1e-8: denom = 1e-8
            
        gamma = (1 - self.p) / denom
        self.value = self.value * (1 - gamma) + value * gamma

    def __str__(self):
        return f"{self.value:.4f}"

    def __repr__(self):
        return str(self)

    def __format__(self, spec):
        return format(self.value, spec)


class Mean:
    def __init__(self):
        self.value = 0
        self.count = -1

    def update(self, values):
        # [FIX] 自动兼容标量(float/int)和列表
        if not isinstance(values, (list, tuple, np.ndarray)):
            values = [values]
            
        for _v in values:
            self.update_one(_v)

    def update_one(self, value):
        self.count += 1
        gamma = 1 / (1 + self.count)
        self.value = self.value * (1 - gamma) + value * gamma

    def __str__(self):
        return f"{self.value:.4f}"

    def __repr__(self):
        return str(self)

    def __format__(self, spec):
        return format(self.value, spec)