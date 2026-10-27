"""针对已确认条件错误的可选兼容策略，沿用依赖原有的增密与优化器更新。"""

import inspect

from gsplat import DefaultStrategy
from gsplat.strategy.ops import reset_opa


class DefaultStrategyWithOpacityReset(DefaultStrategy):
    def __post_init__(self):
        super().__post_init__()
        # 只修补已核对的表达式；依赖已修复或改变实现时必须重新检查，禁止重复重置。
        source = inspect.getsource(DefaultStrategy.step_post_backward)
        if "step % self.reset_every == 0 & step > 0" not in source:
            raise RuntimeError("当前增密策略不是已确认的不透明度重置错误版本，请重新核对兼容补丁")

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        super().step_post_backward(params, optimizers, state, step, info, packed=packed)
        if 0 < step < self.refine_stop_iter and step % self.reset_every == 0:
            # 复用依赖的参数替换与动量清零，不能只改不透明度而留下旧优化器状态。
            reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2.0)
