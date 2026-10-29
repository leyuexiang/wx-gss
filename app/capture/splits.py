"""用文件名固定监督集合，重建注册顺序或帧数变化时禁止悄悄更换验证图。"""


def training_indices(names, validation_indices, requested_names=None):
    """允许冻结共同训练视角；未选视角只参与既有几何，不参与颜色损失。"""
    validation = set(validation_indices)
    if requested_names is None:
        return [i for i in range(len(names)) if i not in validation]
    if (not isinstance(requested_names, list) or any(not isinstance(name, str) for name in requested_names)
            or len(set(requested_names)) != len(requested_names)):
        raise ValueError("冻结训练视角必须是不重复的文件名列表")
    requested = set(requested_names)
    if requested - set(names):
        raise ValueError("冻结训练视角存在未注册图像")
    if requested & {names[i] for i in validation}:
        raise ValueError("冻结训练视角不能包含验证图")
    return [i for i, name in enumerate(names) if name in requested]
