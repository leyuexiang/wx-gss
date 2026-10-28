"""固定真实初始化与首张训练图，检查渲染和反向链路的逐位重复性。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir")
    parser.add_argument("audit_dir")
    parser.add_argument("output")
    args = parser.parse_args()
    from gsplat import rasterization
    from pytorch_msssim import SSIM
    from app.dataset import ColmapDataset, ColmapScene
    from app.training_audit import array_digest
    audit = Path(args.audit_dir)
    with np.load(audit / "initialization.npz") as source:
        params = {key[6:]: torch.tensor(source[key], device="cuda", requires_grad=True)
                  for key in source.files if key.startswith("param_")}
    index = int(np.load(audit / "sampled_indices.npy")[0])
    dataset = ColmapDataset(ColmapScene(str(Path(args.task_dir) / "dataset"), 1), uint8_images=True)
    data = dataset[index]
    target = data["image"].cuda().float().div(255)
    height, width = target.shape[1:]
    view = torch.linalg.inv(data["camtoworld"].cuda())[None]
    intrinsics = data["K"].cuda()[None]
    metric = SSIM(data_range=1., size_average=True, channel=3)
    first_prediction, first_gradients, records = None, None, []
    for repeat in range(4):
        # 不执行优化器、不增密、不重新初始化，每次均从同一参数计算。
        for value in params.values():
            value.grad = None
        rendered, _, _ = rasterization(means=params["means"], scales=params["scales"].exp(),
            quats=F.normalize(params["quats"], dim=-1), opacities=params["opacities"].sigmoid(),
            colors=torch.cat([params["sh0"], params["shN"]], dim=1), viewmats=view,
            Ks=intrinsics, width=width, height=height, sh_degree=0, packed=True,
            sparse_grad=False, rasterize_mode="antialiased")
        prediction = rendered[0].permute(2, 0, 1)
        loss = .8 * F.l1_loss(prediction, target) + .2 * (1 - metric(prediction[None], target[None]))
        loss.backward()
        current = prediction.detach().cpu().numpy()
        gradients = {key: value.grad.detach().cpu().numpy().copy() for key, value in params.items()}
        if first_prediction is None:
            first_prediction, first_gradients = current.copy(), gradients
        records.append({"repeat": repeat, "loss": loss.item(),
            "prediction_equal_first": bool(np.array_equal(current, first_prediction)),
            "gradient_sha256": {key: array_digest(value) for key, value in gradients.items()},
            "gradient_max_abs_difference": {key: float(np.max(np.abs(value - first_gradients[key]), initial=0))
                                            for key, value in gradients.items()}})
    with Path(args.output).open("x", encoding="utf-8") as destination:
        json.dump({"image": dataset.scene.names[index], "records": records,
            "scope": "相同参数和图像下的完整渲染/损失反传链路，未单独定位到某个内核"},
                  destination, ensure_ascii=False, indent=2)
    print(json.dumps([{k: row[k] for k in ("repeat", "prediction_equal_first", "gradient_max_abs_difference")}
                      for row in records]))


if __name__ == "__main__":
    main()
