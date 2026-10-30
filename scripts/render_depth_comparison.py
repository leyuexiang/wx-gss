"""输出固定留出视角与相机平移探针对照图；探针没有真实参考，不计算精度。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('trial')
    parser.add_argument('--groups', nargs='+', default=['none', 'sparse', 'dense'], help='按相同参考评估的实验组目录')
    parser.add_argument('--evaluation-subdir', default='evaluation/validation', help='各组的共同参考评估目录')
    args = parser.parse_args()
    from app.dataset import ColmapScene
    from app.evaluation import load_ply_params
    from gsplat import rasterization
    task, trial = Path(args.task), Path(args.trial)
    scene = ColmapScene(str(task / 'dataset'), 1, rigid_normalization=True)
    kinds = tuple(args.groups)
    evaluation = Path(args.evaluation_subdir)
    metrics = json.loads((trial / kinds[0] / evaluation / 'metrics.json').read_text(encoding='utf-8'))
    records = metrics['records']
    # 完整联系表保留全部验证图，避免仅挑选看起来改善的方向。
    canvas = Image.new('RGB', ((len(kinds) + 1) * 200, len(records) * 270), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for row, record in enumerate(records):
        width, height = record['width'], record['height']
        filename = f"view_{record['index']:04d}.png"
        for col, label in enumerate(('reference',) + kinds):
            kind = kinds[0] if col == 0 else label
            with Image.open(trial / kind / evaluation / filename) as panels:
                left = 0 if col == 0 else width
                tile = panels.crop((left, 24, left + width, 24 + height)).resize((200, 245), Image.Resampling.LANCZOS)
            canvas.paste(tile, (col * 200, row * 270 + 25))
            draw.text((col * 200 + 4, row * 270 + 5), label + ' ' + record['name'], fill='white')
    canvas.save(trial / 'validation-contact.jpg', quality=92)
    names = ['frame_0008.png', 'frame_0032.png', 'frame_0067.png', 'frame_0109.png']
    camera_records, renders = [], {kind: [] for kind in kinds}
    center = np.median(scene.points, axis=0)
    for name in names:
        i = scene.names.index(name)
        for sign in (-1, 1):
            camera = scene.camtoworlds[i].copy()
            distance = float(np.linalg.norm(camera[:3, 3] - center))
            # 固定沿相机右轴平移距离的 2%，保留旋转；只作邻域破损观察，不模拟真实新拍摄。
            camera[:3, 3] += sign * .02 * distance * camera[:3, 0]
            K = scene.Ks[i].copy()
            dim = scene.dimensions[i]
            width, height = dim['width'] // 3, dim['height'] // 3
            K[0] *= width / dim['width']
            K[1] *= height / dim['height']
            camera_records.append({'name': name, 'shift_sign': sign, 'camtoworld': camera.tolist(),
                                   'K': K.tolist(), 'width': width, 'height': height})
    for kind in kinds:
        params = load_ply_params(str(trial / kind / 'point_cloud.ply'))
        colors = torch.cat([params['sh0'], params['shN']], dim=1)
        for record in camera_records:
            rgb, _, _ = rasterization(means=params['means'], quats=F.normalize(params['quats'], dim=-1),
                scales=params['scales'].exp(), opacities=params['opacities'].sigmoid(), colors=colors,
                viewmats=torch.linalg.inv(torch.tensor(record['camtoworld'], device='cuda', dtype=torch.float32))[None],
                Ks=torch.tensor(record['K'], device='cuda', dtype=torch.float32)[None],
                width=record['width'], height=record['height'], sh_degree=3, packed=True,
                rasterize_mode='antialiased')
            image = Image.fromarray((rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            renders[kind].append(image)
        del params, colors
        torch.cuda.empty_cache()
    canvas = Image.new('RGB', (len(kinds) * 240, len(camera_records) * 320), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for row, record in enumerate(camera_records):
        for col, kind in enumerate(kinds):
            canvas.paste(renders[kind][row].resize((240, 295), Image.Resampling.LANCZOS), (col * 240, row * 320 + 25))
            draw.text((col * 240 + 3, row * 320 + 5), f"{kind} {record['name']} {record['shift_sign']:+}", fill='white')
    canvas.save(trial / 'shifted-camera-contact.jpg', quality=92)
    (trial / 'probe_cameras.json').write_text(json.dumps({'cameras': camera_records,
        'scope': '固定平移探针，仅目视观察，无新视角真值，不报告几何精度'}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
