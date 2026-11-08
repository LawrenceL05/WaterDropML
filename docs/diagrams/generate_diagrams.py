"""Generate matching SVG and editable Excalidraw diagrams (Python stdlib only).

Run from any directory: python docs/diagrams/generate_diagrams.py
Short labels and the blue/green/orange palette follow the companion
topology-aware-gpu-scheduling repository. Details belong in the README.
"""
from pathlib import Path
from html import escape
import json
import math

ROOT = Path(__file__).resolve().parent
INK = '#222222'
BLUE = ('#e8eefb', '#4263a5')
GREEN = ('#e8f5e9', '#28823b')
ORANGE = ('#fff3e0', '#c77800')


class Diagram:
    def __init__(self, name, title, description, height, width=900):
        self.name, self.width, self.height = name, width, height
        self.elements = []
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                    f'<title id="title">{escape(title)}</title>',
                    f'<desc id="desc">{escape(description)}</desc>',
                    f'<rect width="{width}" height="{height}" fill="white"/>']
        self.text(28, 20, title, 28)

    def element(self, kind, x, y, width, height, **extra):
        n = len(self.elements) + 1
        item = dict(id=f'{self.name}-{n}', type=kind, x=x, y=y,
                    width=width, height=height, angle=0, strokeColor=INK,
                    backgroundColor='transparent', fillStyle='solid',
                    strokeWidth=2, strokeStyle='solid', roughness=0,
                    opacity=100, groupIds=[], frameId=None, roundness=None,
                    seed=n, version=1, versionNonce=n, isDeleted=False,
                    boundElements=None, updated=1, link=None, locked=False)
        item.update(extra)
        self.elements.append(item)
        return item

    def text(self, x, y, content, size=22, color=INK):
        lines = content.split('\n')
        # Conservative width for editable text; SVG uses the same Arial face.
        width = max(len(line) for line in lines) * size * .57
        self.element('text', x, y, width, len(lines)*size*1.3,
                     text=content, originalText=content, fontSize=size,
                     fontFamily=2, textAlign='left', verticalAlign='top',
                     containerId=None, autoResize=True, lineHeight=1.3,
                     strokeColor=color)
        for i, line in enumerate(lines):
            self.svg.append(f'<text x="{x}" y="{y + size + i*size*1.3}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}">{escape(line)}</text>')

    def box(self, x, y, w, h, label, palette=BLUE, size=22):
        fill, stroke = palette
        self.element('rectangle', x, y, w, h, backgroundColor=fill, strokeColor=stroke)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        self.text(x+16, y+(h-len(label.split('\n'))*size*1.3)/2, label, size)

    def line(self, points, color='#64748b', arrow=False, dash=False, fill='none', width=2):
        x, y = points[0]
        xs, ys = zip(*points)
        self.element('arrow' if arrow else 'line', x, y, max(xs)-min(xs), max(ys)-min(ys),
                     points=[[px-x, py-y] for px, py in points],
                     strokeColor=color, strokeWidth=width,
                     strokeStyle='dashed' if dash else 'solid',
                     backgroundColor='transparent' if fill == 'none' else fill,
                     startBinding=None, endBinding=None, startArrowhead=None,
                     endArrowhead='arrow' if arrow else None)
        coords = ' '.join(f'{px:.2f},{py:.2f}' for px, py in points)
        dashed = ' stroke-dasharray="6 5"' if dash else ''
        self.svg.append(f'<polyline points="{coords}" fill="{fill}" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dashed}/>')
        if arrow:
            a, b = points[-2], points[-1]
            t = math.atan2(b[1]-a[1], b[0]-a[0])
            wings = [(b[0]-10*math.cos(t-s), b[1]-10*math.sin(t-s)) for s in (.5, -.5)]
            self.svg.append(f'<polyline points="{wings[0][0]:.2f},{wings[0][1]:.2f} {b[0]},{b[1]} {wings[1][0]:.2f},{wings[1][1]:.2f}" fill="none" stroke="{color}" stroke-width="2"/>')

    def save(self):
        (ROOT / f'{self.name}.svg').write_text('\n'.join(self.svg + ['</svg>'])+'\n', encoding='utf-8')
        doc = dict(type='excalidraw', version=2, source='https://excalidraw.com',
                   elements=self.elements, appState=dict(viewBackgroundColor='#ffffff', gridSize=None), files={})
        (ROOT / f'{self.name}.excalidraw').write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def workflow():
    d = Diagram('01-measurement-workflow', 'From image to contact angle',
                'Current classical pipeline. Align the image, segment the droplet, fit circle and side tangents, then save and review.', 470)
    d.box(28, 85, 240, 88, 'Microscope image\nRotate upright')
    d.box(330, 85, 240, 88, 'Find substrate\nCorrect tilt')
    d.box(632, 85, 240, 88, 'Trace droplet\nIgnore reflection')
    d.line([(268,129),(322,129)], arrow=True)
    d.line([(570,129),(624,129)], arrow=True)
    d.line([(752,173),(752,203),(246,203),(246,234)], arrow=True)
    d.line([(652,203),(652,234)], arrow=True)
    d.box(76, 242, 340, 80, 'Circle fit\nPrimary angle estimate', GREEN)
    d.box(484, 242, 340, 80, 'Tangent fits\nLeft and right angles', GREEN)
    d.line([(246,322),(246,349),(450,349),(450,373)], arrow=True)
    d.line([(654,322),(654,349),(450,349)])
    d.box(180, 381, 540, 65, 'Save angles, masks and review overlays', GREEN)
    d.save()


def repository():
    d = Diagram('02-repository-map', 'Where files go',
                'Raw images feed a metadata export and the measurement runner. The runner calls contact_angle.py and saves angles, overlays and masks. JayB comparison is a separate step.', 570)
    d.text(28, 72, 'INPUT', 18, BLUE[1])
    d.text(337, 72, 'RUN', 18, BLUE[1])
    d.text(643, 72, 'OUTPUT', 18, GREEN[1])
    d.box(28, 231, 230, 82, 'data/raw/\nDroplet images', size=21)
    d.box(337, 110, 230, 76, 'src/dataset.py\nExport metadata', size=21)
    d.box(643, 110, 229, 76, 'data/index.csv', GREEN, 21)
    d.line([(258,260),(294,260),(294,148),(329,148)], arrow=True)
    d.line([(567,148),(635,148)], arrow=True)
    d.box(337, 231, 230, 82, 'src/run_baseline.py\nCalls contact_angle.py', size=20)
    d.line([(258,272),(329,272)], arrow=True)
    d.box(643, 222, 229, 100, 'outputs/\nbaseline_angles.csv\noverlays/', GREEN, 20)
    d.line([(567,272),(635,272)], arrow=True)
    d.box(643, 352, 229, 76, 'data/masks_auto/\nMasks + upright JPGs', GREEN, 20)
    d.line([(567,293),(606,293),(606,390),(635,390)], arrow=True)
    d.box(28, 471, 350, 72, 'data/reference_jayb/\nReference angles', size=21)
    d.box(487, 471, 385, 72, 'Separate comparison step\noutputs/comparison_vs_jayb.csv', GREEN, 20)
    d.line([(378,507),(479,507)], arrow=True)
    d.line([(872,272),(887,272),(887,451),(680,451),(680,463)], arrow=True)
    d.save()


def filenames():
    d = Diagram('03-filename-key', 'Read an image filename',
                'Example split into date, condition and time code, droplet, magnification and optional shot. All code mappings are in the adjacent README table.', 300)
    d.text(28, 76, '20261015_B3_2_30_1.jpg', 32, BLUE[1])
    labels = [('20261015', 'Date', '15 Oct 2026', 204),
              ('B3', 'Condition / time', 'B; code 3', 188),
              ('2', 'Droplet', 'Droplet 2', 130),
              ('30', 'Magnification', '30×', 162),
              ('1', 'Shot (optional)', 'Shot 1', 160)]
    x = 28
    for value, meaning, detail, width in labels:
        d.box(x, 141, width, 115, f'{value}\n{meaning}\n{detail}', GREEN if value == 'B3' else BLUE, 19)
        x += width + 0
    d.text(28, 270, 'A = untreated   ·   B = UVO-treated   ·   Time code 3 = 1 h', 19)
    d.save()


def geometry():
    d = Diagram('04-contact-angle-and-qc', 'Check the contact angle',
                'Illustrative droplet profile. The contact angle is measured through the liquid between substrate and tangent. Check substrate alignment, outer rim and quality notes before accepting a measurement.', 430)
    base, cx, cy, r = 285, 260, 325, 150
    alpha = math.asin((cy-base)/r)
    profile = [(cx+r*math.cos(t), cy-r*math.sin(t))
               for t in [math.pi-alpha-i*(math.pi-2*alpha)/80 for i in range(81)]]
    d.line(profile+[profile[0]], BLUE[1], fill=BLUE[0], width=3)
    d.line([(28,base),(493,base)], '#00829b', width=3)
    d.text(32, 303, 'Substrate', 21, '#00829b')
    d.text(220, 226, 'Liquid', 22)
    x, y = profile[0]
    theta = math.acos((cy-base)/r)
    d.line([(x,y),(x+95*math.cos(theta),y-95*math.sin(theta))], '#ac343d', width=3)
    d.text(49, 140, 'Tangent', 21, '#ac343d')
    d.line([(111,166),(x+25,base-83)], '#ac343d')
    arc = [(x+46*math.cos(t),base-46*math.sin(t)) for t in [i*theta/30 for i in range(31)]]
    d.line(arc, '#ac343d')
    d.text(x+53, base-49, 'θ', 26, '#ac343d')
    d.text(32, 365, 'θ is measured inside the liquid.', 21)
    d.text(530, 85, 'Before accepting a result', 24, GREEN[1])
    d.text(530, 138, '1. Baseline sits on substrate.\n\n2. Outline follows the outer rim.\n\n3. Check crop and failure notes.', 21)
    d.box(530, 323, 342, 78, 'Use circle angle first.\nReview left/right gaps >15°.', ORANGE, 21)
    d.save()


def unet_training():
    d = Diagram('05-unet-training', 'How WaterDropML trains a U-Net',
                'Reviewed image-mask pairs are prepared and split by droplet. Training predicts a mask, compares it with the label and updates weights. Held-out validation selects the best checkpoint.', 1100, 1000)
    d.text(28, 67, 'PREPARE IMAGE + MASK PAIRS', 18, BLUE[1])
    d.box(28, 105, 280, 115, 'Review labels\nCorrect masks_gt/\nQC: good / fixable / bad', ORANGE, 20)
    d.box(360, 105, 280, 115, 'prepare_data.py\nCrop below baseline\nDownscale to 25%', size=20)
    d.box(692, 105, 280, 115, 'Split by droplet\nTrain and validation\nNever split by shot', size=20)
    d.line([(308, 162), (352, 162)], arrow=True)
    d.line([(640, 162), (684, 162)], arrow=True)
    d.text(28, 240, 'Corrected masks take priority; eligible automatic masks are the fallback.', 20)
    d.text(28, 284, 'TRAINING: REPEAT FOR EACH BATCH', 18, BLUE[1])
    d.line([(832, 220), (832, 320), (168, 320), (168, 337)], arrow=True)
    d.box(28, 345, 280, 125, 'Training pairs\n384 px random crops\nPaired augmentation', size=20)
    d.box(360, 345, 280, 125, 'U-Net predicts mask\nResNet-34 encoder\nDecoder + skip links', size=20)
    d.box(692, 345, 280, 125, 'Compare with label\nDice + BCE loss\nSmaller loss is better', ORANGE, 20)
    d.line([(308, 385), (352, 385)], arrow=True)
    d.line([(640, 385), (684, 385)], arrow=True)
    d.text(314, 352, 'RGB', 16)
    d.text(644, 352, 'logits', 16)
    d.line([(168, 470), (168, 516), (832, 516), (832, 478)], arrow=True)
    d.text(28, 530, 'Target mask: same crop + geometry', 18)
    d.box(360, 570, 280, 100, 'Backpropagation\nAdamW updates weights\nCosine learning rate', ORANGE, 20)
    d.line([(832, 470), (832, 620), (648, 620)], arrow=True)
    d.line([(500, 570), (500, 478)], arrow=True, color=ORANGE[1])
    d.text(28, 573, 'Initial encoder weights:\npretrained on ImageNet\n\nDefault: 60 epochs', 20)
    d.text(28, 712, 'VALIDATE AFTER EACH EPOCH', 18, GREEN[1])
    d.box(28, 760, 420, 120, 'Whole held-out images\nMeasure IoU, Dice and boundary error\nValidation does not update weights', GREEN, 20)
    d.box(552, 760, 420, 120, 'Keep highest validation IoU\nmodels/unet_best.pt\nLog every epoch: train_log.csv', GREEN, 20)
    d.line([(448, 820), (544, 820)], arrow=True)
    d.line([(972, 162), (986, 162), (986, 735), (238, 735), (238, 752)], arrow=True)
    d.line([(640, 440), (662, 440), (662, 699), (410, 699), (410, 752)], arrow=True)
    d.text(676, 681, 'Current model', 18)
    d.box(28, 920, 944, 135, 'Use the saved model: predict.py -> droplet mask -> existing angle fitting\nEvaluate against corrected held-out masks, then correct errors and retrain.\nU-Net replaces segmentation; orientation and baseline detection remain separate.', GREEN, 20)
    # Use Excalifont in the new editable diagram without changing existing exports.
    for element in d.elements:
        if element['type'] == 'text':
            element['fontFamily'] = 5
    d.save()


if __name__ == '__main__':
    workflow()
    repository()
    filenames()
    geometry()
    unet_training()
