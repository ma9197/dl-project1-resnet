"""Build results.ipynb (then execute it with nbconvert so all outputs are saved in the file)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# Project 1: ResNet under 5M parameters on CIFAR-10\n"
       "Mirsaid Abbasov (ma9197), NYU Deep Learning, Fall 2026. Solo project.\n\n"
       "This notebook loads the trained models and logs produced by `train.py`, prints the parameter count and the "
       "final test accuracy, plots the training curves, compares the experiments, and writes the Kaggle submission.\n\n"
       "Reproduce the final model (about 32 min on one RTX 4090 Laptop GPU):\n"
       "```\npython train.py --name final_B_ta_e200 --channels 84 168 336 --blocks 2 2 2 --epochs 200 --trivialaug --full-train\n"
       "python predict.py --run final_B_ta_e200\n```"),
    code("import json, csv\nfrom pathlib import Path\nimport numpy as np, torch\nimport matplotlib.pyplot as plt\n"
         "from torchsummary import summary\nfrom train import ResNet, count_params, load_cifar, evaluate\n"
         "FINAL = 'final_B_ta_e200'\nprint(torch.__version__, torch.cuda.get_device_name())"),
    md("## Final model: architecture and number of parameters"),
    code("ck = torch.load(f'runs/{FINAL}/model.pt', map_location='cuda', weights_only=False)\na = ck['args']\n"
         "model = ResNet(a['channels'], a['blocks'], a['kernel'], a['skip_kernel']).cuda()\n"
         "model.load_state_dict(ck['model']); model.eval()\n"
         "summary(model, (3, 32, 32))\n"
         "n = count_params(model)\nprint(f'Trainable parameters: {n:,}  (limit 5,000,000)')\nassert n <= 5_000_000"),
    md("## Final test accuracy (standard CIFAR-10 test set, 10,000 images)"),
    code("xtr, ytr, xte, yte = load_cifar('cuda')\n"
         "loss, acc = evaluate(model, xte, yte)\n_, acc_tta = evaluate(model, xte, yte, tta=True)\n"
         "print(f'CIFAR-10 test accuracy: {100*acc:.2f}%   (with horizontal-flip TTA: {100*acc_tta:.2f}%)')"),
    md("## Training curves (200 epochs, with and without TrivialAugment)"),
    code("fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
         "for run, lab in [('final_B_ta_e200', 'TrivialAugment + crop/flip/Cutout'), ('final_B_base_e200', 'crop/flip/Cutout only')]:\n"
         "    log = json.loads(Path(f'runs/{run}/log.json').read_text())\n"
         "    ep = [r['epoch'] for r in log]\n"
         "    ax[0].plot(ep, [r['train_loss'] for r in log], label=lab)\n"
         "    ax[1].plot(ep, [100*r['val_acc'] for r in log], label=lab)\n"
         "ax[0].set(xlabel='epoch', ylabel='training loss (label smoothing 0.1)', title='Training loss')\n"
         "ax[1].set(xlabel='epoch', ylabel='CIFAR-10 test accuracy (%)', title='Test accuracy', ylim=(80, 98))\n"
         "for x in ax: x.grid(alpha=0.3); x.legend()\nplt.tight_layout(); plt.savefig('figures/curves.png', dpi=160); plt.show()"),
    md("## All experiments"),
    code("rows = []\nfor s in sorted(Path('runs').glob('*/summary.json')):\n    d = json.loads(s.read_text())\n"
         "    rows.append((d['name'], d['params'], d['channels'], d['blocks'], d['epochs'], d['args'].get('trivialaug', False),\n"
         "                 100*d['cifar10_test_acc'], 100*d['cifar10_test_acc_flipTTA'], d['minutes']))\n"
         "print(f\"{'run':20s} {'params':>10s} {'channels':16s} {'blocks':10s} {'ep':>4s} {'TA':>5s} {'test':>7s} {'+TTA':>7s} {'min':>5s}\")\n"
         "for r in rows:\n    print(f'{r[0]:20s} {r[1]:>10,} {str(r[2]):16s} {str(r[3]):10s} {r[4]:>4d} {str(r[5]):>5s} {r[6]:6.2f}% {r[7]:6.2f}% {r[8]:5.1f}')"),
    md("## Per-class accuracy of the final model"),
    code("with torch.no_grad():\n    from train import normalize\n    preds = []\n    for k in range(0, 10000, 1000):\n"
         "        xb = normalize(xte[k:k+1000])\n        with torch.autocast('cuda', dtype=torch.bfloat16):\n"
         "            o = model(xb).float().softmax(1) + model(xb.flip(3)).float().softmax(1)\n        preds.append(o.argmax(1))\n"
         "preds = torch.cat(preds).cpu().numpy(); y = yte.cpu().numpy()\n"
         "classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']\n"
         "cm = np.zeros((10, 10), int)\nfor t, p in zip(y, preds): cm[t, p] += 1\n"
         "for i, c in enumerate(classes): print(f'{c:11s} {100*cm[i,i]/cm[i].sum():6.2f}%')\n"
         "plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap='Blues'); plt.colorbar()\n"
         "plt.xticks(range(10), classes, rotation=60); plt.yticks(range(10), classes)\n"
         "plt.xlabel('predicted'); plt.ylabel('true'); plt.title('Confusion matrix (CIFAR-10 test)')\n"
         "plt.tight_layout(); plt.savefig('figures/confusion.png', dpi=160); plt.show()"),
    md("## Kaggle submission (2,000 held-out images, labels hidden)"),
    code("import subprocess, sys\nprint(subprocess.run([sys.executable, 'predict.py', '--run', FINAL], capture_output=True, text=True).stdout)\n"
         "sub = list(csv.DictReader(open(f'submissions/{FINAL}.csv')))\nprint(len(sub), 'rows; first rows:', sub[:3])"),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "results.ipynb")
print("wrote results.ipynb")
