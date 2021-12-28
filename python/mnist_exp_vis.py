import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys

fig, ax = plt.subplots(ncols=2)
plt.subplots_adjust(bottom=0.15)
dirname = sys.argv[1]
if dirname[-1] != '/':
    dirname += '/'
exp_filename = dirname + "exps.csv"
prb_filename = dirname + "probs.csv"

exps = np.genfromtxt(exp_filename, delimiter=',', filling_values=0.5, skip_header=0)
n = exps.shape[0]-1
finit = n
probs = np.genfromtxt(prb_filename, delimiter=',', skip_header=0)
if len(probs.shape) == 2:
    probs = probs[:,0]
ax[0].imshow(exps[0].reshape(28,28), cmap='gray', vmin=0, vmax=1)
ax[0].set_title("Actual label: %d, Exp pred: %f" % (int(probs[0]), probs[1]))
img = ax[1].imshow(exps[finit].reshape(28,28), cmap='gray', vmin=0, vmax=1)
ax[1].set_title("Exp pred: %f" % (probs[finit+1]))

nfax = plt.axes([0.2,0.05,0.65,0.03])
nfslider = Slider(nfax, 'num features', valmin=1, valmax=n, valinit=finit, valstep=1)

def slider_update(val):
    img.set_data(exps[int(val)].reshape(28,28))
    ax[1].set_title("Exp pred: %f" % (probs[int(val+1)]))
    fig.canvas.draw_idle()

nfslider.on_changed(slider_update)

plt.show()