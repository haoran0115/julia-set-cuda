import numpy as np
import matplotlib.pyplot as plt

dpi = 1000
# load information
DIM, theta, x_min, x_max, y_min, y_max = np.loadtxt("coord.dat")
DIM = int(DIM)

# load data
print("Loading data...")
fractal = np.fromfile("fractal.dat")
fractal = fractal.reshape([DIM, DIM])

# figure plotting
print("Plotting figure...")
plt.imshow(fractal, cmap=plt.cm.hot, extent=(x_min, x_max, y_min, y_max))
plt.title("Julia set for $f(Z) = Z^2 + e^{i\\theta}" + ",\\theta = {:2.3f}$".format(theta))
plt.tight_layout()
filename = "julia_DIM{}_theta{}_dpi{}.png".format(DIM, theta, dpi)
print("Saving figure as {}...".format(filename))
plt.savefig("julia_DIM{}_theta{}_dpi{}.png".format(DIM, theta, dpi), dpi=dpi, origin='lower')
print("Done.")
