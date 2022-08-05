import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from utils.hypothesis_helper import getReproErrs
import numpy as np



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

bp = [0.0, 0.1, 0.6, 1.0]
cdict = {'red':   [[bp[0],  0.0, 0.0],
                   [bp[1],  0.01, 0.01],
                   [bp[2],  1.0, 1.0],
                   [bp[3],  1.0, 1.0]],
         'green': [[bp[0],  1.0, 1.0],
                   [bp[1],  0.2, 0.2],
                   [bp[2],  0.64, 0.64],
                   [bp[3],  0.0, 0.0]],
         'blue':  [[bp[0],  0.0, 0.0],
                   [bp[1],  0.0, 0.0],
                   [bp[2],  0.0, 0.0],
                   [bp[3],  0.0, 0.0]]}

newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

class ErrorVisualizer:
    def __init__(self, sampling, scene_coords, cam_mat, errors, image):
        self.sampling = sampling
        self.scene_coords = scene_coords
        self.errors = errors
        self.image = image
        self.cam_mat = cam_mat

    def draw_errors(self, reproj_errors, name="1", title="1", save=False, ax=None, c="r"):
        sampling_res = self.sampling.reshape((2, 4800)).T
        errors_res = reproj_errors.reshape((4800,))
        if ax is None:
            fig = plt.imshow(rgb2gray(self.image), cmap=plt.get_cmap('gray'))
            plt.scatter(sampling_res[:, 0], sampling_res[:, 1], s=2, c=errors_res, cmap=newcmp)
            plt.title(title)
            if save:
                plt.savefig(name + ".png", dpi=300)
            else:
                plt.show()
            plt.close()
        else:
            ax.imshow(rgb2gray(self.image), cmap=plt.get_cmap('gray'))
            ax.scatter(sampling_res[:, 0], sampling_res[:, 1], s=2, c=errors_res, cmap=newcmp)
            ax.set_title(title, c=c, fontweight='bold')



    def compute_and_draw_errors(self, hyp, name="1", title="1", ax=None, c="r"):
        reproj_errors = getReproErrs(self.scene_coords,
                                     hyp[0],
                                     hyp[1],
                                     self.sampling,
                                     self.cam_mat,
                                     maxReproj=100)

        self.draw_errors(reproj_errors=reproj_errors, name=name, title=title, ax=ax, c=c)