from visu.error_visualizer import ErrorVisualizer
import matplotlib
import matplotlib.pyplot as plt

plt.ion()

class VisuWorkFlow:
    def __init__(self, sampling, scene_coords, cam_mat, errors, image, img_name):
        self.gnn_ev = ErrorVisualizer(sampling, scene_coords, cam_mat, errors, image)
        self.random_ev = ErrorVisualizer(sampling, scene_coords, cam_mat, errors, image)
        self.fig, self.axes = plt.subplots(2, 3)
        self.name = img_name

        gs_rot = self.axes[0, 1].get_gridspec()
        gs_trans = self.axes[0, 2].get_gridspec()
        # remove the underlying axes
        for ax in self.axes[0:, 1]:
            ax.remove()
        for ax in self.axes[0:, 2]:
            ax.remove()
        self.rot_ax = self.fig.add_subplot(gs_rot[0:, 1])
        self.trans_ax = self.fig.add_subplot(gs_trans[0:, 2])

    def clear_plots(self):
        self.rot_ax.clear()
        self.trans_ax.clear()

    def draw_error_plot(self, gnn_rot, gnn_trans, random_rot, random_trans):
        self.clear_plots()
        self.rot_ax.plot(gnn_rot, 'r-', label='GNN Ransac')
        self.rot_ax.plot(random_rot, 'b-', label='Vanilla Ransac')
        self.trans_ax.plot(gnn_trans, 'r-', label='GNN Ransac')
        self.trans_ax.plot(random_trans, 'b-', label='Vanilla Ransac')

        self.rot_ax.legend()
        self.trans_ax.legend()

        self.rot_ax.set_title("Rotation Error")
        self.rot_ax.set_xlabel("Epoch")
        self.rot_ax.set_ylabel("Degrees [°]")
        self.rot_ax.set_xlim(left=-0.5, right=10)
        self.rot_ax.set_ylim(bottom=0, top=6)

        self.trans_ax.set_title("Translation Error")
        self.trans_ax.set_xlabel("Epoch")
        self.trans_ax.set_ylabel("Centimeters [cm]")
        self.trans_ax.set_xlim(left=-0.5, right=10)
        self.trans_ax.set_ylim(bottom=0, top=0.06)

    def draw_gnn(self, hyp, name="1", title="1"):
        self.axes[1, 0].clear()
        self.fig.patches.append(matplotlib.patches.Rectangle((0.1, 0.07), 0.27, 0.43, facecolor='r', linewidth=0, zorder=-1,transform=self.fig.transFigure))
        self.gnn_ev.compute_and_draw_errors(hyp, name=name, title=title, ax=self.axes[1,0], c="w")
        self.axes[1, 0].tick_params(axis='x', colors='w')
        self.axes[1, 0].tick_params(axis='y', colors='w')

    def draw_ransac(self, hyp, name="1", title="1"):
        self.axes[0, 0].clear()
        self.fig.patches.append(matplotlib.patches.Rectangle((0.1, 0.5), 0.27, 0.43, facecolor='b', linewidth=0, zorder=-1, transform=self.fig.transFigure))
        self.random_ev.compute_and_draw_errors(hyp, name=name, title=title, ax=self.axes[0,0], c="w")
        self.axes[0, 0].tick_params(axis='x', colors='w')
        self.axes[0, 0].tick_params(axis='y', colors='w')

    def show(self, save=False, name="1"):
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.pause(1)
        if save:
            self.fig.savefig(self.name + "_" + name + ".png", dpi=200)
        plt.draw()
        plt.show()
        plt.pause(1)
        #plt.close()


