import argparse
from line_dataset import LineDataset
from line_nn import LineNN
from line_loss import LineLoss
from gnn_dsac import GNNDSAC
import torch.optim as optim
import torch
import os
from visualization_helper import draw_models, draw_wpoints
import cv2
import numpy as np
from skimage.io import imsave


parser = argparse.ArgumentParser()

parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
	help='number of line hypotheses sampled for each image')

parser.add_argument('--inlierthreshold', '-it', type=float, default=10, 
	help='threshold used in the soft inlier count. Its measured in relative image size (1 = image width)')

parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, 
	help='learning rate')

parser.add_argument('--trainiterations', '-ti', type=int, default=10, 
	help='number of training iterations (= parameter updates)')

parser.add_argument('--samplesize', '-ss', type=int, default=5, 
	help='number of runs to approximate the expectation when training GNN')

opt = parser.parse_args()



# first train the point NN from DSAC
def prepare_data(inputs, labels):

    inputs = torch.from_numpy(inputs)
    labels = torch.from_numpy(labels)
    inputs.transpose_(1,3).transpose_(2, 3)
    inputs = inputs - 0.5

    return inputs, labels

def draw_viz(dsac, points, log_probs, labels, iteration):
	dsac(points, log_probs, labels)
	score = dsac.batch_inliers[0].sum() / points.shape[2]
	image_src = inputs.cpu().permute(0,2,3,1).numpy() #Torch to Numpy
	viz_probs = image_src.copy() * 0.2 # make a faint copy of the input image
	if score > 0.4:
	    image_src = draw_models(dsac.est_parameters, clr=(0,0,1), data=image_src)

	viz = [image_src]
	viz_score = viz_probs.copy()
	viz_probs = draw_models(dsac.est_parameters, clr=(0.3,0.3,0.3), data=viz_probs)
	viz_inliers = viz_probs.copy()
	viz_probs = draw_wpoints(points, viz_probs, weights=torch.exp(log_probs), clrmap=cv2.COLORMAP_PLASMA)
	viz_inliers = draw_wpoints(points, viz_inliers, weights=dsac.batch_inliers, clrmap=cv2.COLORMAP_WINTER)
	color_map = np.arange(256).astype('u1')
	color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_HSV)
	color_map = color_map[:,:,::-1]

	score = int(score*100) #using only the first portion of HSV to get a nice (red, yellow, green) gradient
	clr = color_map[score, 0] / 255

	viz_score = draw_models(dsac.est_parameters, clr=clr, data=viz_score)

	viz = viz + [viz_probs, viz_inliers, viz_score]

	viz = np.concatenate(viz, axis=2)

	if not os.path.isdir('viz'):
		os.makedirs('viz')

	imsave(f'viz/visualization_{iteration}.png', viz[0])


dataset = LineDataset(64, 64)
point_nn = LineNN(4, 65)
loss = LineLoss(64)
dsac = GNNDSAC(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, loss)
opt_point_nn = optim.Adam(point_nn.parameters(), lr=opt.learningrate)
images, labels = dataset.sample_lines(1)
inputs, labels = prepare_data(images, labels)
for iteration in range(100):
    point_nn.train()
    point_prediction, log_prob = point_nn(inputs)
    log_prob.fill_(1/log_prob.size(1))
    log_prob = torch.log(log_prob)
    
    g_points = torch.zeros(point_prediction.size())

    exp_loss = 0 
    losses = [] 
    for s in range(1):
        cur_loss, _ = dsac(point_prediction, log_prob, labels)
        g_points += torch.autograd.grad(cur_loss, point_prediction)[0]
        exp_loss += cur_loss
        losses.append(cur_loss)
    
    g_points /= 1
    exp_loss /= 1
    
    torch.autograd.backward((point_prediction), (g_points))
    opt_point_nn.step()
    opt_point_nn.zero_grad()
    print('Iteration: %6d, DSAC Exp. Loss: %2.2f' % (iteration, exp_loss), flush=True)
    del exp_loss, point_prediction, log_prob, g_points, losses

# Now train graphNN to learn good samples

# initial point weigths visualization before training GNN
point_nn.eval()
with torch.no_grad():
	points, log_probs = point_nn(inputs)
draw_viz(dsac, points, log_probs, labels, 0)
for iteration in range(opt.trainiterations):
	point_nn.train()
	point_prediction, log_prob = point_nn(inputs)
	point_prediction = point_prediction.detach()

	g_log_probs = torch.zeros(log_prob.size())
	g_points = torch.zeros(point_prediction.size())

	exp_score = 0 
	scores = [] 
	sample_grads = []
	for s in range(opt.samplesize):
	    _, hyp_scores = dsac(point_prediction, log_prob, labels)
	    sample_grads.append(dsac.g_log_probs)
	    scores.append(hyp_scores.detach())
	    exp_score += hyp_scores.detach().mean()

	g_points /= opt.samplesize
	exp_score /= opt.samplesize
	for i, l in enumerate(scores):
	    for j in range(l.size(0)):
	        g_log_probs += sample_grads[i][0][j].unsqueeze(0) * (exp_score - l[j])

	torch.autograd.backward((log_prob), (g_log_probs))
	opt_point_nn.step()
	opt_point_nn.zero_grad()
	print('Iteration: %6d, GNN Exp. Score: %2.2f' % (iteration, exp_score), flush=True)
	del point_prediction, log_prob, g_log_probs, g_points, sample_grads, exp_score
	point_nn.eval()
	with torch.no_grad():
	    points, log_probs = point_nn(inputs)
	draw_viz(dsac, points, log_probs, labels, iteration+1)


