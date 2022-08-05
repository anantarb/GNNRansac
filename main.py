import torch.optim as optim
import torch
import statistics
import argparse

from models.gnn import GNNSample

from utils.hypothesis_helper import sample_hypothesis
from utils.hypothesis_helper import getReproErrs
from utils.hypothesis_helper import getHypScore
from utils.hypothesis_helper import refineHyp


from utils.helper import get_error
from utils.helper import prepare_GNN_input
from utils.helper import sample_random_points
from utils.helper import get_required_components
from utils.helper import load_sc_network

from utils.utils import recursive_glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()

parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
	help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
	help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, 
	help='learning rate')

parser.add_argument('--trainiterations', '-ti', type=int, default=10, 
	help='number of training iterations (= parameter updates)')

parser.add_argument('--samplesize', '-ss', type=int, default=10, 
	help='number of runs to approximate the expectation when training GNN')

parser.add_argument('--scene', '-s', type=str, default='chess', 
	help='select one scene out of 7 scenes')

parser.add_argument('--split', '-sp', type=str, default='test', 
	help='GNN to run on either test or train set of that scene')

parser.add_argument('--seq', '-sq', type=str, default='test', 
	help='GNN to run on either test or train set of that scene')

parser.add_argument('--verbose', '-v', type=bool, default=False, 
	help='Log everything or not')

opt = parser.parse_args()

# update graph NN based on sampled score to maximize expected score
def update_network(gcn_nn, gnn_input, edge_index, edge_weights, sampled_scores, sampled_indices):
	gnn_optimizer = optim.Adam(gcn_nn.parameters(), lr=0.001)
	gcn_nn.train()
	log_prob = gcn_nn(gnn_input, edge_index, edge_weights)
	avg_score = sum(sampled_scores) / len(sampled_indices)
	gradient = []
	gradients = torch.zeros(log_prob.size())

	for k in range(len(sampled_indices)):
	    gradients = torch.zeros(log_prob.size(), dtype=torch.double)
	    for l in range(len(sampled_indices[k])):
	        gradients[sampled_indices[k][l], 0] += 1
	    gradient.append(gradients)
	    
	for i, score in enumerate(sampled_scores):
	    gradients += gradient[i] * (avg_score - score)
	    
	    
	gradients /= len(sampled_scores)

	torch.autograd.backward((log_prob), (gradients))
	gnn_optimizer.step()
	gnn_optimizer.zero_grad()

# select the best 8 * 8 points and use GNN to find the best inlier in them
def apply_optimization(gcn_nn, hyp):
	gnn_input, edge_index, edge_weights = prepare_GNN_input(hyp[0], hyp[1], hyp[2])
	best_score = 0
	best_hyp = None
	if opt.verbose:
		print("Inlier Score Before Optmization: ", hyp[4])
	for epoch in range(opt.trainiterations):
		sampled_indices = []
		sampled_scores = []
		random_score = []
		log_prob = gcn_nn(gnn_input, edge_index, edge_weights).detach()
		for i in range(opt.samplesize):

			# sometimes sampled hypothesis will output None rotation vector so keep in try block
			_, _, rotation_vector, translation_vector, indices = sample_hypothesis(hyp[1], 
			                                                                 hyp[2], 
			                                                                 hyp[3],
			                                                                 out=log_prob,
			                                                                 max_hypothesis_tries=100000, 
			                                                                 inlierThreshold=10 
			                                                                )

			if rotation_vector is None:
			    sampled_scores.append(-100)
			    sampled_indices.append(indices)
			    continue

			try:
				reproj_errors = getReproErrs(hyp[1],
				                             rotation_vector, 
				                             translation_vector, 
				                             hyp[2], 
				                             hyp[3], 
				                             maxReproj=100)
			except:
				continue
			score = getHypScore(reproj_errors)
			if score > best_score:
			    best_score = score
			    best_hyp = (rotation_vector, translation_vector)
			sampled_scores.append(score)
			sampled_indices.append(indices)

			if opt.verbose:

				_, _, rotation_vector, translation_vector, indices = sample_hypothesis(hyp[1], 
				                                                                 hyp[2], 
				                                                                 hyp[3],
				                                                                 out=None,
				                                                                 max_hypothesis_tries=100000, 
				                                                                 inlierThreshold=10 
				                                                                )

				try:
					reproj_errors = getReproErrs(hyp[1],
					                             rotation_vector, 
					                             translation_vector, 
					                             hyp[2], 
					                             hyp[3], 
					                             maxReproj=100)
				except:
					continue
				rand_score = getHypScore(reproj_errors)
				random_score.append(rand_score)


			if opt.verbose:
				print(f"Epoch {epoch} Expected Random Score: {sum(random_score)/ len(random_score)}, Expected GNN Score: {sum(sampled_scores)/ len(sampled_scores)}")
			update_network(gcn_nn, gnn_input, edge_index, edge_weights, sampled_scores, sampled_indices)
	if opt.verbose:        
		print("Inlier Score After Optmization: ", best_score)
	return best_hyp


files = recursive_glob(f"datasets/7scenes_{opt.scene}/{opt.split}", ".png")
sc_network = load_sc_network(f"saved_models/rgb/7scenes/7scenes_{opt.scene}.net")
rot_err = []
t_err = []
# loop through the images and run GNN on all of them
for f in files:
	best_score = 0
	best_hyp = None
	gcn_nn = GNNSample().double()
	features, scene_coordinates, sampling, camMat, gt_pose = get_required_components(f, sc_network)
	for i in range(opt.hypotheses):
		# sample 8 * 8
		_features, _scene_coord, _sampling = sample_random_points(features, scene_coordinates, sampling)
		_, _, rotation_vector, translation_vector, indices = sample_hypothesis(_scene_coord, 
                                                                               _sampling, 
                                                                               camMat,
                                                                               out=None,
                                                                               inlierThreshold=opt.threshold 
                                                                              )
		if rotation_vector is None:
		    continue

		reproj_errors = getReproErrs(_scene_coord,
		                             rotation_vector, 
		                             translation_vector, 
		                             _sampling, 
		                             camMat, 
		                             maxReproj=opt.maxpixelerror)
		score = getHypScore(reproj_errors)
		if score > best_score:
			best_score = score
			best_hyp = (_features, _scene_coord, _sampling, camMat, score, indices)

	updated_hyp = apply_optimization(gcn_nn, best_hyp)
	try:
		reproj_errors = getReproErrs(scene_coordinates,
		                                 updated_hyp[0], 
		                                 updated_hyp[1], 
		                                 sampling, 
		                                 camMat, 
		                                 maxReproj=100)

		rotation_vector, translation_vector, reproErrs = refineHyp(scene_coordinates,
		                                                        reproj_errors,
		                                                        sampling,
		                                                        camMat,
		                                                        updated_hyp[0],
		                                                        updated_hyp[1]
		                                                       )
	except:
	    continue
	r_err, trans_err = get_error(rotation_vector, translation_vector, gt_pose)
	if opt.verbose:
		print(f"Rotation Error for {f}: ", r_err)
		print(f"Translation Error for {f}: ", trans_err)
	rot_err.append(r_err)
	t_err.append(trans_err)
	
print("Median Rotation Error: ", statistics.median(rot_err))
print("Median Translation Error:", statistics.median(t_err))