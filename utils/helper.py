import numpy as np
from torchvision import transforms
from skimage import io
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import kneighbors_graph
import cv2
import math
import matplotlib.image as mpimg


from models.network import Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# returns all the components needed to train GNN and later compare with the GT pose
def get_required_components(file_name, network):
          
    pose_path = file_name.split("/")
    pose_path[3] = "poses"
    pose_path[4] = pose_path[4][:-9] + "pose.txt" 
    pose_path = "/".join(pose_path)
    image, gt_pose, focal_length = load_image(file_name, pose_path)
    img_visu = mpimg.imread(file_name)
    image = image.to(device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features, scene_coordinates = network(image)
    imH = scene_coordinates.size(2)
    imW = scene_coordinates.size(3)
    sampling = create_sampling(imW, imH, network.OUTPUT_SUBSAMPLE, 0, 0)
    
    camMat = create_cammat(focal_length, float(image.size(3) / 2), float(image.size(2) / 2))
    
    return features, scene_coordinates, sampling, camMat, gt_pose, img_visu

# loads image and return image, gt_pose and focal length
def load_image(image_path, gt_pose_path):
    image_height = 480
    focal_length = 525.0
    image_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(image_height),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4],
                            std=[0.25]
                        )
                    ])
    
    image = io.imread(image_path)
    if len(image.shape) < 3:
        image = color.gray2rgb(image)

    f_scale_factor = image_height / image.shape[0]
    focal_length *= f_scale_factor
    
    pose = np.loadtxt(gt_pose_path)
    pose = torch.from_numpy(pose).float()
    image = image_transform(image)
    
    return image, pose, focal_length

# returns 80 * 60 * 2 2D pixel coordinates from dsac* paper
def create_sampling(outimW, outimH, subSampling, shiftX, shiftY):
    sampling = np.zeros((2, outimH, outimW))
    for x in range(outimW):
        for y in range(outimH):
            sampling[:, y, x] = np.array([x * subSampling + subSampling / 2 - shiftX, 
                                         y * subSampling + subSampling / 2 - shiftY])
            
    return sampling


# loads scene coordinate network from dsac* paper
def load_sc_network(path):

    network = Network(torch.zeros((3)), tiny=False)
    network.load_state_dict(torch.load(path, map_location=device))
    network = network.to(device)
    network.eval()

    return network

# creates camera matrix given focal length cx, and cy
def create_cammat(focal_length, ppointX, ppointY):
	camMat = np.eye(3)
	camMat[0, 0] = focal_length
	camMat[1, 1] = focal_length
	camMat[0, 2] = ppointX
	camMat[1, 2] = ppointY

	return camMat

# returns error with respect to gt_pose given the hypothesis
def get_error(rotation_vector, translation_vector, gt_pose):
    out_pose = pose2trans(rotation_vector, translation_vector)
    t_err = np.linalg.norm(gt_pose[0:3, 3] - out_pose[0:3, 3])

    gt_R = gt_pose[0:3, 0:3].numpy()
    out_R = out_pose[0:3, 0:3]
    r_err = np.matmul(out_R, np.transpose(gt_R))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi
    return r_err, t_err

def normalize_pc(pc_np):
    # Normalize the cloud to unit cube
    # input numpy ndarray -> output numpy ndarray
    center = np.mean(pc_np, axis = 0)
    pc_np_norm = pc_np - center
    distance = np.max(np.sqrt(np.sum(abs(pc_np_norm)**2,axis=-1)))
    pc_np_norm /= distance
    return pc_np_norm

# prepares the GNN input using KNN
def prepare_GNN_input(image, scene_coord, sampling):
    image = image.detach().clone().cpu().numpy()[0]
    
    scene_coord = scene_coord.detach().clone().cpu().numpy()[0]
    scene_coord = scene_coord.reshape(3, -1).T
    sampling = sampling.reshape(2, -1).T
    image = image.reshape(512, -1).T
    sampling_not_norm = sampling.copy()
    
    A = kneighbors_graph(image, 40, mode='distance', include_self=False)
    
    A.data = 1 / A.data
    
    scene_coord = normalize_pc(scene_coord)
    sampling = normalize_pc(sampling)
    
    adj = A.toarray()
    edge_index, edge_weights = from_scipy_sparse_matrix(A)
    gnn_input = torch.tensor(np.concatenate([image, scene_coord, sampling], axis=1), dtype=torch.double, requires_grad=False)
    return gnn_input, edge_index, edge_weights

# samples random 8*8 2D-2D points given 80 * 60
def sample_random_points(features, scene_coord, sampling):
    
    new_features = []
    new_scene_coord = []
    new_sampling = []
    
    col_ind = np.random.choice(np.arange(scene_coord.size(3)), size=8, replace=False)
    row_ind = np.random.choice(np.arange(scene_coord.size(2)), size=8, replace=False)
    
    for i in row_ind:
        temp1 = []
        temp2 = []
        temp3 = []
        for j in col_ind:
            temp1.append(scene_coord[0, :, i, j].detach().cpu().numpy().tolist())
            temp2.append(features[0, :, i, j].detach().cpu().numpy().tolist())
            temp3.append(sampling[:, i, j].tolist())
            
        new_scene_coord.append(temp1)
        new_features.append(temp2)
        new_sampling.append(temp3)
    
    new_scene_coord = np.array(new_scene_coord).transpose(2, 0, 1)
    new_features = np.array(new_features).transpose(2, 0, 1)
    new_sampling = np.array(new_sampling).transpose(2, 0, 1)

    new_scene_coord = torch.from_numpy(new_scene_coord).unsqueeze(0)
    new_features = torch.from_numpy(new_features).unsqueeze(0)
    
    
    return new_features, new_scene_coord, new_sampling

# returns rotation, translation given pose
def trans2pose(trans):
	invTrans = np.linalg.inv(trans)
	rot = cv2.Rodrigues(invTrans[0:3, 0:3])[0]
	trans = invTrans[0:3, 3]
	return (rot, trans)

# returns pose given rotation, translation
def pose2trans(rotation_vector, translation_vector):
    trans = np.eye(4)
    rot = cv2.Rodrigues(rotation_vector)[0]
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = translation_vector.flatten()
    out_pose = np.linalg.inv(trans)
    return out_pose
