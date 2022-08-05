import random
import numpy as np
import cv2
import torch


# given 2D-3D correspondences we sample 4 points either using GNN or randomly
def sample_hypothesis(sceneCoordinates, sampling, camMat, out=None, max_hypothesis_tries=1000000, inlierThreshold=10, test=False):
    
    imH = sceneCoordinates.size(2)
    imW = sceneCoordinates.size(3)
    sceneCoordinates = sceneCoordinates.detach().clone().cpu().numpy()[0]
    sceneCoordinates = sceneCoordinates.reshape(3, -1).T
    sampling = sampling.reshape(2, -1).T
    for i in range(max_hypothesis_tries):
        imgPts = []
        objPts = []
        indices = []
        if out is None:
        	# sample randomly
            for j in range(4):
                indx = random.randint(0, (imW * imH) - 1)
                imgPts.append(sampling[indx, :].tolist())
                objPts.append(sceneCoordinates[indx, :].tolist())
                indices.append(indx)

        else:
            # sample according to graph NN
            a = torch.exp(out.clone().flatten().detach())
            indx = torch.multinomial(a, 4).numpy().tolist()
            for j in range(len(indx)):
                imgPts.append(sampling[indx[j], :].tolist())
                objPts.append(sceneCoordinates[indx[j], :].tolist())
                indices.append(indx[j])

        imgPts = np.array(imgPts, dtype='double')
        objPts = np.array(objPts, dtype='double')
        if len(imgPts) <= 4:
            success, rotation_vector, translation_vector = cv2.solvePnP(objPts, 
                                                                        imgPts, 
                                                                        camMat, 
                                                                        distCoeffs=None,
                                                                        useExtrinsicGuess=False,
                                                                        flags=cv2.SOLVEPNP_P3P)
        else:
            success, rotation_vector, translation_vector = cv2.solvePnP(objPts, 
                                                                        imgPts, 
                                                                        camMat, 
                                                                        distCoeffs=None,
                                                                        useExtrinsicGuess=False,
                                                                        flags=0)



        if not success:
            continue

        point2D, _ = cv2.projectPoints(objPts, 
                                       rotation_vector, 
                                       translation_vector, 
                                       camMat, 
                                       None)
    
        point2D = point2D.reshape(point2D.shape[0], -1)
        if ((np.sum(np.abs(imgPts - point2D)**2, axis=-1) ** (1./2)) < inlierThreshold).all():
            break
            
    return imgPts, objPts, rotation_vector, translation_vector, indices

# returns reprojection errors given the hypothesis
def getReproErrs(sceneCoordinates, 
                 rotation_vector, 
                 translation_vector,
                 sampling,
                 camMat, 
                 maxReproj=100):
    points2D = []
    points3D = []
    for y in range(sampling.shape[1]):
        for x in range(sampling.shape[2]):
            points2D.append(sampling[:, y, x].tolist())
            points3D.append(sceneCoordinates[0, :, y, x].cpu().detach().numpy().tolist())
    
    points2D = np.array(points2D, dtype='double')
    points3D = np.array(points3D, dtype='double')
    projections, _ = cv2.projectPoints(points3D, 
                                       rotation_vector, 
                                       translation_vector, 
                                       camMat, 
                                       None)
    
    projections = projections.reshape(projections.shape[0], -1)
    errors = (np.sum(np.abs(points2D - projections)**2, axis=-1)) ** (1./2)
    errors = errors.reshape(sampling.shape[1], sampling.shape[2])
    errors = np.clip(errors, 0, maxReproj)
    return errors

# soft hypothesis score
def getHypScore(reproErrs, inlierThreshold=10, inlierAlpha=100):
    inlierBeta = 5 / inlierThreshold
    softThreshold = inlierBeta * (reproErrs - inlierThreshold)
    softThreshold = 1 / (1 + np.exp(-softThreshold))
    softThreshold = 1 - softThreshold
    score = np.sum(softThreshold)
    return score * (inlierAlpha / reproErrs.shape[1] / reproErrs.shape[0])

# hypothesis refinement
def refineHyp(sceneCoordinates,
              reproErrs,
              sampling,
              camMat,
              rotation_vector,
              translation_vector,
              inlierThreshold=10,
              maxRefSteps=100,
              maxReproj=100):
    
    localReproErrs = reproErrs.copy()
    bestInliers = 4
    
    for i in range(maxRefSteps):
        localImgPts = []
        localObjPts = []
        localInlierMap = np.zeros_like(localReproErrs)
        
        for x in range(sampling.shape[2]):
            for y in range(sampling.shape[1]):
                if localReproErrs[y, x] < inlierThreshold:
                    localImgPts.append(sampling[:, y, x].tolist())
                    localObjPts.append(sceneCoordinates[0, :, y, x].cpu().detach().numpy().tolist())
                    localInlierMap[y, x] = 1
        
        if len(localImgPts) <= bestInliers:
            break
        
        bestInliers = len(localImgPts)
        
        localImgPts = np.array(localImgPts, dtype='double')
        localObjPts = np.array(localObjPts, dtype='double')
        
        rotation_update = rotation_vector.copy()
        translation_update = translation_vector.copy()
        
        if localImgPts.shape[0] > 4:
            success, rotation_update, translation_update = cv2.solvePnP(localObjPts, 
                                                                        localImgPts, 
                                                                        camMat, 
                                                                        distCoeffs=None,
                                                                        useExtrinsicGuess=True,
                                                                        rvec=rotation_update,
                                                                        tvec=translation_update,
                                                                        flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            success, rotation_update, translation_update = cv2.solvePnP(localObjPts, 
                                                                        localImgPts, 
                                                                        camMat, 
                                                                        distCoeffs=None,
                                                                        useExtrinsicGuess=True,
                                                                        rvec=rotation_update,
                                                                        tvec=translation_update,
                                                                        flags=cv2.SOLVEPNP_P3P)
        if not success:
            break
        
        inlierMap = localInlierMap
        rotation_vector = rotation_update
        translation_vector = translation_update
        
        localReproErrs = getReproErrs(sceneCoordinates, 
                                      rotation_vector, 
                                      translation_vector,
                                      sampling,
                                      camMat
                                     )
    
        
    return rotation_vector, translation_vector, localReproErrs