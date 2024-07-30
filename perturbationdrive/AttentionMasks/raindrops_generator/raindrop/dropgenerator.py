import random
from random import randint
import cv2
import math
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageDraw
from skimage.measure import label as skimage_label
from .raindrop import Raindrop, make_bezier
from .snowflake import SnowFlake

"""
This module contain two functions:
Check Collision -- handle the collision of the drops
generateDrops -- generate raindrops on the image

Author: Chia-Tse, Chang
Edited by Vera, Soboleva
"""


def CheckCollision(DropList):
	"""
	This function handle the collision of the drops
	:param DropList: list of raindrop class objects 
	"""
	listFinalDrops = []
	Checked_list = []
	list_len = len(DropList)
	# because latter raindrops in raindrop list should has more colision information
	# so reverse list	
	DropList.reverse()
	drop_key = 1
	for drop in DropList:
		# if the drop has not been handle	
		if drop.getKey() not in Checked_list:			
			# if drop has collision with other drops
			if drop.getIfColli():
				# get collision list
				collision_list = drop.getCollisionList()
				# first get radius and center to decide how  will the collision do
				final_x = drop.getCenters()[0] * drop.getRadius()
				final_y = drop.getCenters()[1]  * drop.getRadius()
				tmp_devide = drop.getRadius()
				final_R = drop.getRadius()  * drop.getRadius()
				for col_id in collision_list:
					col_id = int(col_id)
					Checked_list.append(col_id)
					# list start from 0
					final_x += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[0]
					final_y += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[1]
					tmp_devide += DropList[list_len - col_id].getRadius()
					final_R += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getRadius() 
				final_x = int(round(final_x/tmp_devide))
				final_y = int(round(final_y/tmp_devide))
				final_R = int(round(math.sqrt(final_R)))
				# rebuild drop after handled the collisions
				newDrop = Raindrop(drop_key, (final_x, final_y), final_R)
				drop_key = drop_key+1
				listFinalDrops.append(newDrop)
			# no collision
			else:
				drop.setKey(drop_key)
				drop_key = drop_key+1
				listFinalDrops.append(drop)
	

	return listFinalDrops



def generate_label(h, w, chosen_pos, cfg, prev_shape=None, prev_size=None,type_drop=None):
    """
    This function generates a list of raindrop class objects and a label map of these drops in the image.
    :param h: image height
    :param w: image width
    :param cfg: config with global constants
    :param chosen_pos: positions for the raindrops
    :param prev_shape: previous shapes of the raindrops
    :param prev_size: previous sizes of the raindrops
    :return: list of final raindrops, list of shapes, list of sizes
    """
    maxDrop = cfg["maxDrops"]
    minDrop = cfg["minDrops"]
    maxR = cfg["maxR"]
    minR = cfg["minR"]
    drop_num = random.randint(minDrop, maxDrop)
    imgh = h + 20
    imgw = w + 20

    # Random drops position
    if chosen_pos is None:
        ran_pos = [(int(random.random() * imgw), int(random.random() * imgh)) for _ in range(drop_num)]
    else:
        ran_pos = chosen_pos

    listRainDrops = []
    lisShapes = []
    lisSizes = []
    
    for key, pos in enumerate(ran_pos):
        if prev_size is None:
            radius = random.randint(minR, maxR)
        else:
            radius = prev_size[key]
        if prev_shape is None:
            shape = random.randint(0, 2)
        else:
            shape = prev_shape[key]
        lisShapes.append(shape)
        lisSizes.append(radius)
        key = key + 1
        if type_drop=="snowflake":
            drop = SnowFlake(key, pos, radius, shape)
        else:
            drop = Raindrop(key, pos, radius, shape)
        listRainDrops.append(drop)

    # Initialize label map and collision checks
    label_map = np.zeros([h + 20, w + 20])
    listFinalDrops = list(listRainDrops)
    collisionNum = len(listFinalDrops)
    loop = 0

    # while collisionNum > 0:
    #     loop += 1
    #     collisionNum = len(listFinalDrops)
    #     label_map.fill(0)  # Reset label map

    #     # Check Collision
    #     for drop in listFinalDrops:
    #         # Check the bounding
    #         (ix, iy) = drop.getCenters()
    #         radius = drop.getRadius()
    #         ROI_WL = min(2 * radius, ix)
    #         ROI_WR = min(2 * radius, imgw - ix)
    #         ROI_HU = min(3 * radius, iy)
    #         ROI_HD = min(2 * radius, imgh - iy)

    #         drop_label = drop.getLabelMap()

    #         # Ensure the calculated indices are within the valid range
    #         start_y, end_y = iy - ROI_HU, iy + ROI_HD
    #         start_x, end_x = ix - ROI_WL, ix + ROI_WR

    #         if start_y < 0 or end_y > imgh or start_x < 0 or end_x > imgw:
    #             continue  # Skip if any of the indices are out of bounds

    #         # Validate the dimensions of the slices match
    #         label_map_slice = label_map[start_y:end_y, start_x:end_x]
    #         drop_label_slice = drop_label[3 * radius - ROI_HU:3 * radius + ROI_HD, 2 * radius - ROI_WL:2 * radius + ROI_WR]

    #         if label_map_slice.shape != drop_label_slice.shape:
    #             continue  # Skip if the shapes of the slices do not match

    #         # Apply raindrop label map to the Image's label map
    #         if label_map[iy, ix] > 0:
    #             col_ids = np.unique(label_map[start_y:end_y, start_x:end_x])
    #             col_ids = col_ids[col_ids != 0]
    #             drop.setCollision(True, col_ids)
    #             label_map[start_y:end_y, start_x:end_x] = drop_label_slice * drop.getKey()
    #         else:
    #             label_map[start_y:end_y, start_x:end_x] = drop_label_slice * drop.getKey()
    #             collisionNum -= 1

    #     if collisionNum > 0:
    #         listFinalDrops = CheckCollision(listFinalDrops)

    return listFinalDrops, lisShapes, lisSizes
    
    


def generateDrops(bg_img, cfg, listFinalDrops):
    """
    Generate raindrops on the image
    :param bg_img: image on which you want to generate drops
    :param cfg: config with global constants
    :param listFinalDrops: final list of raindrop class objects after handling collisions
    """
    edge_ratio = cfg["edge_darkratio"]

    label_map = np.zeros_like(bg_img)[:,:,0]
    imgh, imgw, _ = bg_img.shape

    alpha_map = np.zeros_like(label_map).astype(np.float64)

    for drop in listFinalDrops:
        (ix, iy) = drop.getCenters()
        radius = drop.getRadius()
        ROI_WL = 2 * radius
        ROI_WR = 2 * radius
        ROI_HU = 3 * radius
        ROI_HD = 2 * radius

        if (iy - 3 * radius) < 0:
            ROI_HU = iy
        if (iy + 2 * radius) > imgh:
            ROI_HD = imgh - iy
        if (ix - 2 * radius) < 0:
            ROI_WL = ix
        if (ix + 2 * radius) > imgw:
            ROI_WR = imgw - ix

        drop_alpha = drop.getAlphaMap()

        alpha_map_slice = alpha_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix + ROI_WR]
        drop_alpha_slice = drop_alpha[3 * radius - ROI_HU:3 * radius + ROI_HD, 2 * radius - ROI_WL: 2 * radius + ROI_WR]

        if alpha_map_slice.shape == drop_alpha_slice.shape:
            alpha_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix + ROI_WR] += drop_alpha_slice

    alpha_map = alpha_map / np.max(alpha_map) * 255.0

    for idx, drop in enumerate(listFinalDrops):
        (ix, iy) = drop.getCenters()
        radius = drop.getRadius()
        ROIU = iy - 3 * radius
        ROID = iy + 2 * radius
        ROIL = ix - 2 * radius
        ROIR = ix + 2 * radius

        if (iy - 3 * radius) < 0:
            ROIU = 0
            ROID = 5 * radius
        if (iy + 2 * radius) > imgh:
            ROIU = imgh - 5 * radius
            ROID = imgh
        if (ix - 2 * radius) < 0:
            ROIL = 0
            ROIR = 4 * radius
        if (ix + 2 * radius) > imgw:
            ROIL = imgw - 4 * radius
            ROIR = imgw

        tmp_bg = bg_img[ROIU:ROID, ROIL:ROIR, :]
        try:
            drop.updateTexture(tmp_bg)
        except:
            del listFinalDrops[idx]
            continue

        tmp_alpha_map = alpha_map[ROIU:ROID, ROIL:ROIR]
        output = drop.getTexture()
        tmp_output = np.asarray(output).astype(np.float64)[:, :, -1]
        tmp_alpha_map = tmp_alpha_map * (tmp_output / 255)
        tmp_alpha_map = Image.fromarray(tmp_alpha_map.astype('uint8'))

        edge = ImageEnhance.Brightness(output)
        edge = edge.enhance(edge_ratio)
        PIL_bg_img = Image.fromarray(bg_img)
        PIL_bg_img.paste(edge, (ix - 2 * radius, iy - 3 * radius), output)
        PIL_bg_img.paste(output, (ix - 2 * radius, iy - 3 * radius), output)
        bg_img = np.asarray(PIL_bg_img)

    return bg_img


