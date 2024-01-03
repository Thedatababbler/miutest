# refrence  1. https://github.com/waspinator/pycococreator/blob/master/examples/shapes/shapes_to_coco.py
#           2. https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
#           3. https://www.immersivelimit.com/create-coco-annotations-from-scratch
# modified by Huahui Yi


import os
import re
import json
import fnmatch
import datetime
import argparse
import cv2


import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
from  tqdm import tqdm
import pycocotools.mask as mask_util

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]




class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        # bbox[2] += bbox[0]
        # bbox[3] += bbox[1]
        return bbox





def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width, height))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x, y), 1)

    return sub_masks

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

def get_segmentations(category_info, binary_mask, image_size=None, tolerance=2):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)


    if category_info["is_crowd"]:
        segmentations = binary_mask_to_rle(binary_mask)
    else :
        segmentations = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentations:
            return None

    return segmentations



def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png', '*.bmp']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_masks(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files




INFO = {
    "description": "Mask2Polygon Dataset",
    "url": "xxx",
    "version": "0.1.0",
    "year": 2023,
    "contributor": "Huahui Yi",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]


CATEGORIES = [
    {
        'id': 1,
        'name': 'liver',
        'supercategory': 'endoscopy',
    },
    {
        'id': 2,
        'name': 'left kidney',
        'supercategory': 'photography',
    },
    {
        'id': 3,
        'name': 'right kidney',
        'supercategory': 'photography',
    },
    {
        'id': 4,
        'name': 'spleen',
        'supercategory': 'photography',
    },
    # {
    #     'id': 5,
    #     'name': 'nuclei',
    #     'supercategory': 'histopathology',
    # },
    # {
    #     'id': 6,
    #     'name': 'tuberculosis',
    #     'supercategory': 'x ray',
    # },
    # {
    #     'id': 7,
    #     'name': 'lung nodule',
    #     'supercategory': 'ct',
    # },
    # {
    #     'id': 8,
    #     'name': 'hippocampus',
    #     'supercategory': 'mri',
    # },
    # {
    #     'id': 9,
    #     'name': 'thyroid nodule',
    #     'supercategory': 'ultrasound',
    # },
    # {
    #     'id': 10,
    #     'name': 'breast nodule',
    #     'supercategory': 'ultrasound',
    # },
    # {
    #     'id': 11,
    #     'name': 'red blood cell',
    #     'supercategory': 'cytology',
    # },
    # {
    #     'id': 12,
    #     'name': 'white blood cell',
    #     'supercategory': 'cytology',
    # },
    # {
    #     'id': 13,
    #     'name': 'platelets',
    #     'supercategory': 'cytology',
    # },
]

# color_category_ids = {
#     # '(1, 1, 1)': 1,
#     '(201, 201, 201)': 1,
#     '(255, 255, 255)':1,
#     '(128, 128, 128)': 1,
#     '(7, 7, 7)': 1,
#     '(2, 2, 2)': 1,
#     '(25, 25, 25)': 1,
#     '(44, 44, 44)':1,
#     '(42, 42, 42)':1,
#     '(45, 45, 45)':1,

# }
from collections import defaultdict

color_category_ids = defaultdict(int)

def parse_args():
    parser = argparse.ArgumentParser(description='A tool for converting binary mask to coco style polygon annotations')
    parser.add_argument(
        '--dataset', 
        type=str,
        default='data/PolypB',
        help='the root path for dataset folder')
    parser.add_argument(
        '--dataname', 
        type=str,
        default='CVC-300',
        help='the dataset name')

    args = parser.parse_args()

    return args


# stand folder style:
#------------------------------------------
#    --dataset
#      --train
#      --val 
#      --test(optional)  
#      --mask_train
#      --mask_val 
#      --mask_test(optional)  
#      --annotations
#        --instance_train.json
#        --instance_val.json 
#        --instance_test.json(optional)  
#------------------------------------------
def main():
    args = parse_args()
    
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    
    image_id = 1
    segmentation_id = 1
    dataset_dir = args.dataset
    data_name = args.dataname
    data_mask_name = 'GroundMask'#'ISIC2018_Task1_Test_GroundTruth'#"mask_" + data_name
    image_dir = os.path.join(dataset_dir, data_name)
    mask_dir = os.path.join(dataset_dir, data_mask_name)
    anno_dir = os.path.join(dataset_dir, "annotations")
    # import pdb; pdb.set_trace()
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)
    
    mask_files = os.listdir(mask_dir)
    # filter for jpeg images
    color_count = 0
    for root, _, files in os.walk(image_dir):
        
    
        # files.sort(key = lambda x: int(x[:-4]))
        image_files = filter_for_jpeg(root, files)

        # go through each image
        # for image_filename in tqdm(image_files):
        for idx, image_filename in tqdm(enumerate(image_files)):
            # if 'mask' in image_filename:
            #     continue
            image = Image.open(image_filename)
            image_info = create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            # for root, _, files in os.walk(mask_dir):
            #     mask_files = filter_for_masks(root, files, image_filename)

                # go through each associated annotation
                # for mask_file in mask_files:
                    
                    # print(mask_file)
                    # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
            # import pdb;pdb.set_trace()
            # mask_file = image_filename.replace(image_dir, mask_dir)
            mask_file = os.path.join(mask_dir, mask_files[idx])
            # mask_file = mask_file.replace('*.png', '_mask.png') 
            # sta, end = image_filename.split('/')[-1].split('.')[0], image_filename.split('/')[-1].split('_')[-1].split('.')[0]
            # sta, end = 
            # end = int(end)+1
            # mask_file = os.path.join(mask_dir, f'{sta}.liver_GT_000 ({end}).png')
            # mask_file = os.path.join(mask_dir, f'{sta}.IMG-0004-00002 ({end}).png')
            # if dataset_dir.split("/")[-1] == "dfuc2022":
            #     # mask_file = mask_file.replace(".jpg", ".png")        
            mask_file = Image.open(mask_file)
            mask_file = mask_file.convert("RGB")
            sub_masks = create_sub_masks(mask_file)
            # import pdb;pdb.set_trace()
            for color, sub_mask in sub_masks.items():
                # if color == '(255, 255, 255)':
                if  color_category_ids[color] == 0:
                    color_count += 1
                    category_id = color_category_ids[color] + color_count
                    color_category_ids[color] = category_id
                    # print(color_count)
                    print(color, color_category_ids[color], color_count)
                else:
                
                    category_id = color_category_ids[color] #+ color_count
                    
                #color_category_ids[color] = category_id
                category_info = {'id': category_id, 'is_crowd': 'crowd' in image_filename}
              
                binary_mask = np.asarray(sub_mask).astype(np.uint8)
                if category_info["is_crowd"]:
                    is_crowd = 1
                else:
                    is_crowd = 0
                #category_info, binary_mask, image_size=None, tolerance=2
                segmentations = get_segmentations(category_info, binary_mask, image.size, tolerance=2)
                
                # import pdb;pdb.set_trace()
                if segmentations:
            
                    for segmentation in segmentations:
                        segmentation = [segmentation]

                        Gseg = GenericMask(segmentation, binary_mask.shape[0], binary_mask.shape[1])
                        area = Gseg.area()
                        bounding_box = Gseg.bbox()
                        if area > 50:
                            annotation_info = {
                                "id": segmentation_id,
                                "image_id": image_id,
                                "category_id": category_info["id"],
                                "iscrowd": is_crowd,
                                "area": area.tolist(),
                                "bbox": bounding_box.tolist(),
                                "segmentation": segmentation,
                                "width": binary_mask.shape[1],
                                "height": binary_mask.shape[0],
                            } 

                            
                            coco_output["annotations"].append(annotation_info)

                            segmentation_id = segmentation_id + 1

            image_id = image_id + 1
    anno_path = os.path.join(anno_dir, 'test.json')
    with open(anno_path, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
        
        
        
if __name__ == "__main__":
    main()