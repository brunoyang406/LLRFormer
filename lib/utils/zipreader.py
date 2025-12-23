from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np

_im_zfile = []
_xml_path_zip = []
_xml_zfile = []


def imread(filename, flags=cv2.IMREAD_COLOR):
    """Read image from zip file."""
    global _im_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        raise ValueError("character '@' is not found from the given path '%s'"%(path))
    path_zip = path[0: pos_at]
    path_img = path[pos_at + 2:]
    if not os.path.isfile(path_zip):
        raise FileNotFoundError("zip file '%s' is not found"%(path_zip))
    for i in range(len(_im_zfile)):
        if _im_zfile[i]['path'] == path_zip:
            data = _im_zfile[i]['zipfile'].read(path_img)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    _im_zfile.append({
        'path': path_zip,
        'zipfile': zipfile.ZipFile(path_zip, 'r')
    })
    data = _im_zfile[-1]['zipfile'].read(path_img)

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def xmlread(filename):
    """Read XML file from zip archive."""
    global _xml_path_zip
    global _xml_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        raise ValueError("character '@' is not found from the given path '%s'"%(path))
    path_zip = path[0: pos_at]
    path_xml = path[pos_at + 2:]
    if not os.path.isfile(path_zip):
        raise FileNotFoundError("zip file '%s' is not found"%(path_zip))
    for i in range(len(_xml_path_zip)):
        if _xml_path_zip[i] == path_zip:
            data = _xml_zfile[i].open(path_xml)
            return ET.fromstring(data.read())
    _xml_path_zip.append(path_zip)
    _xml_zfile.append(zipfile.ZipFile(path_zip, 'r'))
    data = _xml_zfile[-1].open(path_xml)
    return ET.fromstring(data.read())
