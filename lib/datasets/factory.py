# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
import os
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_integrated import pascal_voc_integrated
from datasets.voc_clipart import voc_clipart
from datasets.voc_watercolor import voc_watercolor
from datasets.voc_comic import voc_comic

from datasets.source_composite import pascal_voc_composite,source_dataset_name
from datasets.target_real import pascal_voc_real


# composite
for split in ['trainval']:
  name = 'source_{}'.format(split)
  # __sets[name] = (lambda split=split: pascal_voc_integrated(split, devkit_path='datasets/Pascal/VOCdevkit/VOC_Integrated'))
  __sets[name] = (lambda split=split: pascal_voc_composite(split, devkit_path='datasets/{}'.format(source_dataset_name)))

# target: real not annotated for train
for split in ['train','test']:
    name = 'target_{}'.format(split)
    class_name1 = "smoke"
    img_ext1="png"
    __sets[name] = (lambda split=split: pascal_voc_real('', split, class_name1,img_ext1,devkit_path=os.path.join('datasets/', 'real_not_annotated_voc')))

# targetval: real annotated for test
for split in ['test']:
    name = 'targetval_{}'.format(split)
    class_name1 = "smoke"
    img_ext1="png"
    __sets[name] = (lambda split=split: pascal_voc_real('', split, class_name1,img_ext1,devkit_path=os.path.join('datasets/', 'real_annotated_2_voc')))



# composite shift
for split in ['trainval']:
    for shift in ['CPR']:
        name = 'source{}_{}'.format(shift, split)
        class_name2="gas"
        img_ext2="jpg"
        __sets[name] = (lambda shift=shift, split=split: pascal_voc_real(shift, split, class_name2,img_ext2,devkit_path=os.path.join('datasets/', '{}{}'.format(source_dataset_name,shift))))








def get_imdb(name):
  """Get an imdb (image database) by name."""
  print("name=",name)
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  # print("imdb_name=",name)
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
