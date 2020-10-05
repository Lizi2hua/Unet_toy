import os
#========enviroment conf======#
'''
    如果运行的环境变了，只需将这块的值改了。
'''
DISC=r'F:'
BATCH_SIZE=4
EPOCH=1000
MODEL_SAVEPATH=r'SAVE'
download_url='https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
#========enviroment conf======#

IMG_PATH=os.path.join(DISC,'\VOCdevkit\VOC2012\JPEGImages')
SEGLABE_PATH=os.path.join(DISC,'\VOCdevkit\VOC2012\SegmentationClass')

