# import os 

# root_dir = r"D:\Coding\lightweight-human-pose-estimation.pytorch\data\dewarp_labeled_data\Invoice_Toyota4_Dewarp_Training_HSBCdata"

# index = 0
# for file in os.listdir(root_dir + "/images"):
#     os.rename(root_dir + "/images/" )
import pickle
pk = r"D:\Coding\keypoint-detection\visualize\prepared_train_annotation.pkl"

idx = 0
with open(pk, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    _labels = u.load()
    
    for label in _labels:
        if label['scale_provided'] != 0:
            print(label)
            idx += 1

    print(idx)
    print(len(_labels))