import pickle

file1 = r"D:\Coding\lightweight-human-pose-estimation.pytorch\data\dewarp_labeled_data\train_8_corners\images\prepared_train_annotation.pkl"
file2 = r"D:\Coding\lightweight-human-pose-estimation.pytorch\data\dewarp_labeled_data\Invoice_Toyota4_Dewarp_Training_HSBCdata\images\prepared_train_annotation.pkl"

with open(file1, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    _label1 = u.load()

    print("Length file 1", len(_label1))
    with open(file2, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        _label2 = u.load()
        print("Length file 2", len(_label2))
        _label1.extend(_label2)
        print("Length file +", len(_label1))

    with open(r"D:\Coding\lightweight-human-pose-estimation.pytorch\data\dewarp_labeled_data\train_8_corners\images\prepared_train_annotation.pkl", 'wb') as f:
        pickle.dump(_label1, f)

# w = open("example.txt", 'w')
# for i in range(10):
#      w.write("This is line %d\r" % (i+1))
# w.close() 