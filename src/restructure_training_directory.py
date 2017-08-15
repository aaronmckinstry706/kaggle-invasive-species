import os
import os.path as path

import definitions as defs

if __name__ == '__main__':
    if not path.exists(defs.TRAINING_DATA_DIR + '/0'):
        os.makedirs(defs.TRAINING_DATA_DIR + '/0')
    if not path.exists(defs.TRAINING_DATA_DIR + '/1'):
        os.makedirs(defs.TRAINING_DATA_DIR + '/1')
    
    labels_csv_path = defs.TRAINING_DATA_DIR + '/train_labels.csv'
    for line in open(labels_csv_path, 'r'):
        # Skip header.
        if 'name,invasive' in line:
            continue
        
        # Move each image to the subdirectory named as the image's label.
        line = line.strip()
        image_name, image_label = line.split(',')
        image_name = image_name + '.jpg'
        image_path = defs.TRAINING_DATA_DIR + '/' + image_name
        image_destination_path = \
            defs.TRAINING_DATA_DIR + '/' + image_label + '/' + image_name
        os.system("mv " + image_path + " " + image_destination_path)
