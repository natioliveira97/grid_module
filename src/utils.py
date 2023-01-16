import numpy as np

##############
# DataLoader #
##############
def tile_image(img, num_tiles_height, num_tiles_width, resize_dimensions, tile_dimensions, tile_overlap):
    """
    Description: Tiles image with overlap
    Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    WARNING: Tile size must divide perfectly into image height and width
    """
    bytelength = img.nbytes // img.size
    channels=img.shape[2]
    
    # img.shape = [num_tiles_height, num_tiles_width, tile_height, tile_width, 3]
    img = np.lib.stride_tricks.as_strided(img, 
        shape=(num_tiles_height, 
               num_tiles_width, 
               tile_dimensions[0], 
               tile_dimensions[1],
               channels), 
        strides=(resize_dimensions[1]*(tile_dimensions[0]-tile_overlap)*bytelength*channels,
                 (tile_dimensions[1]-tile_overlap)*bytelength*channels, 
                 resize_dimensions[1]*bytelength*channels, 
                 bytelength*channels, 
                 bytelength), writeable=False)

    # img.shape = [num_tiles, tile_height, tile_width, num_channels]
    img = img.reshape((-1, tile_dimensions[0], tile_dimensions[1], channels))
    
    return img

def tile_labels(labels, num_tiles_height, num_tiles_width, image_dimensions,thresh_overlap=0.3):
    final_label = np.zeros((num_tiles_height,num_tiles_height), np.bool_)
    ground_truth = 0
    for label in labels:
        image = np.zeros(image_dimensions)
        image[label[1]:label[3],label[0]:label[2]]=1.0
        sh = num_tiles_height,image_dimensions[0]//num_tiles_height,num_tiles_width,image_dimensions[1]//num_tiles_width
        image = image.reshape(sh).mean(-1).mean(1)
        image = (image>thresh_overlap)
        final_label=final_label | image
        ground_truth = 1 

    
    return final_label.astype(int).flatten(),ground_truth