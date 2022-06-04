def show_image(ann_id):
    ann = annotations.iloc[i]
    im_id = ann['image_id']
    box_adress = ann['bbox']
    im = imageio.imread('public_dataset/reference_images_part1/'+images.loc[images['id']==im_id]['file_name'].values[0])
    pylab.imshow(im[box_adress[1]:box_adress[1]+box_adress[3], box_adress[0]:box_adress[0]+box_adress[2]])
