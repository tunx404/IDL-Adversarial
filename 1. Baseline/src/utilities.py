import scipy.io as sio

def read_PASCAL_Part_to_dict(file_path):

    file_mat = sio.loadmat(file_path)
    anno_key = file_mat['anno'] # Only care about the 'anno' key

    image_anno_dict = {}

    image_name = anno_key[0][0][0][0]
    image_anno_dict.update(image_name=image_name)

    objects_mat = anno_key[0][0][1][0]
    num_objects = len(objects_mat)
    image_anno_dict.update(num_objects=num_objects)

    objects_list = []
    for obj in objects_mat:
        object_dict = {}
        
        object_class = obj[0][0]
        object_dict.update(obj_class=object_class)

        object_mask = obj[2]
        object_dict.update(obj_mask=object_mask)

        parts_mat = obj[3]
        num_parts = 0 if len(parts_mat) == 0 else len(parts_mat[0])
        object_dict.update(num_parts=num_parts)

        parts_list = []
        if num_parts != 0:
            for part in parts_mat[0]:
                part_dict = {}
                part_name = part[0][0]
                part_mask = part[1]
                part_dict.update(part_name=part_name)
                part_dict.update(part_mask=part_mask)
                parts_list.append(part_dict)
        object_dict.update(parts_list=parts_list)
        objects_list.append(object_dict)
    image_anno_dict.update(objects_list=objects_list)

    return image_anno_dict