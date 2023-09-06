from ipywidgets.widgets.interaction import interactive


def set_base_configs(config_path='',zoom=0.5, box_option='Load Existing Box', pose_option='Sample Random Pose'):
    return {"config_path": config_path,
            "zoom": zoom,
            "create_box": True if box_option == 'Create New Box' else False,
            "sample_pose": True if pose_option == 'Sample Random Pose' else False}


def set_create_box_configs(box_path='', box_center_x=0, box_center_y=0, box_center_z=0, edge_size_x=1., edge_size_y=1., edge_size_z=1.):
    return {'box_path': box_path,
            'box_center_x': box_center_x,
            'box_center_y': box_center_y,
            'box_center_z': box_center_z,
            'edge_size_x': edge_size_x,
            'edge_size_y': edge_size_y,
            'edge_size_z': edge_size_z}


def set_load_box_configs(box_path=''):
    return {'box_path': box_path}


def set_pose_configs(pose_index=0):
    return {'pose_index':pose_index}


w_base_configs = interactive(
    set_base_configs,
    config_path='',
    box_option=['Load Existing Box', 'Create New Box'],
    pose_option=['Sample Random Pose','Set Pose Index'],
    zoom=(0.1, 1.5, 0.1)
)

w_create_box_configs = interactive(
    set_create_box_configs,
    box_path='',
    box_center_x=(-1.5, 1.5, 0.1),
    box_center_y=(-1.5, 1.5, 0.1),
    box_center_z=(-1.5, 1.5, 0.1),
    edge_size_x=(0.0, 1.5, 0.1),
    edge_size_y=(0.0, 1.5, 0.1),
    edge_size_z=(0.0, 1.5, 0.1),
)

w_load_box_configs = interactive(
    set_load_box_configs,
    box_path=''
)

w_pose_configs = interactive(
    set_pose_configs,
    pose_index=(0, 119, 1),
)


