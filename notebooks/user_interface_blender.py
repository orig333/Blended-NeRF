from ipywidgets.widgets.interaction import interactive


def set_base_configs(config_path='', theta=0, phi=0, radius=3.5, box_option='Load Existing Box'):
    return {"config_path": config_path,
            "theta": theta,
            "phi": phi,
            "radius": radius,
            "create_box": True if box_option == 'Create New Box' else False}


def set_create_box_configs(box_path='', box_center_x=0, box_center_y=0, box_center_z=0, edge_size_x=1., edge_size_y=1., edge_size_z=1.):
    return {'box_path': box_path,
            'box_center_x': box_center_x,
            'box_center_y': box_center_y,
            'box_center_z': box_center_z,
            'edge_size_x': edge_size_x,
            'edge_size_y': edge_size_y,
            'edge_size_z': edge_size_z}


w_base_configs = interactive(
    set_base_configs,
    config_path='',
    theta=(-360, 360, 1),
    phi=(-360, 360, 1),
    radius=(1.5, 4.5, 0.1),
    box_option=['Load Existing Box', 'Create New Box'],
)


w_create_box_configs = interactive(
    set_create_box_configs,
    box_path='',
    box_center_x=(-5., 5., 0.1),
    box_center_y=(-5., 5., 0.1),
    box_center_z=(-5., 5., 0.1),
    edge_size_x=(0.0, 3.0, 0.1),
    edge_size_y=(0.0, 3.0, 0.1),
    edge_size_z=(0.0, 3.0, 0.1),
)
