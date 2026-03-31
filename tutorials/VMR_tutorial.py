"""
Basic tutorial of the Vascular Encoding Framework using a model publicly available on the
Vascular Model Repository (https://www.vascularmodel.com/index.html).

To run this tutorial the user need to donwload the case 0010_H_AO_H It can be found
using the search functionality on the filters tab of the repository web.

After downloading and unziping it, the user can either move the directory inside the tutorials
directory or modify this file to set the path to the unziped directory.
"""


import os
import sys

import numpy as np
import pyvista as pv
import vascular_encoding_framework as vef

"""
To ensure that the code correctly references your working directory, you need to modify 
the case_path variable. 

If your directory structure is /home/user/desktop/vascular_encoding_framework/tutorials/0010_H_AO_H, 
you can set it as follows:
"""

case_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mesh_path = os.path.join(case_path, 'Models', '0093_0001.vtp')

mesh = pv.read(mesh_path)
mesh = mesh.threshold(value=0.1, scalars='CapID',
                      method='lower').extract_surface()

# Smooth the mesh to reduce potential segmentation artifacts
# mesh = mesh.smooth(n_iter=0)  # Increased smoothing iterations

mesh = mesh.compute_normals(
    auto_orient_normals=True, flip_normals=False, consistent_normals=True
)

mesh = mesh.subdivide(1)  # Increase the level of subdivision if necessary

# Initialize vef.VascularMesh with the opened mesh
vmesh = vef.VascularMesh(mesh)

vmesh.plot_boundary_ids()
# Define the hierarchy
hierarchy = {
    "B5": {"id": "B5", "parent": None, "children": {"B0"}},
    "B0": {"id": "B0", "parent": "B5", "children": {"B3", "B4", "B1"}},
    "B3": {"id": "B3", "parent": "B0", "children": {"B2"}},
    "B4": {"id": "B4", "parent": "B0", "children": {}},
    "B1": {"id": "B1", "parent": "B0", "children": {}},
    "B2": {"id": "B2", "parent": "B3", "children": {}},
}
vmesh.set_boundary_data(hierarchy)

print("Initial boundaries:")

# Check boundary connectivity
for boundary_id, boundary in vmesh.boundaries.items():
    print(f"Boundary {boundary_id} connected to: {boundary.children}")

# Function to extract centerline domain
# Attempt to extract the centerline domain with adjusted parameters
c_domain = vef.centerline.extract_centerline_domain(
    vmesh=vmesh,
    params={'method': 'seekers', 'reduction_rate': 0, 'eps': 1e-3},
    debug=True
)

# Compute the path tree
cp_xtractor = vef.centerline.CenterlinePathExtractor()
cp_xtractor.debug = True
cp_xtractor.set_centerline_domain(c_domain)
cp_xtractor.set_vascular_mesh(vmesh, update_boundaries=True)
cp_xtractor.compute_paths()

print(f"Number of centerline domain points: {len(c_domain.points)}")
print(f"Centerline domain bounding box: {c_domain.bounds}")

# Define knots for each branch
knot_params = {
    "B5": {"cl_knots": None, "tau_knots": None, "theta_knots": None},
    "B0": {"cl_knots": 15, "tau_knots": 19, "theta_knots": 19},
    "B3": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "B4": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "B1": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "B2": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
}

try:
    cl_tree = vef.CenterlineTree.from_multiblock_paths(
        cp_xtractor.paths,
        **{k: {'n_knots': v['cl_knots']} for k, v in knot_params.items() if v['cl_knots'] is not None}
    )

    # Plot the adapted frame
    cl_tree.plot_adapted_frame(vmesh=vmesh, scale=0.5)

    # Compute centerline association and vessel coordinates
    bid = [
        cl_tree.get_centerline_association(
            p=vmesh.points[i],
            n=vmesh.get_array(name='Normals', preference='point')[i],
            method='scalar',
            thrs=60,
        )
        for i in range(vmesh.n_points)
    ]
    vcs = np.array(
        [
            cl_tree.cartesian_to_vcs(p=vmesh.points[i], cl_id=bid[i])
            for i in range(vmesh.n_points)
        ]
    )
    vmesh['cl_association'] = bid
    vmesh['tau'] = vcs[:, 0]
    vmesh['theta'] = vcs[:, 1]
    vmesh['rho'] = vcs[:, 2]

    # Print data statistics (These are for Unitary Tests)
    print("Tau values - Mean:",
          np.mean(vmesh['tau']), "Std:", np.std(vmesh['tau']))
    print(
        "Theta values - Mean:", np.mean(vmesh['theta']
                                        ), "Std:", np.std(vmesh['theta'])
    )
    print("Rho values - Mean:",
          np.mean(vmesh['rho']), "Std:", np.std(vmesh['rho']))

except Exception as e:
    print(f"Error in creating CenterlineNetwork or processing data: {str(e)}")

print("Script completed.")
