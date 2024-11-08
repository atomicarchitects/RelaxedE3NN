import plotly.graph_objects as go
from e3nn.io import SphericalTensor
import torch
from models.relaxed_e3nn_conv_model import RelaxedConvolution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_geometric.data import Data
from e3nn import o3
import numpy as np

def setup_geom(inp_geom, out_geom, L_max):
    # sets up geometry for square to rectangle etc
    input = torch.zeros((L_max + 1)**2)  # Factor of 2 for both parities?
    input[0] = 1
    #displacements = out_geom-inp_geom
    sph = SphericalTensor(L_max, p_val=1, p_arg=-1)
    # spherical harmonic projection of each displacement vector
    #displacements = out_geom-inp_geom
    # instead just use spherical harmonic projections of the geometries
    projections = torch.stack([sph.with_peaks_at(out_elem.reshape(1,3)) for out_elem in out_geom], dim=0)
    data = Data(x=input.repeat(len(inp_geom),1),pos=inp_geom).to(device)
    return data, projections

def train(data,projections,model,opt,epochs, regularize= False, weight_decay = 1e-6, norm = 2, use_scheduler= False):
    losses = []
    accuracies = []
    best_loss = 1e10
    relaxed_weights = []
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.97)
    for step in range(epochs):
        pred = model(data.to(device))
        loss = (pred-projections.to(device)).pow(2).sum()
        #print(loss)

        # Calculate regularization loss for relaxed weights
        if regularize:
            regularization_loss = torch.tensor(0.).to(device)
            for module in model.model.children():  # Accessing the Sequential part
                if isinstance(module, RelaxedConvolution):
                    regularization_loss += torch.nn.functional.mse_loss(module.relaxed_weights,torch.tensor([0.0] * module.relaxed_weights.size(0)).to(device))
            regularization_loss *= weight_decay
            #print(regularization_loss)
            
            # Total loss including regularization
            loss = loss + regularization_loss
        if loss < best_loss:
            best_output = pred
            best_loss = loss
            best_model = model.state_dict()
        losses.append(loss.detach().cpu().numpy())
        #relaxed_weights.append(model.conv.relaxed_weights.clone().detach().numpy())
        #if loss < 1e-10:
        #    break
        opt.zero_grad()
        loss.backward()
        opt.step()
        if use_scheduler:
            scheduler.step()
        if step % 200 == 0:
            print(f"epoch {step:5d} | loss {loss:<10.6f}")
    return losses, best_output, best_model#, relaxed_weights

def plot_output(start, finish, features, start_label, finish_label, L_max, bound=None):
    if bound is None:
        bound = max(start.norm(dim=1).max(), finish.norm(dim=1).max()).item()
    axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title='',
        nticks=3,
        range=[-bound, bound]
    )

    resolution = 500
    layout = dict(
        width=resolution,
        height=resolution,
        scene=dict(
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=2),
                projection=dict(type='perspective'),
            ),
        ),
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
    )

    traces = [
        go.Scatter3d(x=start[:, 0], y=start[:, 1], z=start[:, 2], mode="markers", name=start_label),
        go.Scatter3d(x=finish[:, 0], y=finish[:, 1], z=finish[:, 2], mode="markers", name=finish_label),
    ]
    
    for center, signal in zip(start, features):
        sph = SphericalTensor(L_max, p_val=1, p_arg=-1)
        if isinstance(signal, torch.Tensor):
            r, f = sph.plot(signal=signal.detach(),center=center)
        else:
            r, f = sph.plot(signal= signal,center=center)
        traces += [go.Surface(x=r[..., 0], y=r[..., 1], z=r[..., 2], surfacecolor=f.numpy(), showscale=False)]
        
    return go.Figure(traces, layout=layout)

def plot_weights(relaxed_weights, layer, L_max, scale_radius=True):
    # visualize relaxed weights 
    # plot spherical signal of weights?
    sph = SphericalTensor(L_max, p_val=1, p_arg=-1)
    #r,f = sph.plot(signal = torch.tensor(relaxed_weights))
    fig = go.Figure([go.Surface(**sph.plotly_surface(torch.tensor(relaxed_weights),radius=scale_radius)[0])])
    fig.update_layout(
        #scene=dict(
        #    camera=dict(
        #        eye=dict(x=2.0, y=2.0, z=2.0)  # Adjust the 'z' value to zoom out/in
        #    )
        #),
        xaxis={'range': [-0.5, 0.5]},  # Set x-axis range
        yaxis={'range': [-0.5, 0.5]},
        title = {'text':'Relaxed Weights Spherical Harmonic Projection ' + 'Layer ' + str(layer),
        'x': 0.4,
        'y': 0.8}
    )
    return fig


def print_relaxed_weights(weights, mul_relaxed = 1,lmax_relaxed = 2):
    sph_tensor = SphericalTensor(lmax_relaxed,p_val=1,p_arg=-1)
    sph_even_ir = []
    irreps_relaxed_iter = o3.Irrep.iterator(lmax_relaxed)
    #irreps_relaxed=o3.Irreps("1x0e+1x1o+2x2e")
    irreps_relaxed = o3.Irreps("")
    for irrep in irreps_relaxed_iter:
        irreps_relaxed += mul_relaxed*irrep
    curr_ind = 0
    for i, (mul, ir) in enumerate(irreps_relaxed):
        print("Irrep ", ir)
        if ir in sph_tensor:
            sph_even_ir.append(np.round(weights[curr_ind:curr_ind+ir.dim*mul_relaxed].detach().cpu().numpy(),6))
        print(np.round(weights[curr_ind:curr_ind+ir.dim*mul_relaxed].detach().cpu().numpy(),6))
        curr_ind += ir.dim
    return sph_even_ir

# make simulation
def sim_E_B_field(E_field, B_field,steps,dt,q,m):
    # Initial conditions for multiple particles
    num_particles = 10  # Number of particles
    np.random.seed(0)
    initial_positions = np.random.rand(num_particles, 3) #* 1e-6  # Random initial positions within a cube of side length 1e-6 m
    initial_velocities = np.random.rand(num_particles, 3)# * 1e4  # Random initial velocities (m/s)

    # Arrays to store trajectories
    trajectories = []

    # array to store velocities
    velocities = []

    # accels
    accels = []
    # Simulation loop for each particle
    for particle in range(num_particles):
        pos = initial_positions[particle]
        vel = initial_velocities[particle]
        particle_trajectory = [pos.copy()]
        particle_velocity = [pos.copy()]
        particle_accels = []
        
        for _ in range(steps):
            # Calculate acceleration using the Lorentz force equation
            accel = (E_field)+ np.cross(vel, B_field)
            #B_field_curr = np.cross(vel,B_field)
            # Update velocity and position using the calculated acceleration
            vel += accel * dt
            pos += vel * dt
            
            # Store the new position in the particle's trajectory array
            particle_trajectory.append(pos.copy())
            particle_velocity.append(vel.copy())
            particle_accels.append(accel.copy())
        
        # Store the particle's trajectory in the trajectories list
        trajectories.append(np.array(particle_trajectory))
        velocities.append(np.array(particle_velocity))
        accels.append(np.array(particle_accels))
    return trajectories, velocities, accels
