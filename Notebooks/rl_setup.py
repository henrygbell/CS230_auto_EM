import tensorflow as tf
import abtem
from abtem.detectors import AnnularDetector
import numpy as np
import gymnasium as gym
import numpy as np
# from ray.rllib.algorithms.ppo import PPOConfig
from scipy.ndimage import laplace, gaussian_filter
import ase
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from ray.rllib.utils.annotations import override
import torch.nn as nn

system = ase.io.read('../data/au_np_system.xyz')


"""
Helper functions for ML environment and simulation
"""

def crop_region(system, center_x, center_y, crop_size):
    # Compute boundaries
    x_min, x_max = center_x - crop_size / 2, center_x + crop_size / 2
    y_min, y_max = center_y - crop_size / 2, center_y + crop_size / 2

    # Create a mask for atoms within the crop area
    positions = system.positions
    mask = (
        (positions[:, 0] >= x_min) &
        (positions[:, 0] <= x_max) &
        (positions[:, 1] >= y_min) &
        (positions[:, 1] <= y_max)
    )

    # Extract the region of interest
    cropped = system[mask]

    # Optionally, redefine cell to the cropped region for clarity
    cell = system.cell.copy()
    cell[0, 0] = crop_size
    cell[1, 1] = crop_size
    cropped.set_cell(cell)
    cropped.center(axis=(0, 1))  # recenters within new cell, keeps z unchanged

    return cropped

def simulate_haadf(
    system, 
    sampling, 
    inner_angle, 
    outer_angle, 
    semi_angle_cutoff,
    crop_size, 
    center_x, 
    center_y, 
    aberation_dict, 
    dose_rate = 1e5,
    interpolation_sampling = 0.1,
): 
    """
    Run a HAADF STEM forward simulation for a cropped region of an ASE `system`.

    Parameters
    ----------
    system : ase.Atoms
        Full atomic system (ASE Atoms) containing substrate + nanoparticles.
    sampling : float
        Real-space sampling of the potential (units must match `system` positions, 
        typically Angstroms per pixel).
    inner_angle : float
        Inner detector angle for the annular detector (in mrad).
    outer_angle : float
        Outer detector angle for the annular detector (in mrad).
    semi_angle_cutoff : float
        Semi-angle cutoff for the probe (in mrad).
    crop_size : float
        Size of the crop box (must be in same length units as `system.positions`; e.g.
        Angstroms if `system` uses Å). Used to form a cropped ASE Atoms object.
    center_x, center_y : float
        Center coordinates (in same units as `system.positions`) for the crop region.
    aberation_dict : dict
        Dictionary of probe aberrations passed to abtem.Probe, e.g. {"C10": value, "C30": value}.
        Values should be in the units expected by your abTEM installation (typically nm for
        defocus/Cn but check abTEM docs for conventions).
    dose_rate : float, optional
        Dose per unit area used by `measurements.poisson_noise` (default 1e5).

    Returns
    -------
    img : array-like
        The simulated detector image array produced by measurements.array.compute().
        If abTEM uses a GPU-backed array (CuPy/Dask), this is the result of `.compute()`
        and may still be a GPU-backed object; calling `.get()` converts to a NumPy array.


    Example
    -------
    img = simulate_haadf(system, sampling=0.06, inner_angle=70, outer_angle=100,
                         crop_size=20.0, center_x=203, center_y=176,
                         aberation_dict={"C10": 0.0}, dose_rate=1e5)
    """

    cropped = crop_region(system, center_x, center_y, crop_size)

    potential = abtem.Potential(
        cropped,
        sampling = sampling,
        projection = 'infinite',
        slice_thickness = 12.0,
        device = 'gpu'
    ).build()

    # Option A: increase probe grid resolution so larger angles are simulated
    probe = abtem.Probe(
        energy = 300e3,
        semiangle_cutoff = semi_angle_cutoff,
        aberrations=aberation_dict,
        gpts = [128,128],
        sampling = [sampling * 0.5, sampling * 0.5],
        device="gpu",
    )
    # probe.grid.match(potential)
    
    detector_haadf = AnnularDetector(
        inner = inner_angle,  # in mrad
        outer = outer_angle, # in mrad
    )

    grid_scan = abtem.GridScan(
        start = [0,0],
        end = [1,1],
        sampling = probe.aperture.nyquist_sampling,
        fractional=True,
        potential=potential,
    )

    measurements = probe.scan(potential, scan = grid_scan, detectors = [detector_haadf])
    
    measurements = measurements.interpolate(sampling = interpolation_sampling) #interpolate to reasonable grid
    if dose_rate is not None:
        measurements = measurements.poisson_noise(dose_per_area = dose_rate) #add noise

    img = measurements.array.compute()

    img = np.clip(img / np.percentile(img, 99), a_min=0, a_max=1, out=img)  # ensure no negative values after noise
    return img

"""
Options for Reward functions
"""

def hf_energy_stack(images, k_cut=0.25):
    """
    Compute high-frequency Fourier energy for each image.
    
    Parameters
    ----------
    images : ndarray, shape (N, H, W)
        Stack of real images.
    k_cut : float in (0,1)
        Radial cutoff as fraction of Nyquist frequency. 
        Frequencies with |k| > k_cut will be counted.
    
    Returns
    -------
    scores : ndarray, shape (N,)
        HF energy score per image.
    """
    N, H, W = images.shape
    # FFT frequency grids
    ky = np.fft.fftfreq(H)  # in cycles/pixel
    kx = np.fft.fftfreq(W)
    kx, ky = np.meshgrid(kx, ky)
    k_r = np.sqrt(kx**2 + ky**2)

    # Mask for HF region
    mask = (k_r >= k_cut)

    scores = np.zeros(N)
    for i in range(N):
        F = np.fft.fft2(images[i])
        mag2 = np.abs(F)**2
        hf_power = np.sum(mag2[mask])
        total_power = np.sum(mag2)
        scores[i] = hf_power / total_power if total_power > 0 else 0.0

    return scores


def laplacian_energy_stack(images, sigma=None):
    """
    Compute Laplacian-based sharpness for each image.

    Parameters
    ----------
    images : ndarray, shape (N, H, W)
        Stack of real images.
    sigma : float or None
        If given, smooth before Laplacian using a Gaussian filter (sigma in pixels).

    Returns
    -------
    scores : ndarray, shape (N,)
        Laplacian energy per image (normalized to max=1).
    """

    N = images.shape[0]
    scores = np.zeros(N)

    for i in range(N):
        img = images[i].astype(np.float64, copy = True)

        # optional Gaussian smoothing before Laplacian
        if sigma is not None and sigma > 0:
            img = gaussian_filter(img, sigma=sigma)

        lap = laplace(img)
        scores[i] = np.sum(np.abs(lap))

    # Normalize to max=1 for convenience
    if scores.max() > 0:
        scores /= scores.max()

    return scores


simulation_parameters = {
    'sampling': 0.06,  # in Angstroms
    'inner_angle': 70,  # in mrad
    'outer_angle': 100,  # in mrad
    "interpolation_sampling": 10/64,
    "max_steps":30,
    'crop_size': 10,  # in nm
    'dose_rate': None,  # electrons per Angstrom^2
    'center_x': 203, #want to change this to sample from a list or something
    'center_y': 176,
    "system": system,
    # "focus_metric": {"func" : laplacian_energy_stack, "args": 1.0}
    "focus_metric": {"func": hf_energy_stack, "args":0.05}
}
img_size = int(np.round(simulation_parameters['crop_size']/simulation_parameters['interpolation_sampling'], 0)) 
simulation_parameters["image_shape"] = (img_size, img_size)


"""
gym and rllib setup
"""
class STEM_environment(gym.Env):
    def __init__(self, config = None):
        self.config = config
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(*config["image_shape"], 1), dtype=np.float32)
        self.action_space = gym.spaces.Box( # normalized action space
            low = np.array([-1., -1., -1., -1.], dtype=np.float32), 
            high = np.array([ 1.,  1.,  1.,  1.], dtype=np.float32),
            dtype=np.float32,
            shape=(4,)
        ) # actions: convergence semi angle change, defocus change, C3 change, C5 change
        self.state = (np.random.rand(4) - 0.5) / 3 # initial state is the absolute imaging parameters
        self.max_steps = config["max_steps"]
        self.x_offset = int(np.random.rand(1) * 10 - 5) # randomize starting offset
        self.y_offset = int(np.random.rand(1) * 10 - 5)

    def reset(self, seed=None, options=None):
        # Return (reset) observation and info dict.
        self.state = (np.random.rand(4) - 0.5) / 3
        self.step_count = 0
        self.x_offset = int(np.random.rand(1) * 10 - 5) # randomize starting offset
        self.y_offset = int(np.random.rand(1) * 10 - 5)
        obs = self._forward_model(self.state)
        # cache last observation and its focus metric to avoid redundant recomputation
        self._last_obs = obs
        self._last_metric = self._focus_metric(obs)
        
        
        return obs, {}
    
    
    def _focus_metric(self, obs):
        # Compute focus metric for given image.
        args = self.config["focus_metric"]["args"]
        func = self.config["focus_metric"]["func"]
        return float(func(obs[:,:,0][None], args))

    def _compute_new_state(self, action):
        return np.clip(self.state + action * 0.2, a_min = -1, a_max = 1) # update state to new absolute parameters

    def _state_to_params(self, state):
        semi_angle_cutoff = 40*(state[0] + 1 )/ 2 + 5 # map to [5, 45] mrad
        defocus = 50*state[1]-25 # map to [-75, 25] nm, central focus is near 25 nm
        C3 = 1000*state[2] # map to [-1000, 1000] nm
        C5 = 1e6*state[3] # map to [-1e6, 1e6] nm
        return semi_angle_cutoff, defocus, C3, C5

    def _forward_model(self, action):
        action_absolute = self._compute_new_state(action) # absolute imaging parameters
        semi_angle_cutoff, defocus, C3, C5 = self._state_to_params(action_absolute)
        
        img = simulate_haadf(
            system = self.config['system'], 
            sampling = self.config['sampling'], 
            inner_angle = self.config['inner_angle'], 
            outer_angle=self.config['outer_angle'], 
            semi_angle_cutoff=semi_angle_cutoff,
            crop_size=self.config['crop_size'], 
            center_x=self.config['center_x'] + self.x_offset,
            center_y=self.config['center_y'] + self.y_offset,
            aberation_dict={"C10": -defocus, "C30": C3, "C50": C5},
            dose_rate = self.config['dose_rate'],
            interpolation_sampling=self.config["interpolation_sampling"]
        )
        obs = img[:,:,None]

        return obs
        
    def step(self, action):
        action = np.clip(action, a_min = -1, a_max = 1)
        obs = self._forward_model(action)
        focus_metric = self._focus_metric(obs)
        
        # Improved reward: combination of absolute quality and improvement
        improvement = focus_metric - self._last_metric
        raw_reward = 10 * improvement + focus_metric  # Reward both improvement and absolute quality
        reward = np.tanh(raw_reward)  # Smoother clipping than hard clip
        
        terminated = False
        truncated = False
        info = {"focus_metric": focus_metric, "improvement": improvement}
        
        self._last_metric = focus_metric
        self.state = self._compute_new_state(action)
        self.step_count += 1
        
        terminated = self.step_count >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    

class STEM_environment_history(gym.Env):
    """
    STEM environment that tracks the last 2 images and last 2 actions.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.max_steps = config.get("max_steps", 30)
        self.system = config["system"]
        
        # History parameters
        self.num_frames = 2  # Keep last 2 images
        self.num_actions = 2  # Keep last 2 actions
        
        img_shape = config["image_shape"]
        
        # Observation space: last 2 images + last 2 actions
        # FIXED: Images should be (num_frames, H, W) for CNN processing
        self.observation_space = gym.spaces.Dict({
            "images": gym.spaces.Box(
                low=0, high=1, 
                shape=(self.num_frames, img_shape[0], img_shape[1]),  # Changed order
                dtype=np.float32
            ),
            "actions": gym.spaces.Box(
                low=-1, high=1, 
                shape=(self.num_actions,),
                dtype=np.float32
            ),
        })
        
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        # Extract simulation parameters
        self.sampling = config['sampling']
        self.inner_angle = config['inner_angle']
        self.outer_angle = config['outer_angle']
        self.crop_size = config['crop_size']
        self.center_x = config['center_x']
        self.center_y = config['center_y']
        self.dose_rate = config.get('dose_rate', None)
        self.interpolation_sampling = config['interpolation_sampling']
        
        # Focus metric configuration
        self.focus_metric_func = config['focus_metric']['func']
        self.focus_metric_args = config['focus_metric']['args']
        
        # History buffers
        self.image_buffer = None
        self.action_buffer = None
        
        # State tracking
        self.state = 100  # Current defocus value
        self.step_count = 0
        self._last_metric = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state
        self.step_count = 0
        self.state = np.random.uniform(-50, 50) - 25  # Random initial defocus
        
        # Get initial observation with no action (action=0)
        obs = self._forward_model(0)
        self._last_metric = self._focus_metric(obs)
        
        # Initialize buffers with the same initial image
        self.image_buffer = [obs[:,:,0]] * self.num_frames  # Remove channel dim
        self.action_buffer = [0.0] * self.num_actions

        self.center_x = self.config["center_x"] + int(np.random.rand(1) * 20 - 10)  # randomize starting offset
        self.center_y = self.config["center_y"] + int(np.random.rand(1) * 20 - 10)
        
        # Stack images properly: (num_frames, H, W)
        observation = {
            "images": np.stack(self.image_buffer, axis=0),
            "actions": np.array(self.action_buffer, dtype=np.float32),
        }
        
        info = {"focus_metric": self._last_metric}
        return observation, info
    
    def step(self, action):
        # Clip action to valid range
        action_scalar = np.clip(action[0], a_min=-1, a_max=1)
        
        # Simulate STEM image with new action
        obs = self._forward_model(action_scalar)
        
        # Calculate focus metric (internal, not given to agent)
        focus_metric = self._focus_metric(obs)
        
        # Calculate reward based on improvement
        improvement = focus_metric - self._last_metric
        raw_reward = 10 * improvement + focus_metric
        reward = np.tanh(raw_reward)  # Smooth clipping
        
        # Update history buffers (FIFO - First In First Out)
        self.image_buffer.pop(0)  # Remove oldest image
        self.image_buffer.append(obs[:,:,0])  # Add new image (remove channel dim)
        
        self.action_buffer.pop(0)  # Remove oldest action
        self.action_buffer.append(float(action_scalar))  # Add new action
        
        # Create observation: stack images as (num_frames, H, W)
        observation = {
            "images": np.stack(self.image_buffer, axis=0),
            "actions": np.array(self.action_buffer, dtype=np.float32),
        }
        
        # Update internal state
        self._last_metric = focus_metric
        self.state = self._compute_new_state(action_scalar)
        self.step_count += 1
        
        # Check termination
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        info = {
            "focus_metric": focus_metric, 
            "improvement": improvement,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _forward_model(self, action):
        """Simulate STEM image with current defocus + action."""
        # Update defocus based on action
        defocus_change = action * 10.0  # Scale action to reasonable defocus change
        new_defocus = self.state + defocus_change
        
        # Prepare aberration dictionary
        aberration_dict = {"C10": new_defocus}
        
        # Run STEM simulation
        img = simulate_haadf(
            system=self.system,
            sampling=self.sampling,
            inner_angle=self.inner_angle,
            outer_angle=self.outer_angle,
            semi_angle_cutoff=30.0,
            crop_size=self.crop_size,
            center_x=self.center_x,
            center_y=self.center_y,
            aberation_dict=aberration_dict,
            dose_rate=self.dose_rate,
            interpolation_sampling=self.interpolation_sampling
        )
        
        # Normalize image to [0, 1]
        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Add channel dimension
        obs = img_normalized[:, :, np.newaxis]
        
        return obs.astype(np.float32)
    
    def _focus_metric(self, image):
        """Calculate focus quality metric."""
        img_2d = image[:, :, 0]
        metric = self.focus_metric_func(img_2d[None], self.focus_metric_args)
        return float(metric)
    
    def _compute_new_state(self, action):
        """Compute new defocus state after taking action."""
        defocus_change = action * 5.0
        new_state = self.state + defocus_change
        new_state = np.clip(new_state, -50, 50)  # Limit to ±50 nm
        return new_state



# Custom CNN model for dict observations with image history and action history
class CNNModelWithHistory(TorchModelV2, nn.Module):
    """
    Custom model that processes:
    - images: (batch, num_frames, H, W) - last 2 images stacked
    - actions: (batch, 2) - last 2 actions
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # Get observation space shapes
        self.image_shape = obs_space.spaces["images"].shape  # (num_frames, H, W)
        self.action_history_size = obs_space.spaces["actions"].shape[0]  # 2
        
        # CNN for processing stacked images
        # Input: (batch, num_frames, H, W)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.image_shape[0], 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # Calculate the flattened size after conv layers
        dummy_input = torch.zeros(1, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        conv_out = self.conv_layers(dummy_input)
        self.conv_out_size = int(np.prod(conv_out.shape[1:]))
        
        # MLP for processing action history
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_history_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # Combined feature processing
        self.latent_dim = 256
        combined_size = self.conv_out_size + 32  # CNN features + action features
        
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_size, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
        )
        
        # Value function head (required by interface)
        self._value_branch = nn.Linear(self.latent_dim, 1)
        
        self._latent = None
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Extract dict observation components
        obs = input_dict["obs"]
        images = obs["images"].float()  # (batch, num_frames, H, W)
        actions = obs["actions"].float()  # (batch, 2)
        
        # Images are already in (batch, C, H, W) format - no permute needed!
        
        # Process images through CNN
        conv_features = self.conv_layers(images)
        conv_features = conv_features.reshape(conv_features.size(0), -1)
        
        # Process action history through MLP
        action_features = self.action_mlp(actions)
        
        # Concatenate CNN and action features
        combined = torch.cat([conv_features, action_features], dim=1)
        
        # Process combined features
        latent = self.fc_combined(combined)
        
        # Store for value function
        self._latent = latent
        
        return latent, state
    
    @override(TorchModelV2)
    def value_function(self):
        assert self._latent is not None, "must call forward() first"
        return self._value_branch(self._latent).squeeze(1)
