# Active Vision
This project aims to develop an RL agent to determine the best viewing angle to reduce ambiguity for vision based tasks. 

## Sample use for shapenet_gym
```
env = shapenet_gym.env.ShapeNetViewEnv(
    dataset_root="data\ShapeNetCore",
    max_steps=20,
    image_size=(224, 224),
    theta_step_deg=15.0,
    phi_step_deg=15.0,
    offscreen=False
)
```