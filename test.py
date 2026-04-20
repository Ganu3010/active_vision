import shapenet_gym.env
import cv2 as cv

env = shapenet_gym.env.ShapeNetViewEnv(
    dataset_root="data\ShapeNetCore",
    max_steps=20,
    image_size=(224, 224),
    theta_step_deg=15.0,
    phi_step_deg=15.0,
    offscreen=False
)

obs, info = env.reset()
print(info['category'])
print(obs.shape)
cv.imshow('Sample', obs)
cv.waitKey(0)