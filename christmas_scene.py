from utils import *
from ray import *
from cli import render
from PIL import Image

"""
STILL TODO: 

1. make the tree look better / add a cone primitive
2. add more details to the snowman
3. make a crescent moon in the nightsky
4. add some cube-shaped "presents" around the tree
5. add christmas ornaments to the tree

"""
load_img_path = './ornament.png'
img = Image.open(load_img_path)
im = np.array(img)

load_moon = './moon.png'
moon = np.array(Image.open(load_moon))

tree_texture = np.array(Image.open('./pine-tree-texture.png'))

silver = np.array(Image.open('./silver.png'))

a = np.log(np.array([0.4, 0.4, 0.2])**(-1))
crystal = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90,
                   k_m=0.3, di=True, n=1.5, a=a)
blue = Material(vec([0.2, 0.2, 0.5]))
gray = Material(vec([0.2, 0.2, 0.2]))
red = Material(vec([0.8, 0.2, 0.2]))
green = Material(vec([0.02, 0.5, 0.02]))
tree = Material(vec([0.01, 0.1, 0.01]))
brown = Material(vec([0.1, 0.1, 0.1]))
gold = Material(vec([.5, .5, 0.1]), k_m=0.5)
ground = Material(vec([1, 1, 1.5]))
snow = Material(vec([1, 1, 1.5]), k_m=0.2)
sparkle_red = Material(vec([0.2, 0.2, 0.2]), k_m=0.5, img=im)
moon = Material(vec([1, 1, 1.5]), img=moon)
silver = Material(vec([1, 1, 1]), k_m=0.5, img=silver)

scene = Scene([
    # Christmas balls
    Sphere(vec([-.9, -0.4, -0.4]), 0.1, silver),
    Sphere(vec([-.8, -0.1, -.4]), 0.1, crystal),
    Sphere(vec([-0.6, 0.1, -.7]), 0.1, sparkle_red),
    Sphere(vec([-0.8, 0.3, -.3]), 0.1, crystal),
    Sphere(vec([-0.7, 0.4, -.7]), 0.1, silver),
    Sphere(vec([-0.5, 0.6, -.7]), 0.1, crystal),
    Sphere(vec([-0.3, -0.5, -.5]), 0.1, sparkle_red),
    Sphere(vec([-0.9, 0.8, -1]), 0.1, silver),
    Sphere(vec([-1, 0.9, -1]), 0.1, crystal),
    Sphere(vec([-0.9, 0.9, -0.5]), 0.1, sparkle_red),
    Sphere(vec([-1.2, -0.7, -0.4]), 0.1, silver),
    Sphere(vec([-0.3, -0.2, -0.5]), 0.1, silver),

    # Snowman :)
    Sphere(vec([0.5, -0.8, -1]), 0.2, snow),
    Sphere(vec([0.5, -0.5, -1]), 0.15, snow),
    Sphere(vec([0.5, -0.3, -1]), 0.1, snow),
    Sphere(vec([0.5, -0.22, -1]), 0.05, red),
    Sphere(vec([0.5, -0.275, -0.9]), 0.01, brown),
    Sphere(vec([0.6, -0.275, -0.9]), 0.01, brown),

    # Moon
    Sphere(vec([-13, 0, -30]), 1, moon),

    # Tree Code
    Sphere(vec([-1, -0.7, -1]), 0.3, brown),
    Sphere(vec([-1, -0.6, -1]), 0.55, tree),
    Sphere(vec([-1, -0.5, -1]), 0.53, tree),
    Sphere(vec([-1, -0.45, -1]), 0.52, tree),
    Sphere(vec([-1, -0.4, -1]), 0.51, gold),
    Sphere(vec([-1, -0.3, -1]), 0.49, tree),
    Sphere(vec([-1, -0.1, -1]), 0.47, tree),
    Sphere(vec([-1, 0, -1]), 0.45, tree),
    Sphere(vec([-1, 0.1, -1]), 0.44, tree),
    Sphere(vec([-1, -0.25, -1]), 0.435, tree),
    Sphere(vec([-1, 0.2, -1]), 0.43, gold),
    Sphere(vec([-1, 0.3, -1]), 0.42, tree),
    Sphere(vec([-1, 0.4, -1]), 0.41, tree),
    Sphere(vec([-1, 0.5, -1]), 0.37, gold),
    Sphere(vec([-1, 0.6, -1]), 0.35, tree),
    Sphere(vec([-1, 0.7, -1]), 0.32, tree),
    Sphere(vec([-1, 0.8, -1]), 0.29, tree),

    # Christmas Tree Topper
    Sphere(vec([-1, 1.1, -1]), 0.1, gold),

    # Ground floor
    Sphere(vec([0, -201, 0]), 200, ground),
],
    bg_color=vec([0.001, 0.01, 0.1])
)

lights = [
    PointLight(vec([12, 10, 5]), vec([10, 10, 10])),
    PointLight(vec([-13, 0, -28]), vec([1000, 1000, 1])),
    PointLight(vec([-1, 1.27, -1]), vec([1, 1, 1])),
    AmbientLight(0.1),
]

camera = Camera(vec([3, 1, 5]), target=vec(
    [0, 0.4, 0]), vfov=24, aspect=9/9)

render(camera, scene, lights)
