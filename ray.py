import numpy as np
from math import *
from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        ec = ray.origin - self.center
        d = ray.direction
        r = self.radius
        discriminant = np.dot(d, ec)**2-np.dot(d, d)*(np.dot(ec, ec)-r**2)
        t = 0
        if discriminant < 0:
            return no_hit
        elif discriminant == 0:
            # t = -np.dot(d, ec)/np.dot(d, d)
            # if t <= ray.start or t >= ray.end:
            #     return no_hit
            return no_hit
        else:
            t1 = (-np.dot(d, ec)+sqrt(discriminant))/np.dot(d, d)
            t2 = (-np.dot(d, ec)-sqrt(discriminant))/np.dot(d, d)
            t1_out = t1 <= ray.start or t1 >= ray.end
            t2_out = t2 <= ray.start or t2 >= ray.end
            if t1_out and t2_out:
                return no_hit
            elif t1_out:
                t = t2
            elif t2_out:
                t = t1
            else:
                t = min(t1, t2)
        point = ray.origin + t*d
        pc = point - self.center
        normal = normalize(pc)
        return Hit(t, point, normal, self.material)


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        a = self.vs[0, 0] - self.vs[1, 0]
        b = self.vs[0, 1] - self.vs[1, 1]
        c = self.vs[0, 2] - self.vs[1, 2]
        d = self.vs[0, 0] - self.vs[2, 0]
        e = self.vs[0, 1] - self.vs[2, 1]
        f = self.vs[0, 2] - self.vs[2, 2]
        g = ray.direction[0]
        h = ray.direction[1]
        i = ray.direction[2]
        j = self.vs[0, 0] - ray.origin[0]
        k = self.vs[0, 1] - ray.origin[1]
        l = self.vs[0, 2] - ray.origin[2]
        m = a*(e*i-h*f) + b*(g*f-d*i) + c*(d*h-e*g)

        t = -(f*(a*k-j*b)+e*(j*c-a*l)+d*(b*l-k*c))/m
        if t < ray.start or t > ray.end:
            return no_hit
        gamma = (i*(a*k-j*b)+h*(j*c-a*l)+g*(b*l-k*c))/m
        if gamma < 0 or gamma > 1:
            return no_hit
        beta = (j*(e*i-h*f)+k*(g*f-d*i)+l*(d*h-e*g))/m
        if beta < 0 or beta > 1-gamma:
            return no_hit

        point = ray.origin + t*ray.direction
        normal = np.cross(self.vs[1]-self.vs[0], self.vs[2]-self.vs[0])
        return Hit(t, point, normal, self.material)


class Camera:

    def __init__(self, eye=vec([0, 0, 0]), target=vec([0, 0, -1]), up=vec([0, 1, 0]),
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        self.target = target
        w = eye-target
        v = up - np.dot(up, w)*w/np.dot(w, w)
        self.v = normalize(v)
        self.dist = np.sqrt(np.dot(w, w))
        self.height = 2*self.dist*tan((vfov/2)*pi/180)
        self.width = aspect * self.height
        self.w = w/self.dist
        self.u = np.cross(self.v, self.w)

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the lower left
                      corner of the image and (1,1) is the upper right
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        displ = 2*img_point - 1
        u_shift = self.u*displ[0]*self.width/2
        v_shift = self.v*displ[1]*self.height/2
        point = self.target + u_shift + v_shift
        return Ray(self.eye, point-self.eye)


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # Compute Diffuse Lighting
        k_d = hit.material.k_d
        dist = self.position-hit.point
        r_sq = np.dot(dist, dist)
        l = normalize(dist)
        n = normalize(hit.normal)
        i = (self.intensity / (r_sq))

        # Compute Specular Lighting
        k_s = hit.material.k_s
        p = hit.material.p
        v = -normalize(ray.direction)
        h = normalize(v+l)

        blocked = False
        t = (self.position - hit.point)[0]/l[0]
        r = Ray(hit.point, l, 1e-6, t)
        point = scene.intersect(r)
        if point != no_hit:
            blocked = True

        if blocked:
            return np.array([0., 0., 0.])
        else:
            return k_d * max(0, np.dot(n, l)) * i + k_s * (max(0, np.dot(n, h)) ** p) * i


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        return hit.material.k_a*self.intensity


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2, 0.3, 0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        intersections = []
        for i in self.surfs:
            intersections.append(i.intersect(ray))
        smallest = no_hit
        for i in intersections:
            if i.t < smallest.t:
                smallest = i
        return smallest


MAX_DEPTH = 4


def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # TODO A4 implement this function
    if depth > MAX_DEPTH:
        return np.array([0., 0., 0.])
    if hit == no_hit:
        return scene.bg_color
    sum = np.array([0., 0., 0.])
    for i in lights:
        sum += (i.illuminate(ray, hit, scene))

    d = ray.direction
    n = hit.normal
    r = d - 2*np.dot(d, n)*n
    new_ray = Ray(hit.point, r, 5e-5, np.inf)
    p = scene.intersect(new_ray)
    return sum + hit.material.k_m*shade(new_ray, p, scene, lights, depth+1)


def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    # TODO A4 implement this function
    img = np.zeros((ny, nx, 3), np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            z = np.array([j, i])
            r = camera.generate_ray(np.array([j/nx, i/ny]))
            p = scene.intersect(r)
            img[i, j] = shade(r, p, scene, lights)
    return img
