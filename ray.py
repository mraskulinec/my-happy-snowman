import numpy as np
from math import *
from utils import *
import random

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

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, di=False, n=1, a=None, img=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
          n : float -- the refractive index
          a : float -- attenuation constant
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d
        self.n = n
        self.di = di
        self.a = a
        self.img = img


class Hit:

    def __init__(self, t, point=None, normal=None, material=None, surf=None):
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
        self.surf = surf


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


# def homogenous(x):
#     return np.array([x[0], x[1], x[2], 1])


# def non_homogenous(x):
#     return np.array([x[0], x[1], x[2]])


# def homogenous3(x):
#     col = np.array([[0.], [0.], [0.]])
#     row = np.array([0, 0, 0, 1])
#     m = np.append(x, col, 1)
#     m = np.append(m, row, 0)
#     return m


# def homogenize(x):
#     return x / x[3]


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

    def texture(self):
        print("test")

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        # (x, y, z) = (self.center)
        # to_center = np.array(
        #     [[1, 0, 0, -x], [0, 1, 0, -y], [0, 0, 1, -z], [0, 0, 0, 1]])
        # away_center = np.array(
        #     [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        # new_m = homogenous3(self.m)
        # transform = away_center @ (new_m @ to_center)

        # m_inv = np.linalg.inv(self.m)
        # new_ray = Ray(m_inv @ ray.origin, m_inv @
        #               ray.direction, ray.start, ray.end)
        new_ray = ray
        ec = new_ray.origin - self.center
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
            t1_out = t1 <= new_ray.start or t1 >= new_ray.end
            t2_out = t2 <= new_ray.start or t2 >= new_ray.end
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
        # normal = normalize(np.transpose(np.linalg.inv(self.m))@normal)
        return Hit(t, point, normal, self.material, self)


class Cone:

    def __init__(self, c, v, theta, h,  material):
        """Create a cone from the given circle center with radius r and height h.

        Parameters:
          c : (3,) -- the coordinate of the tip of the cone.
          v : (3,) -- a unit vector representing the cone's axis in the direction of increasing radius.
          theta : float -- half-angle between the cone's axis and surface.
          h : float -- height of the cone from base to tip.
          material : Material -- the material of the surface.
        """
        self.c = c
        self.v = v
        self.theta = theta
        self.h = h
        self.material = material

      # For ray-cone intersection formula, look here:
      # https://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone/#:~:text=If%20%2C%20the%20ray%20is%20intersecting,%E2%88%92%20b%20%2B%20%CE%94%202%20a%20.

    def intersect(self, ray):
        """Computes the interesection between a ray and this cone, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with this cone
        Return:
          Hit -- the hit data
        """
        d = ray.direction
        oc = ray.origin-self.c
        a = np.dot(d, self.v)**2 - cos(self.theta)**2
        b = 2*(np.dot(d, self.v)*np.dot((oc), self.v) -
               np.dot(oc, d)*(cos(self.theta)**2))
        c = (np.dot(oc, self.v)**2) - np.dot(oc, oc)*(cos(self.theta)**2)
        discriminant = b**2 - 4*a*c
        t = 0
        if discriminant <= 0:
            return no_hit
        else:
            t1 = (-b + sqrt(discriminant))/(2*a)
            t2 = (-b - sqrt(discriminant))/(2*a)
            t1_out = (t1 <= ray.start) or (t1 >= ray.end)
            t2_out = (t2 <= ray.start) or (t2 >= ray.end)
            point1 = ray.origin + t1*d
            point2 = ray.origin + t2*d
            proj1 = np.dot(point1-self.c, self.v)*self.v / \
                sqrt(np.dot(self.v, self.v))
            proj2 = np.dot(point2-self.c, self.v)*self.v / \
                sqrt(np.dot(self.v, self.v))
            height1 = sqrt(np.dot(proj1, proj1))
            height2 = sqrt(np.dot(proj2, proj2))

            t1_out = t1_out or (height1 >= self.h)
            t2_out = t2_out or (height2 >= self.h)
            if t1_out and t2_out:
                return no_hit
            elif t1_out:
                t = t2
            elif t2_out:
                t = t1
            else:
                t = min(t1, t2)
        point = ray.origin + t*d
        tangent = np.cross(point-self.c, self.v)
        normal = normalize(np.cross(point-self.c, tangent))
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
        # m_inv = np.linalg.inv(self.m)
        # ray = Ray(m_inv @ ray.origin, m_inv @
        #           ray.direction, ray.start, ray.end)
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
        # normal = normalize(np.transpose(np.linalg.inv(self.m))@normal)
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
          surf : the surface that the hit is on
        Return:
          (3,) -- the light reflected from the surface
        """
        # Compute Diffuse Lighting
        im = hit.material.img
        if im is None:
            k_d = hit.material.k_d
        else:
            # only for spheres
            (x, y, z) = hit.point - hit.surf.center
            i = (pi + atan2(y, x))/(2*pi)
            radius = sqrt(x**2 + y**2 + z**2)
            j = (pi - acos(z/radius))/pi
            img_i = int(i * im.shape[0])
            img_j = int(j * im.shape[1])
            k_d = im[img_i, img_j]/255
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
          surf : the surface that the hit is on
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        im = hit.material.img
        k_a = hit.material.k_a
        if im is not None:
            # only for spheres
            (x, y, z) = hit.point - hit.surf.center
            i = (pi + atan2(y, x))/(2*pi)
            radius = sqrt(x**2 + y**2 + z**2)
            j = (pi - acos(z/radius))/pi
            img_i = int(i * im.shape[0])
            img_j = int(j * im.shape[1])
            k_a = im[img_i, img_j]/255
        return k_a*self.intensity


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
          Hit, surf -- the hit data and the surface
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


def refract(d, n, nt):
    tir = 1 - (1-np.dot(d, n)**2)/(nt**2)
    if tir < 0:
        return (False, np.array([0., 0., 0.]))
    else:
        return (True, (d-n*np.dot(d, n))/nt - n*np.sqrt(tir))


def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      surf : the surface that the hit is on
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

    # refraction
    if hit.material.di:
        d = normalize(ray.direction)
        n = hit.normal
        r = d - 2*np.dot(d, n)*n
        c = 0
        k = 0
        nt = hit.material.n
        if np.dot(d, n) < 0:
            t = refract(d, n, nt)[1]
            c = -np.dot(d, n)
            k = np.array([1., 1., 1.])
        else:
            # some values of k are inf? (negative t)
            k = e**(-hit.material.a*hit.t)
            result = refract(d, -n, 1/nt)
            t = result[1]
            if result[0]:
                c = np.dot(t, n)
            else:
                new_ray = Ray(hit.point, t, 5e-5, np.inf)
                p = scene.intersect(new_ray)
                return k*shade(new_ray, p, scene, lights, depth+1)

        ref_i = ((nt-1)**2)/((nt+1)**2)
        ref = ref_i + (1-ref_i)*((1-c)**5)
        ref_ray = Ray(hit.point, r, 5e-5, np.inf)
        ref_p = scene.intersect(ref_ray)
        ref_color = shade(ref_ray, ref_p, scene, lights, depth+1)

        trans_ray = Ray(hit.point, t, 5e-5, np.inf)
        trans_p = scene.intersect(trans_ray)
        trans_color = shade(trans_ray, trans_p, scene, lights, depth+1)
        return k*(ref*ref_color+(1-ref)*trans_color)
    # else:
    #     # mirror reflection
    #     d = ray.direction
    #     n = hit.normal
    #     r = d - 2*np.dot(d, n)*n
    #     new_ray = Ray(hit.point, r, 5e-5, np.inf)
    #     p = scene.intersect(new_ray)
    #     return sum + hit.material.k_m*shade(new_ray, p, scene, lights, depth+1)
    return sum


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
        if False:
            img[i, :] = np.array([0., 0., 0.])
        else:
            for j in range(img.shape[1]):
                if True:
                    c = 0
                    n = 1
                    for p in range(n):
                        for q in range(n):
                            e = random.random()
                            z = np.array([(j+(p+e)/n)/nx, (i+(p+e)/n)/ny])
                            r = camera.generate_ray(z)
                            point = scene.intersect(r)
                            c += shade(r, point, scene, lights)
                    img[i, j] = c/(n**2)
                else:
                    img[i, j] = np.array([0., 0., 0.])
    return img
