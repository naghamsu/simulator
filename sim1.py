import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import Circle, Wedge, PathPatch

class Ball:
    def __init__(self, x=0.0, y=0.0, mass=1.0, radius=0.2, v_x=0.0, v_y=0.0, omega = 0.0, color = "red"):
        self.color = color
        self.radius = radius
        self.mass = mass
        self.inertia = 0.4 * self.mass * self.radius**2

        # state
        self.pos = np.array([x, y])
        self.v = np.array([v_x, v_y])
        self.omega = omega

class Plane:
    def __init__(self, y=0.0, angle=0.0):
        # y = kx + b
        self.y = y
        self.angle = angle

class PhysicsEngine:
    def __init__(self, gravity=-9.8):
        self.gravity = np.array([0, gravity])
        self.resistance_k = 0.1

    def update(self, objects, plane, dt):
        for obj in objects:
            # phycics
            Fg = self.gravity
            F_rolling = - 0.7 * obj.mass * np.sin(plane.angle) * self.gravity
            F = Fg + F_rolling
            dv = F / obj.mass

            M = - self.resistance_k * np.sign(obj.omega) * obj.omega
            domega = M / obj.inertia

            # integration
            obj.v += dv * dt
            obj.pos += obj.v * dt
            obj.omega += domega * dt

            obj = self._collision_resolve(obj, plane)

    def _collision_resolve(self, object, plane):
        k = np.tan(plane.angle)
        d = np.abs(k * object.pos[0] - object.pos[1] + plane.y) / np.sqrt(1 + k**2)

        if d - object.radius  < 0.01:
            n = np.array([-k, 1.0]) / np.sqrt(1 + k**2)
            t = np.array([1.0, k]) / np.sqrt(1 + k**2)

            e = 0.95
            Jn = - (1 + e) * object.mass * np.dot(object.v, n)

            mu = 0.1
            v_ct = np.dot(object.v, t) + object.omega * object.radius
            Jt = -mu * Jn * np.sign(v_ct)

            object.v += (Jn * n + Jt * t) / object.mass
            object.omega += object.radius * Jt / object.inertia

            object.pos += (object.radius - d) * n

        return object

class Renderer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-100, 500)
        self.ax.set_ylim(-50, 50)
        self.ax.set_aspect('equal')
        self.list_obj_patch = []
        self.plane_patch = None

    def update(self, objects, plane, dt=0.0001):
        if not self.list_obj_patch:
            for obj in objects:
                patch = Circle(obj.pos, obj.radius, color=obj.color)
                self.ax.add_patch(patch)
                self.list_obj_patch.append(patch)
        else:
            for obj, obj_to_update in zip(objects, self.list_obj_patch):
                obj_to_update.center = obj.pos

        if self.plane_patch is None:
            x_plane = np.array([-100, 500])
            y_plane = np.tan(plane.angle) * x_plane + plane.y 
            self.plane_patch, = self.ax.plot(x_plane, y_plane, 'k-')

        plt.pause(dt)

class World:
    def __init__(self, physics, renderer):
        self.physics = physics
        self.renderer = renderer
        self.plane = None
        self.objects = []
        self.time = 0
        self.dt = 0.01

    def set_plane(self, plane):
        self.plane = plane

    def add_object(self, obj):
        self.objects.append(obj)

    def step(self, i):
            self.physics.update(self.objects, self.plane, self.dt)
            if i % 10 == 0:
                self.renderer.update(self.objects, self.plane)
            self.time += self.dt

    def run(self, steps):
        for i in range(steps):
            self.step(i)
            # time.sleep(0.01)


def main():

    plane = Plane(y=-5.0, angle=-0.3)

    physics = PhysicsEngine(gravity=-9.81)
    renderer = Renderer()

    world = World(physics, renderer)

    world.set_plane(plane)

    for i in range(10):
        ball = Ball(x=10.0 * np.random.rand(), y=30.0 * np.random.rand(), radius=1 * np.random.rand(), v_y=np.random.rand(), omega=10 * np.random.rand(), color="red")
        world.add_object(ball)

    world.run(5000)

if __name__=="__main__":
    main()
