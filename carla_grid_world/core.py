import carla
from gym.spaces import Discrete
from gym.space import Space
import numpy as np
import json
from functools import partial

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return { '__class__': 'np.ndarray',
                    '0': obj.tolist(),
                    '1': obj.dtype.name }
        return json.JSONEncoder.default(self, obj)

class NumpyDecoder(json.JSONDecoder):
    def object_hook(self, dct):
        if dct.get('__class__') == 'np.ndarray':
            return np.array(dct['0'], dct['1'])
        return dct

class PointCloudSpace(Space):
    def __init__(self, mins=[-30, -30, -0.01], maxs=[30, 30, 0.01], D=3):
        self.D = D
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

    def sample(self):
        return self.np_random.rand(self.shape)

    def contains(self, x):
        return (True
                if (isinstance(x, np.ndarray)
                    and x.shape[-1] == D
                    and (x >= self.mins)[:].all()
                    and (x < self.maxs)[:].all())
                else False)

    def to_jsonable(self, x):
        return NumpyEncoder().default(x)

    def from_jsonable(self, x):
        return NumpyDecoder().object_hook(x)

custom_blue_prints = {
        'sensor.camera.semantic_segmentation' : dict(
            image_size_x= '1920',
            image_size_y= '1080',
            fov='90'),
        'sensor.lidar.ray_cast': dict(
            channels=1,
            upper_fov=0,
            lower_fov=0)
        }

class CarlaGridWorld:
    action_space = Discrete(7)
    observation_space = PointCloudSpace()
    def __init__(self, host="localhost", port=2000,
            rng=None):
        self.host = host
        self.port = port
        self._client = None
        self._world = None
        self.sensor = dict()
        self.random = np.random.RandomState() if rng is None else rng
        self.vehicle = None

    def world(self):
        if self._world is None:
            self._client = carla.Client(host, port)
            self._world = self.client.get_world()
        return self._world

    def spawn_actor(self, transform, name, attach_to, blueprint,
            sensor_hook=None,
            blue_prints_overrides=custom_blue_prints):
        semcam = self.world().get_blueprint_library().find(blueprint)
        for name, value in blue_prints_overrides.get(blueprint, {}):
            semcam.set_attribute(name, value)
        self.sensors[name] = self.world().spawn_actor(semcam, transform,
                attach_to=attach_to)
        if sensor_hook:
            self.sensors[name].listen(sensor_hook)

    add_semantic_camera = partial(spawn_actor,
            blueprint='sensor.camera.semantic_segmentation')

    def add_vehicle(self, vehicle_blue_print_filter='vehicle.bmw.*'):
        world = self.world()
        random = self.random
        vehicle_bp = random.choice(
                world.get_blueprint_library().filter(vehicle_blue_print_filter))
        transform = random.choice(
                world.get_map().get_spawn_points())
        self.vehicle = world.try_spawn_actor(vehicle_bp, transform)
        return self.vehicle

    def add_vehicle_with_sensors(self, sensors=[ {
        'blueprint': 'sensor.camera.semantic_segmentation',
        'transform': carla.Transform(
            location=carla.Location(x=0.8, z=1.7),
            rotation=carla.Rotation(yaw=0))
        },
        {
            'blueprint': 'sensor.camera.semantic_segmentation',
            'transform': carla.Transform(
                location=carla.Location(x=0.8, z=1.7),
                rotation=carla.Rotation(yaw=90))
            },
        {
            'blueprint': 'sensor.camera.semantic_segmentation',
            'transform': carla.Transform(
                location=carla.Location(x=0.8, z=1.7),
                rotation=carla.Rotation(yaw=180))
            },
        {
            'blueprint': 'sensor.camera.semantic_segmentation',
            'transform': carla.Transform(
                location=carla.Location(x=0.8, z=1.7),
                rotation=carla.Rotation(yaw=270))
            },
        {
            'blueprint': 'sensor.other.collision',
            'transform': carla.Transform() },
        {
            'blueprint': 'sensor.lidar.ray_cast',
            'transform': carla.Transform(
                location=carla.Location(x=0.8, z=1.7),
                rotation=carla.Rotation(yaw=0)) }
            ]):
        vehicle = self.add_vehicle()
        for i, sensor in enumerate(sensors):
            self.spawn_actor(
                    name='%s.%d' % (sensor['blueprint'], i)
                    attach_to=vehicle,
                    **sensor)


    def _get_snapshot(self):
        world_snapshot = self.world().wait_for_tick()
        return world_snapshot

    def reset(self):
        vehicle = self.add_vehicle()

        return self._get_snapshot()

    def step(self, action):
        # 1. Take action
        obs = self._get_snapshot()
        return obs, reward, False, dict()
