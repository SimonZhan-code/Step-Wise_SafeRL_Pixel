from dm_control.mujoco.wrapper import MjModel
from mujoco_py.builder import cymj

class ModelBridge(cymj.PyMjModel):
    def __init__(self, model: MjModel):
        self._model = model


    def body_name2id(self, name):
        return self._model.name2id(name, 'body')

    def sensor_name2id(self, name):
        return self._model.name2id(name, 'sensor')

    def geom_id2name(self, id):
        return self._model.id2name(id, 'geom')

    def geom_name2id(self, name):
        return self._model.name2id(name, 'geom')

    @property
    def sensor_adr(self):
        return self._model.sensor_adr

    @property
    def sensor_dim(self):
        return self._model.sensor_dim

    @property
    def body_pos(self):
        return self._model.body_pos

    @property
    def actuator_ctrlrange(self):
        return self._model.actuator_ctrlrange



