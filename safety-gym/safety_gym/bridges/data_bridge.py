from dm_control.mujoco.wrapper import MjData, MjModel
from mujoco_py.builder import cymj

class DataBridge(cymj.PyMjData):
    def __init__(self, data: MjData, model: MjModel):
        self._data = data
        self._model = model


    def get_body_xpos(self, name):
        id = self._model.name2id(name, 'body')
        return self._data.xpos[id]

    def get_body_xmat(self, name):
        id = self._model.name2id(name, 'body')
        return self._data.xmat[id].reshape([3, 3])

    def get_body_xvelp(self, name):
        id = self._model.name2id(name, 'body')
        vel = self._data.object_velocity(id, 'body')
        return vel[0]

    @property
    def sensordata(self):
        return self._data.sensordata

    @property
    def contact(self):
        return self._data.contact

    @property
    def ncon(self):
        return self._data.ncon

    @property
    def ctrl(self):
        return self._data.ctrl

    @property
    def time(self):
        return self._data.time

    def set_mocap_pos(self, name, value):
        body_id = self._model.name2id(name, 'body')
        mocap_id = self._model.body_mocapid[body_id]
        self._data.mocap_pos[mocap_id] = value





