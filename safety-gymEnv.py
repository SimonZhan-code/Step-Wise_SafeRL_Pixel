class SafetyGymEnv:
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        import safety_gym
        self.symbolic = symbolic
        self._env = gym.make(env)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self._env.unwrapped.vision_size = (64, 64)
        self._env.unwrapped.observe_vision = True
        self._env.unwrapped.vision_render = False
        # obs_vision_swap = self._env.unwrapped.obs_vision

    def render(self):
        image = Image.fromarray(np.array(self._env.unwrapped.obs_vision() * 255, dtype=np.uint8,
                                             copy=False))
        image = np.asarray(ImageOps.flip(image))
        # image = Image.fromarray(image)
        return image

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self.render(), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(self.render(), self.bit_depth)
        return observation, reward, done


    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


