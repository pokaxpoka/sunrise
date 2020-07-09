import numpy as np
import torch 
from rlkit.torch import pytorch_util as ptu

def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(
        env,
        agent,
        noise_flag=0,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        if noise_flag == 1:
            r += np.random.normal(0,1,1)[0]
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def ensemble_rollout(
        env,
        agent,
        num_ensemble,
        noise_flag=0,
        max_path_length=np.inf,
        ber_mean=0.5,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    masks = [] # mask for bootstrapping
    o = env.reset()
    en_index = np.random.randint(num_ensemble)
    agent[en_index].reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent[en_index].get_action(o)
        next_o, r, d, env_info = env.step(a)
        if noise_flag == 1:
            r += np.random.normal(0,1,1)[0]
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        mask = torch.bernoulli(torch.Tensor([ber_mean]*num_ensemble))
        if mask.sum() == 0:
            rand_index = np.random.randint(num_ensemble, size=1)
            mask[rand_index] = 1
        mask = mask.numpy()
        masks.append(mask)
  
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)
    
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    masks = np.array(masks)

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        masks=masks,
    )


def get_ucb_std(obs, policy_action, inference_type, critic1, critic2, 
                feedback_type, en_index, num_ensemble):
    obs = ptu.from_numpy(obs).float()
    policy_action = ptu.from_numpy(policy_action).float()
    obs = obs.reshape(1,-1)
    policy_action = policy_action.reshape(1,-1)
    
    if feedback_type == 0 or feedback_type==2:
        with torch.no_grad():
            target_Q1 = critic1[en_index](obs, policy_action)
            target_Q2 = critic2[en_index](obs, policy_action)
        mean_Q = 0.5*(target_Q1.detach() + target_Q2.detach())
        var_Q = 0.5*((target_Q1.detach() - mean_Q)**2 + (target_Q2.detach() - mean_Q)**2)
        ucb_score = mean_Q + inference_type * torch.sqrt(var_Q).detach()

    elif feedback_type == 1 or feedback_type==3:
        mean_Q, var_Q = None, None
        L_target_Q = []
        for en_index in range(num_ensemble):
            with torch.no_grad():
                target_Q1 = critic1[en_index](obs, policy_action)
                target_Q2 = critic2[en_index](obs, policy_action)
                L_target_Q.append(target_Q1)
                L_target_Q.append(target_Q2)
                if en_index == 0:
                    mean_Q = 0.5*(target_Q1 + target_Q2) / num_ensemble
                else:
                    mean_Q += 0.5*(target_Q1 + target_Q2) / num_ensemble

        temp_count = 0
        for target_Q in L_target_Q:
            if temp_count == 0:
                var_Q = (target_Q.detach() - mean_Q)**2
            else:
                var_Q += (target_Q.detach() - mean_Q)**2
            temp_count += 1
        var_Q = var_Q / temp_count
        ucb_score = mean_Q + inference_type * torch.sqrt(var_Q).detach()
        
    return ucb_score
    
def ensemble_ucb_rollout(
        env,
        agent,
        critic1,
        critic2,
        inference_type,
        feedback_type,
        num_ensemble,
        noise_flag=0,
        max_path_length=np.inf,
        ber_mean=0.5,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    masks = [] # mask for bootstrapping
    o = env.reset()
    for en_index in range(num_ensemble):
        agent[en_index].reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
        
    while path_length < max_path_length:
        a_max, ucb_max, agent_info_max = None, None, None
        for en_index in range(num_ensemble):
            _a, agent_info = agent[en_index].get_action(o)
            ucb_score = get_ucb_std(o, _a, inference_type, critic1, critic2,
                                    feedback_type, en_index, num_ensemble)
            
            if en_index == 0:
                a_max = _a
                ucb_max = ucb_score
                agent_info_max = agent_info
            else:
                if ucb_score > ucb_max:
                    ucb_max = ucb_score
                    a_max = _a
                    agent_info_max = agent_info

        next_o, r, d, env_info = env.step(a_max)
        if noise_flag == 1:
            r += np.random.normal(0,1,1)[0]
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a_max)
        agent_infos.append(agent_info_max)
        env_infos.append(env_info)
        mask = torch.bernoulli(torch.Tensor([ber_mean]*num_ensemble))
        if mask.sum() == 0:
            rand_index = np.random.randint(num_ensemble, size=1)
            mask[rand_index] = 1
        mask = mask.numpy()
        masks.append(mask)
  
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)
    
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    masks = np.array(masks)

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        masks=masks,
    )


def ensemble_eval_rollout(
        env,
        agent,
        num_ensemble,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    for en_index in range(num_ensemble):
        agent[en_index].reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a = None
        for en_index in range(num_ensemble):
            _a, agent_info = agent[en_index].get_action(o)
            if en_index == 0:
                a = _a
            else:
                a += _a
        a = a / num_ensemble
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
