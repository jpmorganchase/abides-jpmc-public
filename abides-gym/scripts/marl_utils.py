import numpy as np
import gym
import os
from ray.rllib.policy.policy import PolicySpec
import ray.rllib.agents.ppo as ppo
from ray.tune import Analysis
market_obs_list = ['Quoted Price','Quoted Volume','Traded Price','Traded Volume','Spread','Depth','Agent PnLs']
mm_obs_list = ['Inventory','Cash','Spread','Depth','Spread PnL','Inventory PnL','Reward','Matching Agents']
pt_obs_list = ['Inventory','Cash','Distance to mid','Side','Reward','Momentum','Price History','Matching Agents']

def project_to_boundary(norm_state, observation_space, learning_agent_ids):
    for i in range(len(learning_agent_ids)):
        if not observation_space[learning_agent_ids[i]].contains(norm_state[learning_agent_ids[i]]):
            temp_state = norm_state[learning_agent_ids[i]]
            temp_state[temp_state > observation_space[learning_agent_ids[i]].high] = observation_space[learning_agent_ids[i]].high[temp_state > observation_space[learning_agent_ids[i]].high]
            temp_state[temp_state < observation_space[learning_agent_ids[i]].low] = observation_space[learning_agent_ids[i]].low[temp_state < observation_space[learning_agent_ids[i]].low]
            norm_state[learning_agent_ids[i]] = temp_state
    return norm_state

def normalizing_quantities_reward(num_pts, mm_max_depth, order_fixed_size):
    Z = {}
    Z['MM'] = {}
    Z['MM']['Spread PnL'] = mm_max_depth * 20 * order_fixed_size * 2
    Z['MM']['Inventory PnL'] = 1e5 * 100 * order_fixed_size
    for i in range(num_pts):
        Z[f'PT{i+1}'] = 100 * int(order_fixed_size/num_pts) * 1e5
    return Z

def normalizing_quantities_state(num_pts, L, M, num_state_features, add_volume, mm_max_depth,
order_fixed_size, pt_add_momentum):
    z_mm = np.zeros(num_state_features['MM'])
    z_mm[0 : 2 * (L + 1)] = 1e5 #quoted prices
    z_mm[2 * (L + 1) : 4 * (L + 1)] = order_fixed_size * mm_max_depth * 2 #quoted volumes
    z_mm[4 * L + 4] = 20 #market spread
    z_mm[4 * L + 5] = 100 #market depth
    z_mm[4 * L + 6 : 4 * L + 8] = 100 * order_fixed_size #MM inventory
    z_mm[4 * L + 8 : 4 * L + 10] = 100 * order_fixed_size * 1e5 #MM cash

    z_pt = np.ones((num_state_features['PT1'],num_pts))
    z_pt[0 : 4 * L + 10,:] = np.repeat(z_mm[0 : 4 * L + 10].reshape(-1,1),num_pts,1)
    z_pt[4 * L + 6 : 4 * L + 8,:] = 100 * int(order_fixed_size/num_pts) #PT inventory
    z_pt[4 * L + 8 : 4 * L + 10,:] = 100 * int(order_fixed_size/num_pts) * 1e5 #PT cash

    z_mm[4 * L + 10 : 4 * L + 10 + 2 * mm_max_depth] = order_fixed_size * mm_max_depth * 2 #MM exec vols
    if add_volume['MM']:
        vol_start_idx = 4 * L + 10 + 2 * mm_max_depth
        vol_end_idx = 4 * L + 10 + 2 * mm_max_depth + 2 * (M + 1)
        z_mm[vol_start_idx : vol_end_idx] = order_fixed_size * 2 #traded vols
        z_mm[vol_end_idx : ] = 1e5 #traded prices
    if add_volume['PT1']:
        vol_start_idx = 4 * L + 10 
        vol_end_idx = 4 * L + 10 + 2 * (M + 1)
        z_pt[vol_start_idx : vol_end_idx,:] = order_fixed_size * 2 #traded vols
        z_pt[vol_end_idx : ,:] = 1e5 #traded prices
    Z = {}
    Z['MM'] = z_mm.reshape(num_state_features['MM'],1)
    for i in range(num_pts):
        Z[f'PT{i + 1}'] = z_pt[:,i].reshape(num_state_features[f'PT{i + 1}'],1)
    return Z

def episode_averages(array, num_iter_per_ep, cumulative = False):
    # if np.shape(array)[1] > 1:
    num_ep = int(np.shape(array)[0]/num_iter_per_ep)
    avg_array = np.zeros(num_ep)
    for ep in range(num_ep):
        start = ep * num_iter_per_ep
        end = (ep + 1) * num_iter_per_ep
        if cumulative:
            avg_array[ep] = np.sum(array[start:end])
        else:
            avg_array[ep] = np.average(array[start:end])
    return avg_array, num_ep

def moving_average(array, window = 1000):
    try:
        if np.shape(array)[1] > 1:
            num_channels = np.shape(array)[1]
            mov_avg = np.zeros((np.shape(array)[0]-window+1,num_channels))
            for i in range(num_channels):
                mov_avg[:,i] = np.convolve(array[:,i], np.ones(window), 'valid') / window
            return mov_avg
    except IndexError:
        return np.convolve(array, np.ones(window), 'valid') / window

def num_unique_elements(array, move_by = 1000):
    counts = np.zeros_like(array,dtype=int)
    for i in range(len(array)):
        if i % move_by == 0:
            counts[i] = len(set(array[0:i]))
            print(i,len(array),counts[i])
        else:
            counts[i] = counts[i-1]
    return counts

def multi_agent_init(num_pts, mm_add_volume = 0, pt_add_volume = 0, L = 1, d = 0, 
M = 1, base_log_dir = 'results/trial', log_flag = True, mm_max_depth = 5,
order_fixed_size = 20, pt_add_momentum = 0, timestep_duration = None):
    # print(order_fixed_size)
    ## Define multi agent environment 
    learning_agent_ids = ['MM'] + ['PT' + str(x) for x in range(1, num_pts + 1)]
    num_learning_agents = 1 + num_pts
    # State/Observation Space
    # MM: [qt_L^ask,...,qt^ask,qt_L^bid,...,qt^bid] prices (best ask, best bid) +
    #     [qt_L^ask,...,qt^ask,qt_L^bid,...,qt^bid] volumes (best ask, best bid) +
    #     [St,Dt,ht_1,ht,ct_1,ct,{vt_1}--2*D] + 
    #     [vt_M_d^ask,...,vt_d^ask,vt_M_d^bid,...,vt_d^bid] traded volumes +
    #     [pt_M_d,...,pt_d] traded prices
    # PTs: [qt_L^ask,...,qt^ask,qt_L^bid,...,qt^bid] prices +
    #      [qt_L^ask,...,qt^ask,qt_L^bid,...,qt^bid] volumes +
    #      [St,Dt,ht_1,ht,ct_1,ct] + 
    #      [vt_M_d^ask,...,vt_d^ask,vt_M_d^bid,...,vt_d^bid] traded volumes +
    #      [pt_M_d,...,pt_d] traded prices
    #      [pt/pt_1,pt/pt_10,pt/pt_30] momentum signals if pt_add_momentum
    num_state_features = {}
    add_volume = {}
    observation_space = {}
    state_lows = {}
    state_highs = {}
    num_actions = {}
    num_state_features[learning_agent_ids[0]] = 4 * (L + 1) + 6 + 2 * mm_max_depth + 3 * (M + 1) * mm_add_volume
    add_volume[learning_agent_ids[0]] = mm_add_volume
    state_lows[learning_agent_ids[0]] = np.array(
        4 * (L + 1) * [0.0] #Best ask, bid prices, volumes t-L, .., t
        + [
            0.0,  # Spread: t
            0.0, # Depth: t
            -2,  # Holdings: t-1
            -2,  # Holdings: t
            -2,  # Cash: t-1
            -2,  # Cash: t
        ]
        + (2 * mm_max_depth) * [0.0]   # executed volumes
        + (3 * (M + 1) * mm_add_volume) * [0.0],    # traded volume at best ask, bid; traded prices: t-M-d, ... ,t-d
        dtype=np.float32,
        ).reshape(num_state_features[learning_agent_ids[0]], 1)
    state_highs[learning_agent_ids[0]] = np.array(
        4 * (L + 1) * [2.0] #Best ask, bid prices, volumes t-L, .., t
        + [
            5.0,  # Spread: t
            5.0, # Depth: t
            2,  # Holdings: t-1
            2,  # Holdings: t
            2,  # Cash: t-1
            2,  # Cash: t
        ]
        + (2 * mm_max_depth) * [1.0]   # executed volumes
        + (3 * (M + 1) * mm_add_volume) * [mm_max_depth],    # traded volume at best ask, bid; traded prices: t-M-d, ... ,t-d
        dtype=np.float32,
        ).reshape(num_state_features[learning_agent_ids[0]], 1)
    num_actions[learning_agent_ids[0]] = [5,mm_max_depth] #half-spread, depth# , hedging fraction 
    for i in range(num_pts):
        num_state_features[learning_agent_ids[1 + i]] = 4 * (L + 1) + 6 + 3 * (M + 1) * pt_add_volume + 3 * pt_add_momentum
        add_volume[learning_agent_ids[1 + i]] = pt_add_volume
        state_lows[learning_agent_ids[1 + i]] = np.array(
            4 * (L + 1) * [0.0] #Best ask, bid prices, volumes t-L, .., t
            + [
                0.0,  # Spread: t
                0.0, # Depth: t
                -2,  # Holdings: t-1
                -2,  # Holdings: t
                -2,  # Cash: t-1
                -2,  # Cash: t
            ]
            + (3 * (L + 1) * pt_add_volume) * [0.0]    # traded volume at best ask, bid; traded prices: t-M-d, ... ,t-d
            + (3 * pt_add_momentum) * [0.0], # momentum signals at 1min, 10min and 30min
            dtype=np.float32,
        ).reshape(num_state_features[learning_agent_ids[1 + i]], 1)
        state_highs[learning_agent_ids[1 + i]] = np.array(
            4 * (L + 1) * [2.0] #Best ask, bid prices, volumes t-L, .., t
            + [
                5.0,  # Spread: t
                5.0, # Depth: t
                2,  # Holdings: t-1
                2,  # Holdings: t
                2,  # Cash: t-1
                2,  # Cash: t
            ]
            + (3 * (L + 1) * pt_add_volume) * [mm_max_depth]    # traded volume at best ask, bid; traded prices: t-M-d, ... ,t-d
            + (3 * pt_add_momentum) * [2.0], # momentum signals at 1min, 10min and 30min
            dtype=np.float32,
        ).reshape(num_state_features[learning_agent_ids[1 + i]], 1)
        num_actions[learning_agent_ids[1 + i]] = [4,3] #half-spread, sell/hold/buy
    action_space = {}
    for i in range(num_learning_agents):
        # state_highs[learning_agent_ids[i]] = np.finfo(np.float32).max *\
        #                                                 np.ones((num_state_features[learning_agent_ids[i]],1))
        observation_space[learning_agent_ids[i]] \
            = gym.spaces.Box(
                state_lows[learning_agent_ids[i]],
                state_highs[learning_agent_ids[i]],
                shape=(num_state_features[learning_agent_ids[i]], 1),
                dtype=np.float32,
            )
        # print(observation_space[learning_agent_ids[i]])
        action_space[learning_agent_ids[i]] = gym.spaces.MultiDiscrete(num_actions[learning_agent_ids[i]])

    env_config = {
            "mkt_open": "09:30:00",
            "mkt_close": "16:00:00",
            "timestep_duration": "60S" if timestep_duration is None else timestep_duration,#tune.grid_search(["30S", "60S"]),
            "num_pts": num_pts,
            "order_fixed_size": order_fixed_size,#tune.grid_search([10,50,100]),
            "learning_agent_ids": learning_agent_ids,
            "observation_space": observation_space,
            "action_space": action_space,
            "mm_max_depth": mm_max_depth,
            "num_state_features": num_state_features,
            "add_volume": add_volume,
            "quote_history": L,
            "trade_history": M,
            "delay_in_volume_reporting": d,
            "base_log_dir": base_log_dir,
            "log_flag": log_flag,
            "pt_add_momentum": pt_add_momentum
        }
    return env_config

def multi_agent_policies(learning_agent_ids, observation_space, action_space):
    num_learning_agents = len(learning_agent_ids)
    policies = {}
    for i in range(num_learning_agents):
        policies[learning_agent_ids[i]] = PolicySpec(None, observation_space[learning_agent_ids[i]], 
                                            action_space[learning_agent_ids[i]], 
                                            {"gamma": 1.0})
    policy_mapping_fn = lambda agent_id: agent_id
    policies_dict = {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn
            }
    return policies_dict

def run_episode(num_pts, env_config, policy, horizon, seed = None, 
add_volume = None, L = 1, M = 1, pt_add_momentum = None):
    env = gym.make(
        "marl-v0",
        background_config="rmsc04",
        **env_config,
        )
    env.seed(seed)
    norm_state = env.reset()
    # print(f'state: {state}')
    obs_dict = get_observables(num_pts,L,M,horizon,env.state,reset=True,trades=env.trades,
                    pt_add_momentum=pt_add_momentum,
                    price_history_at_t=env.price_history[0])
    for t in range(horizon):
        action = policy.get_action(norm_state)
        norm_state_, norm_reward, done, info = env.step(action)
        mm_pnls = np.array([env.spread_pnl,env.inventory_pnl])
        # print(t, state, action, state_, reward)
        obs_dict = get_observables(num_pts,L,M,horizon,env.state,action,env.reward,mm_pnls,
                            t,obs_dict,False,env.trades,pt_add_momentum,
                            env.price_history[t],[env.ask_depth,env.bid_depth])
        norm_state = norm_state_
    return obs_dict, env.gym_agent.get_matching_agents(), env.gym_agent.get_matched_value_agent_orders(), env.get_agent_pnls()

def get_observables(num_pts, L, M, horizon, state, action = None, reward = None, mm_pnls = None,
t = None, prev_obs_dict = None, reset = False, trades = None, pt_add_momentum = 0,
price_history_at_t = 0, extras = None):
    '''
    Returns a dict by augmenting current (s,a,r) tuple to obs dict

    Observables for Market: (at t)
        Quoted Price: 2 * horizon (best ask, bid)
        Quoted Volume: 2 * horizon (best ask, bid)
        Traded Price: 1 * horizon
        Traded Volume: 2 * horizon (ask, bid)
        Spread: 1 * horizon
        Depth: 2 * horizon (ask, bid)

    Observables for MM: (at t)
        Inventory: 1 * horizon
        Cash: 1 * horizon
        Spread: 1 * horizon
        Depth: 1 * horizon
        Spread PnL: 1 * horizon
        Inventory PnL: 1 * horizon
        Reward: 1 * horizon

    Observables for each PT: (at t)
        Inventory: 1 * horizon
        Cash: 1 * horizon
        Distance to mid: 1 * horizon
        (Order) Side: 1 * horizon
        Reward: 1 * horizon
        Momentum: 3 * horizon
        Price History: 1 * horizon

    Returns obs_dict: 
    {
        "Market": {
            "Price": np.array(horizon),
            "Spread": ,

        },
        "MM": {},
        "PT1": {},
        "PT2": {},
        ...
        "PTn": {}
    }
    '''
    agent_ids = list(state.keys())
    # print(t,horizon)
    if reset:
        obs_dict = {}
        obs_dict['Market'] = {}
        obs_dict['Market']['Quoted Price'] = np.zeros((2,horizon)) #Best ask, bid at time t
        obs_dict['Market']['Quoted Volume'] = np.zeros((2,horizon)) #Best ask, bid at time t
        obs_dict['Market']['Traded Price'] = np.zeros(horizon) #Best ask, bid at time t
        obs_dict['Market']['Traded Volume'] = np.zeros((2,horizon)) #Best ask, bid at time t
        obs_dict['Market']['Spread'] = np.zeros(horizon)
        obs_dict['Market']['Depth'] = np.zeros((2,horizon)) #Ask, bid depth at time t
        obs_dict[agent_ids[0]] = {}
        obs_dict[agent_ids[0]]['Inventory'] = np.zeros(horizon)
        obs_dict[agent_ids[0]]['Cash'] = np.zeros(horizon)
        obs_dict[agent_ids[0]]['Spread'] = np.zeros(horizon)
        obs_dict[agent_ids[0]]['Depth'] = np.zeros(horizon)
        obs_dict[agent_ids[0]]['Spread PnL'] = np.zeros(horizon)
        obs_dict[agent_ids[0]]['Inventory PnL'] = np.zeros(horizon)
        obs_dict[agent_ids[0]]['Reward'] = np.zeros(horizon)
        for i in range(num_pts):
            obs_dict[agent_ids[1 + i]] = {}
            obs_dict[agent_ids[1 + i]]['Inventory'] = np.zeros(horizon)
            obs_dict[agent_ids[1 + i]]['Cash'] = np.zeros(horizon)
            obs_dict[agent_ids[1 + i]]['Distance to mid'] = np.zeros(horizon)
            obs_dict[agent_ids[1 + i]]['Side'] = np.zeros(horizon)
            obs_dict[agent_ids[1 + i]]['Reward'] = np.zeros(horizon)
            if pt_add_momentum:
                obs_dict[agent_ids[1 + i]]['Momentum'] = np.zeros((3,horizon))
                obs_dict[agent_ids[1 + i]]['Price History'] = np.zeros(horizon)
    else:
        obs_dict = prev_obs_dict
        obs_dict['Market']['Quoted Price'][0,t] = state[agent_ids[0]][L]
        obs_dict['Market']['Quoted Price'][1,t] = state[agent_ids[0]][2*L+1]
        obs_dict['Market']['Quoted Volume'][0,t] = state[agent_ids[0]][3*L+2]
        obs_dict['Market']['Quoted Volume'][1,t] = state[agent_ids[0]][4*L+3]
        if trades is not None:
            obs_dict['Market']['Traded Price'][t] = trades[-1,2]
            obs_dict['Market']['Traded Volume'][0,t] = trades[-1,0] #volume traded at best ask at t
            obs_dict['Market']['Traded Volume'][1,t] = trades[-1,1]
        obs_dict['Market']['Spread'][t] = state[agent_ids[0]][4*L+4] # spread at t
        obs_dict['Market']['Depth'][:,t] = state[agent_ids[0]][4*L+5] # depth at t
        if extras is not None:
            obs_dict['Market']['Depth'][0,t] = extras[0] #Ask depth
            obs_dict['Market']['Depth'][1,t] = extras[1] #Bid depth

        obs_dict[agent_ids[0]]['Inventory'][t] = state[agent_ids[0]][4*L+7]
        obs_dict[agent_ids[0]]['Cash'][t] = state[agent_ids[0]][4*L+9]
        obs_dict[agent_ids[0]]['Spread'][t] = 0.5 * (action[agent_ids[0]][0] + 1)
        obs_dict[agent_ids[0]]['Depth'][t] = action[agent_ids[0]][1] + 1
        # obs_dict[agent_ids[0]]['Hedge'][t] = action[agent_ids[0]][2] * 0.2
        obs_dict[agent_ids[0]]['Spread PnL'][t] = mm_pnls[0] if mm_pnls is not None else 0
        obs_dict[agent_ids[0]]['Inventory PnL'][t] = mm_pnls[1] if mm_pnls is not None else 0
        # obs_dict[agent_ids[0]]['Hedging Cost'][t] = mm_pnls[2] if mm_pnls is not None else 0
        obs_dict[agent_ids[0]]['Reward'][t] = reward[agent_ids[0]]

        for i in range(num_pts):
            obs_dict[agent_ids[1 + i]]['Inventory'][t] = state[agent_ids[1 + i]][4*L+7]
            obs_dict[agent_ids[1 + i]]['Cash'][t] = state[agent_ids[1 + i]][4*L+9]
            obs_dict[agent_ids[1 + i]]['Distance to mid'][t] = 0.5 * (action[agent_ids[1 + i]][0] + 1)
            obs_dict[agent_ids[1 + i]]['Side'][t] = action[agent_ids[1 + i]][1]
            obs_dict[agent_ids[1 + i]]['Reward'][t] = reward[agent_ids[1 + i]]
            if pt_add_momentum:
                obs_dict[agent_ids[1 + i]]['Momentum'][:,t] = state[agent_ids[1 + i]][-3:,0]
                obs_dict[agent_ids[1 + i]]['Price History'][t] = price_history_at_t
    return obs_dict

class multi_agent_policy_MARL:
    """
    Policy learned during the training used to compute action
    """
    def __init__(self, data_folder, config, env_name):
        self.learning_agent_ids = list(config["multiagent"]["policies"].keys())
        analysis = Analysis(data_folder)
        try:
            best_trial_path = analysis.get_best_logdir(metric='episode_reward_mean', mode='max')
        except KeyError:
            analysis.set_filetype('json')
            best_trial_path = analysis.get_best_logdir(metric='episode_reward_mean', mode='max')
        best_checkpoint = analysis.get_best_checkpoint(trial = best_trial_path, mode='max')
        self.trainer = ppo.PPOTrainer(config = config, env = env_name)
        self.trainer.restore(best_checkpoint)
        
    def get_action(self, state):
        action = {}
        for i in range(len(self.learning_agent_ids)):
            action[self.learning_agent_ids[i]] = self.trainer.compute_single_action(
                                                    state[self.learning_agent_ids[i]],
                                                    policy_id=self.learning_agent_ids[i]) 
        return action

def load_and_save(ep, path, observables_list, obs_dict, obs_dict_key):
    # print(f'\nIn load and save {obs_dict_key}\n')
    for obs_string in observables_list:
        if os.path.exists(path + f'{obs_string}.npy') and ep > 0: #Load only if at an intermediate testing episode, not the first
            try:
                prev_obs = np.load(path + f'{obs_string}.npy')
            except ValueError:
                print(f'Loading pickle file at episode {ep}?! {obs_dict_key}: {obs_string}')
                print(obs_dict[obs_dict_key][obs_string].dtype,obs.dtype)
                prev_obs = np.load(path + f'{obs_string}.npy',allow_pickle=True)
            obs = np.vstack((prev_obs,obs_dict[obs_dict_key][obs_string]))
        else:
            obs = obs_dict[obs_dict_key][obs_string]
        obs[np.isnan(obs)] = 0
        np.save(path + f'{obs_string}.npy',obs) # num_test * horizon
    return

def log_results(ep, obs_dict, log_dir, num_pts, matching_agents = None, 
matched_value_agent_orders = None, agent_pnls = None):
    print('Logging results\n')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir + 'Market/'):
        os.makedirs(log_dir + 'Market/')
    if not os.path.exists(log_dir + 'MM/'):
        os.makedirs(log_dir + 'MM/')
    for i in range(num_pts):
        if not os.path.exists(log_dir + f'PT{i + 1}/'):
            os.makedirs(log_dir + f'PT{i + 1}/')
    
    if matching_agents is not None:
        obs_dict['MM']['Matching Agents'] = matching_agents['MM'].reshape((1,-1))
        for i in range(num_pts):
            obs_dict[f'PT{i + 1}']['Matching Agents'] = matching_agents[f'PT{i + 1}'].reshape((1,-1))
    if matched_value_agent_orders is not None:
        obs_dict['Market']['Matched Value Agent Orders'] = matched_value_agent_orders # 2 * 390
    if agent_pnls is not None:
        obs_dict['Market']['Agent PnLs'] = agent_pnls.reshape((1,-1))

    # Save market observables
    load_and_save(ep,log_dir+'Market/',market_obs_list + ['Matched Value Agent Orders'],obs_dict,'Market')

    # Save MM observables
    load_and_save(ep,log_dir+'MM/',mm_obs_list,obs_dict,'MM')

    # Save PT observables
    for i in range(num_pts):
        load_and_save(ep,log_dir+f'PT{i + 1}/',pt_obs_list,obs_dict,f'PT{i + 1}')
    # print('Done all load and save!')
    return
