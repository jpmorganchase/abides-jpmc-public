from copy import deepcopy
import importlib
import logging
from multiprocessing.sharedctypes import Value
from typing import Any, Dict, List, Deque, Tuple
from abc import ABC
import numpy as np
from abides_core import Kernel, NanosecondTime
from abides_core.utils import fmt_ts, str_to_ns, subdict
from abides_core.generators import ConstantTimeGenerator
from abides_markets.utils import config_add_agents
import abides_markets.agents.utils as markets_agent_utils
from abides_markets.orders import LimitOrder, MarketOrder
import os

from .markets_environment import AbidesGymMarketsEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ..experimental_agents.marl_gym_agent import MultiAgentGymAgent
from ray.rllib.utils.typing import MultiAgentDict
from scripts.marl_utils import get_observables, log_results, normalizing_quantities_state, \
        normalizing_quantities_reward, project_to_boundary

class SubGymMultiAgentRLEnv_v0(AbidesGymMarketsEnv, MultiAgentEnv):
    """
    Multi Agent RL V0 environnement.
        Learning MM:
        - Action Space:
            - Half spread placement
            - Depth
            # - Hedging fraction
        - State Space:
            - Mid price
            - Spread
            - Holdings
            - Executed volumes
        - Reward:
            - Spread PnL
            - Inventory PnL
            # - Hedging Cost

    Learning Principal Trader:
        - Action Space:
            - Distance from mid
            - Buy/Sell/Hold
        - State Space:
            - Mid price
            - Spread
            - Holdings
        - Reward:
            - Execution PnL
    """

    def __init__(
        self,
        background_config: str = "rmsc04",
        mkt_open: str = "09:30:00",
        mkt_close: str = "16:00:00",
        starting_cash: int = 1_000_000,
        state_history_length: int = 2, # length of history for raw_state
        market_data_buffer_length: int = 10, # length of history for parsed_mkt_data, parsed_volume_data
        first_interval: str = "00:05:00",
        reward_mode: str = "dense",
        debug_mode: bool = False,
        ## Logging info
        kernel_skip_log: bool = True,
        kernel_log_dir: str = None,
        exchange_log_orders: bool = None,
        base_log_dir: str = None,
        log_flag: bool = True,
        num_pts: int = 1,
        timestep_duration: str = "30s",
        order_fixed_size: int = 10,
        learning_agent_ids: List = ['MM','PT'],
        mm_max_depth: int = 5,
        num_state_features: Dict = None,
        add_volume: Dict = None, #keys: agents; values: {0,1}
        quote_history: int = None, #L: history of quoted prices, volumes to add to states of MM, PT
        trade_history: int = None, #M
        delay_in_volume_reporting: int = None, 
        observation_space = None,
        action_space = None,
        gymAgentConstructor = MultiAgentGymAgent,
        pt_add_momentum: int = 0, # if 1: add 1min, 10min and 30min momentum signals to PT state
        ## Arguments for background config - rmsc04
        num_value_agents: int = 2,
        num_momentum_agents: int = 2,
        num_noise_agents: int = 20,
        ## Variable arguments for background config - rmsc04
        momentum_agent_freq: str = "5min",
        linear_oracle: bool = False, #if True, use a linear oracle
        lambda_a: int = 5.7e-12) -> None:

        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config)
        )  #
        self.mkt_open: NanosecondTime = str_to_ns(mkt_open)
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)  #
        self.num_pts = num_pts
        self.starting_cash: int = starting_cash  #
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)  #
        self.max_iter: int = int((self.mkt_close - self.mkt_open)/self.timestep_duration)
        # print('max_iter: ',self.max_iter)
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        self.wakeup_interval_generator = ConstantTimeGenerator(
                step_duration=self.timestep_duration
            )
        self.reward_mode: str = reward_mode
        # self.done_ratio: float = done_ratio
        self.debug_mode: bool = debug_mode
        self.pt_add_momentum = pt_add_momentum
        self.price_history = np.zeros(self.max_iter) #Only updated if pt_add_momentum
        print(f'#PTs: {self.num_pts}; Momentum: {self.pt_add_momentum}; Timestep: {timestep_duration}')

        self.order_fixed_size: int = order_fixed_size
        self.learning_agent_ids: List = learning_agent_ids
        self.num_learning_agents = len(self.learning_agent_ids)
        self.observation_space = observation_space
        self.action_space = action_space
        self.gymAgentConstructor = gymAgentConstructor
        self.mm_max_depth = mm_max_depth
        self.num_state_features = num_state_features
        self.add_volume = add_volume
        self.quote_history = quote_history
        self.trade_history = trade_history
        self.delay_in_volume_reporting = delay_in_volume_reporting
        if self.delay_in_volume_reporting < 0:
            print('Cannot look into future volumes!!!Setting zero delay in volume reporting!')
            self.delay_in_volume_reporting = 0
        if self.delay_in_volume_reporting > self.market_data_buffer_length - 1 - self.trade_history:
            print(f'Cannot delay further than buffer length! Setting to max possible delay\
                 of {self.market_data_buffer_length-1-self.trade_history} for M = {self.trade_history}')
            self.delay_in_volume_reporting = self.market_data_buffer_length - 1 - self.trade_history
        # print(self.delay_in_volume_reporting)
        self.kernel_skip_log = kernel_skip_log
        self.kernel_log_dir = kernel_log_dir
        self.log_dir = base_log_dir + f'_momentum_agent_freq{momentum_agent_freq}_lambda_a{lambda_a}/'
        self.log_flag = log_flag
        if self.log_flag:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.log_dir + 'MM/'):
                os.makedirs(self.log_dir + 'MM/')
            for i in range(self.num_pts):
                if not os.path.exists(self.log_dir + f'PT{i+1}/'):
                    os.makedirs(self.log_dir + f'PT{i+1}/')
        
        # Load normalizing quantities for states and rewards
        self.Z_state = normalizing_quantities_state(self.num_pts,self.quote_history,
                                                self.trade_history,self.num_state_features,
                                                self.add_volume,self.mm_max_depth,
                                                self.order_fixed_size,self.pt_add_momentum)
        self.Z_reward = normalizing_quantities_reward(self.num_pts,self.mm_max_depth,self.order_fixed_size)
        self.num_noise_agents = num_noise_agents
        self.num_value_agents = num_value_agents
        self.num_momentum_agents = num_momentum_agents

        background_config_args = {  
            "mm_flag": False,
            "end_time": self.mkt_close,
            "exchange_log_orders": exchange_log_orders,
            "num_value_agents": num_value_agents,
            "num_momentum_agents": num_momentum_agents,
            "num_noise_agents": num_noise_agents,
            "momentum_agent_freq": momentum_agent_freq,
            "lambda_a": lambda_a,
            "linear_oracle": linear_oracle,
            }
        super().__init__(
            background_config_pair=(
                self.background_config.build_config,
                background_config_args,
            ),
            wakeup_interval_generator=self.wakeup_interval_generator,
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,
            market_data_buffer_length=self.market_data_buffer_length,
            first_interval=self.first_interval,
            gymAgentConstructor=gymAgentConstructor
        )
        self.extra_background_config_kvargs = background_config_args #here because super resets it

        extra_gym_agent_kvargs = {
                                "num_pts": self.num_pts,
                                "names": self.learning_agent_ids,
                                "types": self.learning_agent_ids, 
                                "starting_cashs": self.num_learning_agents*np.ones(self.starting_cash)
                                }
        self.extra_gym_agent_kvargs = extra_gym_agent_kvargs #here because super resets it
        
        # marked_to_market limit to STOP the episode
        # self.up_done_condition = {}
        # # self.down_done_condition = {}
        # self.done_action = {} #agent: action at which it was done the first time
        # self.done_iter = {} #agent: iter at which it was done the first time
        # for i in range(self.num_learning_agents):
        #     self.up_done_condition[self.learning_agent_ids[i]] = (1 + self.done_ratio[i]) * starting_cash 
        #     self.done_action[self.learning_agent_ids[i]] = None
        #     self.done_iter[self.learning_agent_ids[i]] = np.inf

        # instantiate previous_marked_to_market as starting_cash
        self.previous_marked_to_market = self.starting_cash
        self.mid_prices = None
        self.spreads = None

    def reset(self) -> MultiAgentDict:
        # seed = self.np_random.randint(low=0, high=2 ** 32, dtype="uint64")
        seed = np.random.randint(low=0, high=2 ** 32, dtype="uint64")
        background_config_args = self.background_config_pair[1]
        background_config_args.update(
            {"seed": seed, **self.extra_background_config_kvargs}
        )
        background_config_state = self.background_config_pair[0](
            **background_config_args
        )

        gym_agent = self.gymAgentConstructor(
            id=len(background_config_state["agents"]), #105
            symbol="ABM",
            name="GYM_MM_PT",
            starting_cash=self.starting_cash,
            first_interval=self.first_interval,
            max_iter=self.max_iter,
            wakeup_interval_generator=self.wakeup_interval_generator,
            market_data_buffer_length=self.market_data_buffer_length,
            state_buffer_length=self.state_buffer_length,
            num_noise_agents=self.extra_background_config_kvargs["num_noise_agents"],
            num_value_agents=self.extra_background_config_kvargs["num_value_agents"],
            num_momentum_agents=self.extra_background_config_kvargs["num_momentum_agents"],
            **self.extra_gym_agent_kvargs,
        )
        config_state = config_add_agents(background_config_state, [gym_agent] + list(gym_agent.agents.values()))
                                        # self.extra_background_config_kvargs["latency_type"])
        self.config_state = config_state
        self.gym_agent = gym_agent
        # print(fmt_ts(config_state['start_time']),fmt_ts(config_state['stop_time']))
        # print(self.kernel_skip_log,self.kernel_log_dir)
        kernel = Kernel(
            random_state=np.random.RandomState(seed=seed),
            skip_log=self.kernel_skip_log,
            log_dir=self.kernel_log_dir,
            **subdict(
                config_state,
                [
                    "start_time",
                    "stop_time",
                    "agents",
                    "agent_latency_model",
                    "default_computation_delay",
                    "custom_properties",
                ],
            ),
        )
        kernel.gym_agents = [self.gym_agent]
        kernel.initialize()

        raw_state = kernel.runner()
        self.raw_state_log = raw_state
        self.iter = 0
        self.norm_state, self.state = self.raw_state_to_state(deepcopy(raw_state["result"]))
        self.obs_dict = get_observables(self.num_pts,self.quote_history,
                                        self.trade_history,self.max_iter,
                                        self.state,reset=True,trades=self.trades,
                                        pt_add_momentum=self.pt_add_momentum,
                                        price_history_at_t=self.price_history[0])

        self.kernel = kernel
        # print('\nReset env!\n')
        # print(self.state)
        return self.norm_state

    def step(self, action: Dict) -> Tuple[Dict, Dict, Dict, Dict[str, Any]]:    
        self.iter += 1
        # print(f'iter: {self.iter}/{self.max_iter}')
        # print('\nin step')
        # print('MM state: ',self.state['MM'])
        abides_action = self._map_action_space_to_ABIDES_SIMULATOR_SPACE(action)
        # print('action: ',action,abides_action)
        next_raw_state = self.kernel.runner((self.gym_agent, abides_action))
        # print('raw state: ',raw_state["result"])
        # print('\nback from kernel runner\n')
        # print(fmt_ts(next_raw_state["result"][self.learning_agent_ids[0]]\
        #                             [-1]["internal_data"]["current_time"]))
        self.raw_state_log = next_raw_state
        next_norm_state, next_state = self.raw_state_to_state(deepcopy(next_raw_state["result"]),action)
        # print('|| raw state')
        # print('next_state: ',next_state[self.learning_agent_ids[1]])

        self.norm_reward, self.reward = self.raw_state_to_reward(deepcopy(next_raw_state["result"]),action)
        self.done = self.raw_state_to_done(deepcopy(next_raw_state["result"]),action)
        
        # print('reward: ',self.reward)
        # print(self.done)
        self.info = self.raw_state_to_info(deepcopy(next_raw_state["result"]))
        # print(self.state)
        self.obs_dict = get_observables(self.num_pts,self.quote_history,
                            self.trade_history,self.max_iter,
                            self.state,action,self.reward,
                            np.array([self.spread_pnl,self.inventory_pnl]),
                            self.iter-1,self.obs_dict,False,self.trades,
                            self.pt_add_momentum,self.price_history[self.iter-1],
                            [self.ask_depth,self.bid_depth])
        # print(self.obs_dict)
        if self.iter >= self.max_iter and self.log_flag:
            matching_agents = self.gym_agent.get_matching_agents()
            agent_pnls = self.get_agent_pnls()
            print(matching_agents)
            matched_value_agent_orders = self.gym_agent.get_matched_value_agent_orders()
            log_results(1,self.obs_dict,self.log_dir,self.num_pts,matching_agents,matched_value_agent_orders,agent_pnls) #loads for previous training episodes, and saves augmetnted with current training episode
            # print('Back in marl env')
        self.state = next_state
        self.norm_state = next_norm_state
        return (self.norm_state, self.norm_reward, self.done, self.info)        

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(
        self, action: Dict
    ) -> Dict[List[Dict[str, Any]],Any]:
        abides_action = {}
        # print('\nin map action to ABIDES space\n',action)
        mm_action = []
        mm_half_spread = 0.5 * (action[self.learning_agent_ids[0]][0] + 1)
        mm_depth = action[self.learning_agent_ids[0]][1] + 1 # in {1,2,3,4,5}
        mid_prices = self.mid_prices
        # print('mid: ',mid_prices)
        for i in range(mm_depth): #for each level of MM order placement
            mm_action.append({"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                                "limit_price": int(mid_prices[-1] - mm_half_spread - i)})
            mm_action.append({"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                                "limit_price": int(mid_prices[-1] + mm_half_spread + i)})
        # Place MM hedging order --- TODO: make sure it cant match its own order!!!
        # mm_inv = self.state[self.learning_agent_ids[0]][4*self.quote_history+7,0]
        #self.raw_state_log['result'][self.learning_agent_ids[0]][-1]["internal_data"]["holdings"]
        # mm_direction = "BUY" if mm_inv < 0 else "SELL"
        # mm_hedge_size = int(0.2 * action[self.learning_agent_ids[0]][2] * np.abs(mm_inv))
        # if mm_hedge_size > 0:
        #     mm_action.append({"type": "MKT", "direction": mm_direction, "size": mm_hedge_size})
        abides_action[self.learning_agent_ids[0]] = mm_action    

        # Place LO of PT at given dist to mid, side
        for i in range(self.num_pts):
            pt_action = []
            pt_dist = 0.5*(action[self.learning_agent_ids[1 + i]][0] + 1)
            pt_side = None
            if action[self.learning_agent_ids[1 + i]][1] == 0:
                pt_side = "SELL"
                pt_price = mid_prices[-1] + pt_dist
            elif action[self.learning_agent_ids[1 + i]][1] == 2:
                pt_side = "BUY"
                pt_price = mid_prices[-1] - pt_dist
            if pt_side is not None:
                pt_action = [{"type": "LMT", "direction": pt_side, "size": int(self.order_fixed_size/self.num_pts),
                                        "limit_price": int(pt_price)}]
            abides_action[self.learning_agent_ids[1 + i]] = pt_action
        return abides_action
    
    def raw_state_to_state(self, raw_state: Dict[str, Any], action: Dict = None) -> Dict:
        # print('\nin raw state to state\n')
        try:
            prev_mid = self.mid_prices[-1] if self.mid_prices is not None else raw_state[self.learning_agent_ids[0]][-1]["parsed_mkt_data"][-1]["last_transaction"]
        except IndexError:
            print(raw_state[self.learning_agent_ids[0]][-1]["parsed_mkt_data"])
            
        computed_state = {}
        # 1) Holdings
        mm_holdings = raw_state[self.learning_agent_ids[0]][-1]["internal_data"]["holdings"] 
        prev_mm_holdings = int(self.state[self.learning_agent_ids[0]][4*self.quote_history+7,0]) if self.state is not None else mm_holdings
        mm_cash = raw_state[self.learning_agent_ids[0]][-1]["internal_data"]["cash"] 
        prev_mm_cash = int(self.state[self.learning_agent_ids[0]][4*self.quote_history+9,0]) if self.state is not None else mm_cash
        # print('MM',raw_state[self.learning_agent_ids[0]][-1]["parsed_mkt_data"])
        # print('prev_mm_holds: ',prev_mm_holdings)
        # State for MM
        mm_executed_orders = raw_state[self.learning_agent_ids[0]][-1]["internal_data"]["inter_wakeup_executed_orders"] #list of (qty,price)
        # print('mm_exec_orders: ',mm_executed_orders)
        mm_exec_vols = np.zeros((self.mm_max_depth,2)) #sell, buy
        mm_prev_half_spread = 0.5 * (action[self.learning_agent_ids[0]][0] + 1) if action is not None else 0.5
        mm_prev_depth = action[self.learning_agent_ids[0]][1] + 1 if action is not None else 1 # in {1,2,3,4,5}
        # prev_mid = self.state[self.learning_agent_ids[0]][1] if self.state is not None else mid_prices[-1]
        for order in mm_executed_orders:
            # print(order,prev_mid)
            if isinstance(order,LimitOrder):
                for i in range(mm_prev_depth):
                    if order.side.is_ask() and order.limit_price == int(prev_mid + mm_prev_half_spread + i):
                        mm_exec_vols[i,0] = order.quantity
                    elif order.side.is_bid() and order.limit_price == int(prev_mid - mm_prev_half_spread - i):
                        mm_exec_vols[i,1] = order.quantity
        # print('mm_exec_vols: ',mm_exec_vols)  
        self.mm_exec_vols = mm_exec_vols
        ## Market states
        quotes = self.get_quotes(raw_state) #self.quote_history+1 * 4 (ask,bid prices and ask, bid volumes)
        if self.iter == 0 and self.pt_add_momentum:
            self.price_history[0] = (quotes[-1,0] + quotes[-1,1])/2 # initial price
        trades = self.get_trades(raw_state) #self.trade_history+1 * 3 (ask,bid vols and prices)
        mid_prices, spreads, depth = self.get_mid_spread_depth(raw_state)
        # print(f'mid: {mid_prices[-1]}')
        # print('mid and spread: ',mid_prices,spreads)
        MM_state = np.array([],dtype=np.float32)
        MM_state = np.append(MM_state,quotes.flatten('F'))
        MM_state = np.append(MM_state,[spreads[-1],depth,prev_mm_holdings,mm_holdings,prev_mm_cash,mm_cash])
        computed_state[self.learning_agent_ids[0]] = np.append(MM_state,self.mm_exec_vols.flatten('F'))
                                                            
        # State for PTs
        for i in range(self.num_pts):
            pt_holdings = raw_state[self.learning_agent_ids[1 + i]][-1]["internal_data"]["holdings"]
            prev_pt_holdings = int(self.state[self.learning_agent_ids[1 + i]][4*self.quote_history+7,0]) if self.state is not None else pt_holdings
            pt_cash = raw_state[self.learning_agent_ids[1 + i]][-1]["internal_data"]["cash"]
            prev_pt_cash = int(self.state[self.learning_agent_ids[1 + i]][4*self.quote_history+9,0]) if self.state is not None else pt_cash
            # print('PT',raw_state[self.learning_agent_ids[1]][-1]["internal_data"])
            PT_state = np.array([],dtype=np.float32)
            PT_state = np.append(PT_state,quotes.flatten('F'))
            computed_state[self.learning_agent_ids[1 + i]] = np.append(PT_state,[spreads[-1],depth,prev_pt_holdings,pt_holdings,prev_pt_cash,pt_cash])
            # print(f'PT{i+1}: {pt_holdings}, {pt_cash}')
        for i in range(self.num_learning_agents):
            if self.add_volume[self.learning_agent_ids[i]]:
                computed_state[self.learning_agent_ids[i]] = np.append(computed_state[self.learning_agent_ids[i]],
                                                                trades.flatten('F'))
            try:
                computed_state[self.learning_agent_ids[i]] = computed_state[self.learning_agent_ids[i]].\
                                                            reshape(self.num_state_features[self.learning_agent_ids[i]],1)#torch.from_numpy()
            except ValueError: #Happens if pt_add_momentum is true so we still need to add more before reshaping
                pass
        # print(computed_state)
        if self.pt_add_momentum:
            mom_signals = self.get_momentum_signals(quotes)
            for i in range(self.num_pts):
                computed_state[self.learning_agent_ids[1 + i]] = np.append(computed_state[self.learning_agent_ids[1 + i]],mom_signals)
                computed_state[self.learning_agent_ids[1 + i]] = computed_state[self.learning_agent_ids[1 + i]].\
                                                                reshape(self.num_state_features[self.learning_agent_ids[1 + i]],1)
        # print(computed_state)
        norm_state = self.normalize_state(computed_state)
        # print(norm_state,computed_state)
        return norm_state, computed_state

    def get_momentum_signals(self, quotes):
        mom = np.ones(3)
        if self.iter < self.max_iter:
            self.price_history[self.iter] = (quotes[-1,0] + quotes[-1,1])/2
            mom[0] = self.price_history[self.iter]/self.price_history[self.iter - 1] if self.price_history[self.iter - 1] > 0 else 1
            if self.iter <= 10:
                mom[1:] = self.price_history[self.iter]/self.price_history[0] if self.price_history[0] > 0 else 1
            elif self.iter <= 30:
                mom[1] = self.price_history[self.iter]/self.price_history[self.iter - 10] if self.price_history[self.iter - 10] > 0 else 1 
                mom[2] = self.price_history[self.iter]/self.price_history[0] if self.price_history[0] > 0 else 1
            else:
                mom[1] = self.price_history[self.iter]/self.price_history[self.iter - 10] if self.price_history[self.iter - 10] > 0 else 1
                mom[2] = self.price_history[self.iter]/self.price_history[self.iter - 30] if self.price_history[self.iter - 30] > 0 else 1
            mom[mom == np.inf] = 1
            mom[np.isnan(mom)] = 1
        # print(self.iter,mom)
        return mom

    def get_quotes(self, raw_state: Dict[str, Any]):
        quotes = np.zeros((self.quote_history + 1,4))
        for l in range(self.quote_history + 1):
            try:
                mkt_data_l = raw_state[self.learning_agent_ids[0]][-1]["parsed_mkt_data"]\
                                    [l - 1 - self.quote_history] 
            except IndexError:
                # print(raw_state[self.learning_agent_ids[0]][-1]["parsed_mkt_data"])
                # print(self.iter,l,self.quote_history)
                if self.iter > 0: 
                    # print('Oh No!')
                    print(f'\nParsed mkt data of MM does not have enough history even with {self.iter} iterations!!\n')
                try:
                    mkt_data_l = raw_state[self.learning_agent_ids[0]][-1]["parsed_mkt_data"][0]
                    print('Using the earliest measurement!')
                except IndexError: #no history at all
                    print('No measurements at all!')
                    mkt_data_l = {'asks': [], 'bids': [], 'last_transaction': self.mid_prices[-1] if self.mid_prices is not None else 0}
            #mkt_data_l: {'bids': [(price,qty)], 'asks': [(price,qty)],...}
            # best ask, best bid prices
            quotes[l,0] = mkt_data_l['asks'][0][0] if len(mkt_data_l['asks']) > 0 else mkt_data_l['last_transaction']
            quotes[l,1] = mkt_data_l['bids'][0][0] if len(mkt_data_l['bids']) > 0 else mkt_data_l['last_transaction']
            # best ask, best bid volumes
            quotes[l,2] = mkt_data_l['asks'][0][1] if len(mkt_data_l['asks']) > 0 else 0
            quotes[l,3] = mkt_data_l['bids'][0][1] if len(mkt_data_l['bids']) > 0 else 0
        self.quotes = quotes
        return quotes

    def get_trades(self, raw_state: Dict[str, Any]):
        trades = np.zeros((self.trade_history+1,3)) 
        #[:,0]: vol traded at best ask
        #[:,1]: vol traded at best bid 
        #[0,:]: at t-M-d and [-1,:]: at t-d
        # print('\nin get prices and volumes\n')
        # print(raw_state[self.learning_agent_ids[0]][-1]["parsed_volume_data"])
        for m in range(self.trade_history + 1):
            try:
                vol_data_m = raw_state[self.learning_agent_ids[0]][-1]["parsed_volume_data"]\
                                    [m - 1 - self.trade_history - self.delay_in_volume_reporting]
            except IndexError:
                # print(raw_state[self.learning_agent_ids[0]][-1]["parsed_volume_data"])
                # print(self.iter,m,self.trade_history,self.delay_in_volume_reporting)
                if self.iter > 0: 
                    print(f'Parsed vol data of MM does not have enough history even with {self.iter} iterations!!')
                try:
                    print('Using the earliest measurement!\n')
                    vol_data_m = raw_state[self.learning_agent_ids[0]][-1]["parsed_volume_data"][0]
                except IndexError:
                    print('No measurements at all!')
                    vol_data_m = {'ask_volume': 0, 'bid_volume': 0, 'last_transaction': self.mid_prices[-1] if self.mid_prices is not None else 0}
            #mkt_data_l: {'bids': [(price,qty)], 'asks': [(price,qty)],...}
            trades[m,0] = vol_data_m['ask_volume']
            trades[m,1] = vol_data_m['bid_volume']
            trades[m,2] = vol_data_m['last_transaction']
        # print(volumes)
        self.trades = trades
        return trades

    def get_mid_spread_depth(self, raw_state: Dict[str, Any]):
        # print('\nin get mid, spread and depth\n')
        mid_prices = np.zeros(2) #t-1, t
        spreads = np.zeros(2) #t-1,t
        try:
            bids = raw_state[self.learning_agent_ids[0]][-1]["parsed_mkt_data"][-1]["bids"]
            asks = raw_state[self.learning_agent_ids[0]][-1]["parsed_mkt_data"][-1]["asks"]
            last_transactions = raw_state[self.learning_agent_ids[0]][-1]["parsed_mkt_data"][-1]["last_transaction"]
            mid = markets_agent_utils.get_mid_price(bids, asks, last_transactions)
        except IndexError:
            bids = []
            asks = []
            mid = self.price_history[self.iter - 1]
            pass
        mid_prices[-1] = mid
        best_bids = bids[0][0] if len(bids) > 0 else mid
        best_asks = asks[0][0] if len(asks) > 0 else mid
        worst_bids = bids[-1][0] if len(bids) > 0 else best_bids
        worst_asks = asks[-1][0] if len(asks) > 0 else best_asks
        self.bid_depth = best_bids - worst_bids
        self.ask_depth = worst_asks - best_asks
        depth = (worst_asks - worst_bids)/2
        spreads[-1] = best_asks - best_bids
        if self.mid_prices is not None and self.spreads is not None:
            mid_prices[0] = self.mid_prices[-1]
            spreads[0] = self.spreads[-1]
        else:
            mid_prices[0] = mid_prices[-1]
            spreads[0] = spreads[-1]
        self.mid_prices = mid_prices
        self.spreads = spreads
        # print(mid_prices,spreads,depth)
        return mid_prices, spreads, depth #mid: t-1, t, spread: t-1,t; depth: t

    def raw_state_to_reward(self, next_raw_state: Dict[str, Any], action: Dict) -> Dict:
        # print('in raw state to reward')
        if self.reward_mode == "dense":
            rewards = {}
            norm_rewards = {}
            mid_prices = self.mid_prices
            spreads = self.spreads
            # mid_prices, spreads = self.get_mid_spread(next_raw_state)
            # print('mid and spread: ',mid_prices,spreads)
            # Reward for MM
            mm_action = action[self.learning_agent_ids[0]]
            mm_holdings = next_raw_state[self.learning_agent_ids[0]][-1]["internal_data"]["holdings"]
            prev_mm_holdings = int(self.state[self.learning_agent_ids[0]][4*self.quote_history+7,0]) if self.state is not None else mm_holdings
            # print('prev_mm_holds: ',prev_mm_holdings)
            # print('mm_holds: ',mm_holdings)
            # print('mm_action: ',mm_action)#,
            spread_pnl = 0
            for l in range(1, mm_action[1] + 2):
                spread_pnl += (0.5 * (mm_action[0] + 1) + l - 1) * (self.mm_exec_vols[l-1,0] + self.mm_exec_vols[l-1,1])
                # print(spread_pnl)
            # hedge_cost = 0#np.abs(int(prev_mm_holdings * mm_action[2] * 0.2)) * spreads[-1] / 2
            inventory_pnl = (mid_prices[-1] - mid_prices[-2]) * prev_mm_holdings
            rewards[self.learning_agent_ids[0]] = spread_pnl + inventory_pnl 
            norm_rewards[self.learning_agent_ids[0]] = spread_pnl/self.Z_reward[self.learning_agent_ids[0]]['Spread PnL']\
                                                        + inventory_pnl/self.Z_reward[self.learning_agent_ids[0]]['Inventory PnL']
            self.spread_pnl = spread_pnl
            self.inventory_pnl = inventory_pnl
            # self.hedge_cost = hedge_cost
            # print('inv pnl: ',inventory_pnl)
            # print('spread pnl: ',spread_pnl)
            # print('hedge cost: ',hedge_cost)

            # Reward for PTs
            for i in range(self.num_pts):
                pt_holdings = next_raw_state[self.learning_agent_ids[1 + i]][-1]["internal_data"]["holdings"]
                prev_hold = self.state[self.learning_agent_ids[1 + i]][4*self.quote_history+6,0] if self.state is not None else pt_holdings
                hold = self.state[self.learning_agent_ids[1 + i]][4*self.quote_history+7,0] if self.state is not None else pt_holdings
                prev_cash = self.state[self.learning_agent_ids[1 + i]][4*self.quote_history+8,0] if self.state is not None else self.starting_cash
                cash = self.state[self.learning_agent_ids[1 + i]][4*self.quote_history+9,0] if self.state is not None else self.starting_cash
                if self.add_volume[self.learning_agent_ids[1 + i]]:
                    prev_price = self.trades[-2,2]
                    price = self.trades[-1,2]
                else:
                    prev_price = (self.quotes[-2,0] + self.quotes[-2,1]) / 2
                    price = (self.quotes[-1,0] + self.quotes[-1,1]) / 2
                rewards[self.learning_agent_ids[1 + i]] = cash + hold * price - prev_cash - prev_hold * prev_price
                norm_rewards[self.learning_agent_ids[1 + i]] = rewards[self.learning_agent_ids[1 + i]]/self.Z_reward[self.learning_agent_ids[1 + i]]
            # print(self.quotes)
            # print('pt_action: ',action[self.learning_agent_ids[1]])
            # print(f'ht_1: {prev_hold}, ht: {hold}, ct_1: {prev_cash}, ct: {cash}, pt_1: {prev_price}, pt: {price}')
            # print(f'ct+p_th_t: {cash + hold * price}, ct_1+pt_1ht_1: {prev_cash + prev_hold * prev_price}')
            # print(f'rt: {rewards[self.learning_agent_ids[1]]}')
            # print('pt_holdings: ',pt_holdings)
            # print('prev pt_hold: ',prev_pt_holdings)
            return norm_rewards, rewards

        elif self.reward_mode == "sparse":
            r = {}
            r[self.learning_agent_ids[0]] = 0
            for i in range(self.num_pts):
                r[self.learning_agent_ids[i + i]] = 0
            return r, r
    
    def raw_state_to_done(self, raw_state: Dict[str, Any], action: Dict) -> Dict:
        # print(action)
        dones = {} #True only at end of episode, done automatically with horizon
        for i in range(self.num_learning_agents):
            dones[self.learning_agent_ids[i]] = False
            # agent_holdings = raw_state[self.learning_agent_ids[i]][-1]["internal_data"]["holdings"]
            # agent_cash = raw_state[self.learning_agent_ids[i]][-1]["internal_data"]["cash"]
            # last_transaction = raw_state[self.learning_agent_ids[i]][-1]["parsed_mkt_data"][-1]["last_transaction"]
            # agent_marked_to_market = agent_cash + agent_holdings * last_transaction
            # dones[self.learning_agent_ids[i]] = False
            # if self.iter < self.done_iter[self.learning_agent_ids[i]]: #done for the first time?
            #     dones[self.learning_agent_ids[i]] = (agent_marked_to_market >= self.up_done_condition[self.learning_agent_ids[i]])
            #     if dones[self.learning_agent_ids[i]]:
            #         self.done_iter[self.learning_agent_ids[i]] = self.iter
            #         self.done_action[self.learning_agent_ids[i]] = action[self.learning_agent_ids[i]]
        dones['__all__'] = all(dones.values()) #or self.iter >= self.max_iter
        # print(self.iter,dones)
        
        # print(self.iter,dones)
        return dones 
    
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        info = {}
        # MM_rewards = np.zeros(3) #spread pnl, inv pnl, hedge cost
        # MM_rewards[0] = self.spread_pnl
        # MM_rewards[1] = self.inventory_pnl
        # MM_rewards[2] = self.hedge_cost
        # info[self.learning_agent_ids[0]] = MM_rewards
        return info

    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> bool:
        rewards = {}
        for i in range(self.num_learning_agents):
            rewards[self.learning_agent_ids[i]] = 0
        return rewards
    
    def normalize_state(self, state: Dict):
        norm_state = {}
        for i in range(self.num_learning_agents):
            norm_state[self.learning_agent_ids[i]] = np.divide(state[self.learning_agent_ids[i]],\
                                                        self.Z_state[self.learning_agent_ids[i]])
        # print(norm_state)
        norm_state = project_to_boundary(norm_state,self.observation_space,self.learning_agent_ids)
        # print(norm_state)
        return norm_state

    def get_agent_pnls(self):
        agent_pnls = np.zeros(self.num_noise_agents + self.num_value_agents + self.num_momentum_agents + 1 + self.num_pts) #25 + num_pts
        for i in range(self.num_noise_agents + self.num_value_agents + self.num_momentum_agents): #[0,..,19,20,21,22,23]
            agent = self.config_state["agents"][i + 1]
            agent_pnls[i] = agent.mark_to_market(agent.holdings) - agent.starting_cash
        offset = self.num_noise_agents + self.num_value_agents + self.num_momentum_agents # 24
        for i in range(1 + self.num_pts): #[0,..,num_pts]
            agent = self.gym_agent.agents[i]
            agent_pnls[i + offset] = agent.mark_to_market(agent.holdings) - agent.starting_cash
        # print('Avg Noise PnL: ',np.average(agent_pnls[0:self.num_noise_agents]))
        # print('Avg Value PnL: ',np.average(agent_pnls[self.num_noise_agents:self.num_noise_agents+self.num_value_agents]))
        # print('Avg Momentum PnL: ',np.average(agent_pnls[self.num_noise_agents+self.num_value_agents:self.num_noise_agents+self.num_value_agents+self.num_momentum_agents]))
        # print('MM PnL: ',agent_pnls[self.num_noise_agents+self.num_value_agents+self.num_momentum_agents+2])
        # print('Avg PT PnL: ',np.average(agent_pnls[-self.num_pts:]))
        return agent_pnls