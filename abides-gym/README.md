# gym_example

## New TODO list

### General things
- Clean proper version of the gym script 
- Clean proper version of the ray tune script
- Get a working version of the rollout
- Check and clean the MDP formulation for the environment 

### Core Environment
TODO: look at whether some cleaning functions needed for abides
Close function -> Do we implement some cleaning ? 

### Market Exec Env
raw_state_to_update_reward -> to be verified

### Market POV
Check the different asserts

### utils env
-> Should we put that in utilities in ABIDES-Markets ?
- todo: check if this works with all type of data: l1, l2, l3
- todo: test corner cases when liquidity drops: is it a None or an empty list ????
- todo: think if some should be function of the DIRECTION

### Financial GYM Agent 
Lookback period for volume subs, not sure what we should set 
- TODO: probably do something smarter for the lookback
- TODO: should be a parameter of the environment maybe


## TODO List 

### File Name:


Example implementation of an [OpenAI Gym](http://gym.openai.com/) environment,
to illustrate problem representation for [RLlib](https://rllib.io/) use cases.

## Usage

Clone the repo and connect into its top level directory.

To initialize and run the `gym` example:

```
pip install -r requirements.txt
pip install -e gym-example

python sample.py
```

To run Ray RLlib to train a policy based on this environment:

```
python train.py
```


## Kudos

h/t:

  - <https://github.com/DerwenAI/gym_trivial>
  - <https://github.com/DerwenAI/gym_projectile>
  - <https://github.com/apoddar573/Tic-Tac-Toe-Gym_Environment/>
  - <https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa>
  - <https://github.com/openai/gym/blob/master/docs/creating-environments.md>
  - <https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai>


# TODO List 

32 in total:
   - 5 in experimental agent folder
   - 24 in gim-abides folder 
   - 3 in runner folder
   
## experimental agent
- l156 # TODO: probaly will need to include what type of subscription in parameters here
        

## gym_runner 
- l11 #TODO: GOOD EXAMPLE OF HOW TO DO REGISTRATION : https://github.com/openai/gym-soccer
- l43     #todo: probably better way to register event: see the original medium
- l44 # todo: the mothod here was found: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py


## abides_gym ( the main one)
### test_config_making
- maybe no need to modify here -> will get rid 
### abides_example_env 
( I m not sure for all of them... I guess we need to see irectly the TODOs in the source code) 
- l55     #TODO: resolve why doesn't work with 0 lower bound
- l83 todo:all of this is hardcoded - will need further study
- l90 # TODO: Heads-up to be replaced by the new config_gym.py
- l121 todo: maybe need to change the function - not sure of the use?
- l159 todo: maybe need to change the function - not sure of the use?
- 163 # todo: not sure how we call it
- l 296 ##TODO: look at wheter some cleaning functions needed for abides
- l317 # todo: need to implements levels etc
- l460 # todo: handle empty book

In summary: 
- transformative state function - need to make sure it takes into account corner cases 
- need to figure out how the seed is fixed in a gym env 
- then how to pass this seed to abides config 
- need to refactor (formalize) the definition of the simulation env ( parameters/conf etc)
- need to solve the issue with the state space lower bound not working 


